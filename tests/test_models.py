"""Tests of the rule, the LPs, the scenario model and the evaluation (synthetic data)."""

import numpy as np
import pytest

from lb.lp import RecedingLP, solve_pf
from lb.rules import rule_step, run_rule, setpoints_from_plan
from lb.scenarios import ScenarioModel
from lb.system import DT, TARIFFS, Battery, T, Tariff, evaluate

BAT = Battery(capacity=10.0, power=5.0)


def synthetic(n_days, seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(T) / 4
    pv_shape = np.clip(np.sin(np.pi * (h - 6) / 12), 0, None)
    load = 0.3 + 0.7 * np.exp(-((h - 20) ** 2) / 4) + 0.4 * np.exp(-((h - 8) ** 2) / 2)
    L = np.clip(load + 0.2 * rng.normal(size=(n_days, T)), 0.05, None)
    S = 4.0 * pv_shape * rng.uniform(0.2, 1.0, size=(n_days, 1))
    X = 250 - 120 * pv_shape + 20 * rng.normal(size=(n_days, T))
    return L, S, np.clip(X, 50, None)


def test_rule_is_self_consumption():
    L, S, _ = synthetic(3)
    bp, bm, E = run_rule(L.ravel(), S.ravel(), BAT, 5.0)
    net = (L - S).ravel()
    assert (E >= -1e-9).all() and (E <= BAT.capacity + 1e-9).all()
    assert (bp * bm == 0).all()
    # never charges more than the surplus, never discharges more than the deficit
    assert (bm <= np.maximum(-net, 0) + 1e-9).all() and (bp <= np.maximum(net, 0) + 1e-9).all()
    # SOC dynamics
    assert np.allclose(E[1:], E[:-1] + DT * (BAT.eta * bm - bp / BAT.mu))
    # daily cycle budget
    thr = DT * (bp / BAT.mu + BAT.eta * bm).reshape(3, T).sum(1)
    assert (thr <= BAT.budget + 1e-9).all()


def test_rule_setpoints():
    # floor: no discharge below the floor; target: grid charging towards the target
    p, m = rule_step(5.0, 2.0, 0.0, BAT, floor=5.0)
    assert p == 0 and m == 0
    p, m = rule_step(3.0, 1.0, 0.0, BAT, target=4.0)
    assert p == 0 and m == pytest.approx(min(BAT.power, 1.0 / (BAT.eta * DT)))
    assert rule_step(5.0, 1.0, 0.0, BAT) == rule_step(5.0, 1.0, 0.0, BAT, 0.0, 0.0, np.inf)
    # dump: with a deficit of 1 kW the battery discharges towards the dump level, exporting the excess
    p, m = rule_step(5.0, 1.0, 0.0, BAT, dump=4.0)
    assert m == 0 and p == pytest.approx(min(BAT.mu * 1.0 / DT, BAT.mu * BAT.power))


@pytest.mark.parametrize("obj", ["aef", "cost"])
def test_pf_is_lower_bound(obj):
    L, S, X = synthetic(4, seed=1)
    tar = TARIFFS[obj]
    if obj == "cost":
        X = X * 0.5  # EUR/MWh-like values
    ld, pv, x = L.ravel(), S.ravel(), X.ravel()
    bp, bm, E = run_rule(ld, pv, BAT, 5.0)
    ab = evaluate(ld, pv, x, bp, bm, tar, BAT, 5.0, E[-1])["value"]
    sol = solve_pf(ld, pv, x, tar, BAT, 5.0)
    pf = evaluate(ld, pv, x, sol["bp"], sol["bm"], tar, BAT, 5.0, sol["E"][-1])["value"]
    assert pf == pytest.approx(sol["obj"], rel=1e-6)
    assert pf <= ab + 1e-6
    nb = evaluate(ld, pv, x, 0 * ld, 0 * ld, tar, BAT, 0, 0)["value"]
    assert ab <= nb


def test_receding_two_stage_matches_det_with_identical_scenarios():
    L, S, X = synthetic(1, seed=2)
    tar = TARIFFS["aef"]
    det = RecedingLP("det", 1, BAT, tar, eps=0.0)
    r1 = det.solve(L, S, X, 4.0, BAT.budget, 0)
    assert np.allclose(r1["E"][1:], r1["E"][:-1] + DT * (BAT.eta * r1["bm"] - r1["bp"] / BAT.mu), atol=1e-6)
    two = RecedingLP("2s", 5, BAT, tar, eps=0.0)
    r2 = two.solve(np.repeat(L, 5, 0), np.repeat(S, 5, 0), np.repeat(X, 5, 0), 4.0, BAT.budget, 0)
    assert 0 <= r2["bp0"] <= BAT.mu * BAT.power + 1e-9 and 0 <= r2["bm0"] <= BAT.power + 1e-9
    # identical scenarios: the two-stage model is the deterministic one
    assert two.m.ObjVal == pytest.approx(det.m.ObjVal, rel=1e-7)
    # with E0 = 0 the one-day window equals the perfect-foresight LP on that day
    det.solve(L, S, X, 0.0, BAT.budget, 0)
    pf = solve_pf(L.ravel(), S.ravel(), X.ravel(), tar, BAT, 0.0)
    assert det.m.ObjVal == pytest.approx(pf["obj"], rel=1e-6)


def test_receding_cycle_budget_split():
    # with no budget left today, nothing can happen before midnight
    L, S, X = synthetic(1, seed=3)
    lp = RecedingLP("det", 1, BAT, TARIFFS["aef"], eps=0.0)
    r = lp.solve(L, S, X, 5.0, 0.0, 40)
    assert np.allclose(r["bp"][: T - 40], 0, atol=1e-7) and np.allclose(r["bm"][: T - 40], 0, atol=1e-7)


def test_scenario_model_conditioning():
    L, S, X = synthetic(28, seed=4)
    sm = ScenarioModel([L, S, X], log_blocks=(2,), K=4, caps={0: (0, None), 1: (0, 4.0)})
    rng = np.random.default_rng(0)
    out = sm.sample(50, rng, {0: L[0, :10], 1: S[0, :10], 2: X[0]})
    assert all(o.shape == (50, T) for o in out)
    assert np.allclose(out[0][:, :10], L[0, :10]) and np.allclose(out[2], X[0])
    assert (out[1] >= 0).all() and (out[1] <= 4.0).all() and (out[0] >= 0).all()
    # PV is zero where it is never observed
    night = S.max(0) <= 1e-9
    assert np.allclose(out[1][:, night], 0)
    # conditioning on the whole day reduces the spread of the unobserved block
    free = sm.sample(500, np.random.default_rng(1))
    cond = sm.sample(500, np.random.default_rng(1), {0: L[3], 2: X[3]})
    assert cond[1].std(0).mean() < free[1].std(0).mean()


def test_setpoints_from_plan():
    net_hat = np.array([1.0, 1.0, -1.0, 1.0])
    bp = np.array([1.0, 0.0, 0.0, 0.5])
    bm = np.array([0.0, 0.0, 2.0, 0.0])
    E = np.array([5.0, 4.0, 4.0, 5.5, 5.0])
    sp = setpoints_from_plan(bp, bm, E, net_hat)
    assert list(sp["floor"]) == [0.0, 4.0, 0.0, 5.0]  # hold at t=1, partial discharge at t=3
    assert list(sp["target"]) == [0.0, 0.0, 5.5, 0.0]  # grid charging at t=2
    assert list(sp["dump"]) == [np.inf] * 4
    assert list(sp["ceiling"]) == [np.inf] * 4
    sp = setpoints_from_plan(
        np.array([2.0, 0.0]), np.array([0.0, 0.5]), np.array([5.0, 3.0, 3.4]), np.array([1.0, -2.0])
    )
    assert sp["dump"][0] == 3.0  # discharge beyond the deficit: export
    assert sp["ceiling"][1] == 3.4  # store less than the surplus: export PV


def test_hybrid_with_exact_plan_reproduces_plan():
    # a plan computed with perfect information, executed by the rule on the same data
    L, S, X = synthetic(1, seed=5)
    tar = TARIFFS["cost"]
    X = 60 + 80 * np.sin(np.arange(T) / T * 2 * np.pi)  # prices with a morning/evening pattern
    r = RecedingLP("det", 1, BAT, tar, eps=1e-7).solve(L, S, X[None], 3.0, BAT.budget, 0)
    sp = setpoints_from_plan(r["bp"], r["bm"], r["E"], (L - S)[0])
    bp, bm, E = run_rule(L[0], S[0], BAT, 3.0, setpoints=sp)
    plan = evaluate(L[0], S[0], X, r["bp"], r["bm"], tar, BAT, 0, 0)["value"]
    rule = evaluate(L[0], S[0], X, bp, bm, tar, BAT, 0, 0)["value"]
    assert rule == pytest.approx(plan, abs=1e-3)


def test_evaluate_terminal_and_export_credit():
    tar = Tariff("t", "x", k_buy=1.0, c_buy=0.0, k_sell=0.5, wear=0.0)
    ld, pv, x = np.array([1.0, 0.0]), np.array([0.0, 2.0]), np.array([100.0, 100.0])
    z = np.zeros(2)
    r = evaluate(ld, pv, x, z, z, tar, BAT, E0=2.0, E_end=1.0)
    assert r["value"] == pytest.approx(DT * (100 - 0.5 * 100 * 2) + 1.0 * 100 / BAT.eta)


@pytest.mark.parametrize("kind", ["det", "2s"])
def test_receding_budget_is_per_calendar_day(kind):
    # strong arbitrage signal and a small budget: the plan must respect the budget of
    # each calendar day separately, without carrying today's unused budget to tomorrow
    bat = Battery(capacity=10.0, power=5.0, cycles=0.2)  # budget 4 kWh per day
    M, tau = (1 if kind == "det" else 3), 60
    L = np.full((M, T), 1.0)
    S = np.zeros((M, T))
    X = np.tile(np.where(np.arange(T) % 8 < 4, 50.0, 500.0), (M, 1))
    lp = RecedingLP(kind, M, bat, Tariff("t", "x"), eps=0.0)
    lp.solve(L, S, X, 5.0, bat.budget, tau)
    bp, bm = np.atleast_2d(lp.bp.X), np.atleast_2d(lp.bm.X)
    thr = DT * (bp / bat.mu + bat.eta * bm)
    today, tomorrow = thr[:, : T - tau].sum(axis=1), thr[:, T - tau :].sum(axis=1)
    assert (today <= bat.budget + 1e-6).all() and (tomorrow <= bat.budget + 1e-6).all()
    assert tomorrow.max() > 0.5 * bat.budget  # the budget of tomorrow is used
    # with no budget left today, tomorrow still has its own
    lp.solve(L, S, X, 5.0, 0.0, tau)
    bp, bm = np.atleast_2d(lp.bp.X), np.atleast_2d(lp.bm.X)
    thr = DT * (bp / bat.mu + bat.eta * bm)
    assert np.allclose(thr[:, : T - tau], 0, atol=1e-7) and thr[:, T - tau :].sum(axis=1).max() > 0.5 * bat.budget


def synthetic_case(objective="aef", n_days=40):
    from lb.data import Case

    L, S, X = synthetic(n_days, seed=11)
    rng = np.random.default_rng(12)
    dates = np.array([str(d)[:10] for d in np.arange("2024-02-01", n_days, dtype="datetime64[D]")])
    return Case("h", "milan", 4.5, BAT, objective, dates, L, S, np.clip(S * rng.uniform(0.6, 1.4, S.shape), 0, 4.5), X)


@pytest.mark.parametrize("tau", [0, 10, 30, 60, 95])
def test_forecast_information_is_causal(tau):
    # changing forecast values that are not yet available at the decision time
    # must not change the scenarios (same random stream)
    from exp_closed_loop import Forecaster, fc_available

    day = 30
    a, b = synthetic_case(), synthetic_case()
    n_today, n_next = fc_available(tau)
    b.pv_fc[day, n_today:] += 1.0
    b.pv_fc[day + 1, n_next:] += 1.0
    b.pv_fc[day + 2 :] += 1.0
    wa = Forecaster(a, True).window(day, tau, 20, True, np.random.default_rng(0))
    wb = Forecaster(b, True).window(day, tau, 20, True, np.random.default_rng(0))
    assert all(np.array_equal(u, v) for u, v in zip(wa, wb, strict=True))
    # sanity: an available value does change them
    if n_today > 60 and tau <= 30:
        b.pv_fc[day, 50] += 1.0
        wc = Forecaster(b, True).window(day, tau, 20, True, np.random.default_rng(0))
        assert not np.array_equal(wa[1], wc[1])


def test_previous_year_signal():
    from lb.data import previous_year_signal

    dates = np.array([str(d) for d in np.arange("2023-01-01", "2025-01-01", dtype="datetime64[D]")])
    q = np.array([(int(d[5:7]) - 1) // 3 for d in dates])
    y = np.array([int(d[:4]) for d in dates])
    x = (100 * (y - 2023) + 10 * q)[:, None] + np.arange(T)[None, :]
    out = previous_year_signal(dates, x.astype(float))
    assert np.isnan(out[y == 2023]).all()
    assert np.array_equal(out[y == 2024], x[y == 2024] - 100)
