"""
exp_daily.py

Experiment 1: single-day (day-ahead) comparison of the battery models on large
scenario sets, with training scenarios Xi generated from one month of data and
test scenarios Omega generated from the following month.

Usage:
  python exp_daily.py <season> scen                         (generate and cache the scenarios)
  python exp_daily.py <season> <model> [first_fold last_fold]
  python exp_daily.py <season> collect

season in {winter, spring, summer}; model in {NB, AB, PB, OB, MB, FB0, FB1, FB2, FB3}.
Results are written to results_daily/<season>_<model>[_fold<j>].npz.
"""
import os
import sys
import time
import json
import numpy as np
import core
from core import (T, DT, ScenarioModel, ab_periodic, emissions, simulate_affine, schedule_policy,
                  fb_policy, solve_pb, solve_ob, solve_fb)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_daily")
os.makedirs(RES, exist_ok=True)

SEASONS = {
    "winter": ("2026-01-01", "2026-01-31", "2026-02-01", "2026-02-28"),
    "spring": ("2026-03-12", "2026-04-12", "2026-04-13", "2026-05-13"),
    "summer": ("2026-06-15", "2026-07-15", "2026-07-16", "2026-08-15"),
}
N_TRAIN, N_TEST, K = 14_000, 10_000, 5
FOLD_SIZE = {"FB0": 100, "FB1": 200, "FB2": 500, "FB3": 500}
TIME_LIMIT = {"FB0": 2 * 3600, "FB1": 2 * 3600, "FB2": 3600, "FB3": 3600}
# Number of folds solved for every FB model: by default the folds cover the whole
# training set (140 for FB0, 70 for FB1, 28 for FB2 and FB3), as in the paper.
N_FOLDS = {k: int(os.environ.get("N_FOLDS", N_TRAIN // v)) for k, v in FOLD_SIZE.items()}


def scenarios(season):
    path = os.path.join(RES, f"scen_{season}.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return (z["lx"], z["sx"], z["ax"]), (z["lo"], z["so"], z["ao"])
    d = np.load(os.path.join(HERE, "data", "dataset.npz"))
    dates = d["dates"]
    a, b, c, e = SEASONS[season]
    tr = (dates >= a) & (dates <= b)
    te = (dates >= c) & (dates <= e)
    m_tr = ScenarioModel(d["load"][tr], d["solar"][tr], d["alpha"][tr], K=K)
    m_te = ScenarioModel(d["load"][te], d["solar"][te], d["alpha"][te], K=K)
    lx, sx, ax = m_tr.sample(N_TRAIN, np.random.default_rng(1))
    lo, so, ao = m_te.sample(N_TEST, np.random.default_rng(2))
    np.savez(path, lx=lx, sx=sx, ax=ax, lo=lo, so=so, ao=ao,
             expl_tr=m_tr.explained, expl_te=m_te.explained)
    return (lx, sx, ax), (lo, so, ao)


def evaluate_schedule(data, bp, bm, E0):
    l, s, a = data
    BP, BM, ET, proj = simulate_affine(l, s, schedule_policy(bp, bm), E0)
    e, viol = emissions(l, s, a, BP, BM, E0, ET)
    return e, proj, viol


def run(season, model, folds=None):
    Xi, Om = scenarios(season)
    out = os.path.join(RES, f"{season}_{model}.npz")
    t0 = time.time()
    if model == "NB":
        z = np.zeros_like(Xi[0])
        e_in, _ = emissions(*Xi, z, z)
        zo = np.zeros_like(Om[0])
        e_out, viol = emissions(*Om, zo, zo)
        np.savez(out, opt=e_in.mean(), time=0.0, e_out=e_out, proj=0.0, viol=viol.mean())
    elif model == "AB":
        res = []
        for (l, s, a) in (Xi, Om):
            bp, bm, E0, ET = ab_periodic(l, s)
            e, viol = emissions(l, s, a, bp, bm, E0, ET)
            res.append((e, viol))
        np.savez(out, opt=res[0][0].mean(), time=(time.time() - t0) / 2, e_out=res[1][0],
                 proj=0.0, viol=res[1][1].mean())
    elif model == "PB":
        r = solve_pb(*Xi)
        e, proj, viol = evaluate_schedule(Om, r["bp"], r["bm"], r["E"][0])
        np.savez(out, opt=r["obj"], time=r["time"], e_out=e, proj=proj.mean(), viol=viol.mean(),
                 bp=r["bp"], bm=r["bm"], E=r["E"])
    elif model == "OB":
        r_tr = solve_ob(*Xi)
        r_te = solve_ob(*Om)
        np.savez(os.path.join(RES, f"{season}_OB_train.npz"), bp=r_tr["bp"].mean(0), bm=r_tr["bm"].mean(0),
                 E0=r_tr["E"][:, 0].mean(), obj=r_tr["obj"], time=r_tr["time"])
        np.savez(out, opt=r_tr["obj"].mean(), time=r_tr["time"], e_out=r_te["obj"], proj=0.0, viol=0.0)
    elif model == "MB":
        z = np.load(os.path.join(RES, f"{season}_OB_train.npz"))
        e_in, _, _ = evaluate_schedule(Xi, z["bp"], z["bm"], float(z["E0"]))
        e, proj, viol = evaluate_schedule(Om, z["bp"], z["bm"], float(z["E0"]))
        np.savez(out, opt=e_in.mean(), time=float(z["time"]), e_out=e, proj=proj.mean(), viol=viol.mean())
    elif model.startswith("FB"):
        run_fb_folds(season, model, Xi, folds)


def run_fb_folds(season, model, Xi, folds):
    n = FOLD_SIZE[model]
    nf = N_FOLDS[model]
    first, last = folds if folds else (0, nf - 1)
    last = min(last, nf - 1)
    l, s, a = Xi
    for j in range(first, last + 1):
        path = os.path.join(RES, f"{season}_{model}_fold{j}.npz")
        if os.path.isfile(path):
            continue
        sl = slice(j * n, (j + 1) * n)
        r = solve_fb(model, l[sl], s[sl], a[sl], time_limit=TIME_LIMIT[model])
        if r is None:
            np.savez(path, ok=False)
            continue
        rest = np.ones(N_TRAIN, bool); rest[sl] = False
        pol = fb_policy(model, r["coef"])
        BP, BM, ET, proj = simulate_affine(l[rest], s[rest], pol, r["B0"])
        e, _ = emissions(l[rest], s[rest], a[rest], BP, BM, r["B0"], ET)
        np.savez(path, ok=True, obj=r["obj"], time=r["time"], status=r["status"], cv=e.mean(),
                 cv_proj=proj.mean(), B0=r["B0"], **{k: v for k, v in r["coef"].items()})
        print(f"{season} {model} fold {j}: obj {r['obj']:.2f} cv {e.mean():.2f} proj {proj.mean():.3f} "
              f"time {r['time']:.0f}s status {r['status']}", flush=True)


def collect(season):
    Xi, Om = scenarios(season)
    rows = {}
    for model in ("NB", "OB", "AB", "PB", "MB"):
        p = os.path.join(RES, f"{season}_{model}.npz")
        if os.path.isfile(p):
            z = np.load(p)
            rows[model] = dict(opt=float(z["opt"]), time=float(z["time"]), out=float(z["e_out"].mean()),
                               std=float(z["e_out"].std()), proj=float(z["proj"]), viol=float(z["viol"]))
    for model in ("FB0", "FB1", "FB2", "FB3"):
        nf = N_FOLDS[model]
        fz = [os.path.join(RES, f"{season}_{model}_fold{j}.npz") for j in range(nf)]
        fz = [np.load(p) for p in fz if os.path.isfile(p)]
        fz = [z for z in fz if bool(z["ok"])]
        if len(fz) < nf:
            print(f"{season} {model}: {len(fz)}/{nf} folds available")
        if not fz:
            continue
        best = min(fz, key=lambda z: float(z["cv"]))
        coef = {k: best[k] for k in ("lam_p", "lam_m", "phi_p", "phi_m", "psi_p", "psi_m")}
        l, s, a = Om
        BP, BM, ET, proj = simulate_affine(l, s, fb_policy(model, coef), float(best["B0"]))
        e, viol = emissions(l, s, a, BP, BM, float(best["B0"]), ET)
        np.save(os.path.join(RES, f"{season}_{model}_eout.npy"), e)
        rows[model] = dict(opt=float(best["obj"]), time=max(float(z["time"]) for z in fz), out=float(e.mean()),
                           std=float(e.std()), proj=float(proj.mean()), viol=float(viol.mean()),
                           cv=float(best["cv"]), folds=len(fz),
                           not_opt=sum(int(z["status"]) != 2 for z in fz))
    ob = rows.get("OB", {}).get("out")
    nb = rows.get("NB", {}).get("out")
    for k, r in rows.items():
        if ob:
            r["gap_ob"] = 100 * (r["out"] / ob - 1)
        if nb:
            r["red_nb"] = 100 * (1 - r["out"] / nb)
    with open(os.path.join(RES, f"{season}_summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for k, r in rows.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})


if __name__ == "__main__":
    season, model = sys.argv[1], sys.argv[2]
    if model == "scen":
        scenarios(season)
    elif model == "collect":
        collect(season)
    else:
        folds = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else None
        run(season, model, folds)
