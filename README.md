# Linear battery models

Code for the paper

> A. Sacilotto, A. M. Bernardelli, S. Gualandi.
> *Emissions and electricity costs of residential PV-battery systems under
> self-consumption and predictive control.*

An earlier version, based on synthetic household profiles, is on arXiv
([2606.26648](https://arxiv.org/abs/2606.26648)); its code is in the history of
this repository (commit `09ebb99`).

The code compares the self-consumption rule of a residential PV-battery system
with optimization-based policies (deterministic and two-stage model predictive
control, and a hybrid policy in which the rule executes setpoints from an hourly
linear program), for three objectives: emissions of imported electricity with
average or marginal emission factors, and retail cost. All optimization models
are linear programs solved with Gurobi.

## Installation

Python ≥ 3.10, [uv](https://docs.astral.sh/uv/) and a Gurobi licence (free for
academic use; gurobipy 12 or 13).

```
uv sync
uv run pytest        # 18 tests
```

## Data

All input data are public.

| Data | Source | Scripts |
|---|---|---|
| Household loads, 15 min, 2023–2024 | Swiss smart-meter dataset, Zenodo [10.5281/zenodo.21398768](https://doi.org/10.5281/zenodo.21398768) (CC BY 4.0) | `fetch_loads.py`, `build_loads.py` |
| Italian generation mix and load, hourly; day-ahead price IT-North | [Energy-Charts](https://www.energy-charts.info) (ENTSO-E, SMARD) | `fetch_grid.py`, `build_grid.py` |
| Satellite irradiance, temperature, archived day-ahead weather forecasts | [Open-Meteo](https://open-meteo.com) (CC BY 4.0) | `fetch_weather.py`, `build_pv.py` |
| PV check | PVGIS 5.3 | `fetch_weather.py`, `build_pv.py` |

`data/` contains the processed inputs used in the paper, except the household
loads (65 MB), which must be rebuilt once (about 1 GB download):

```
uv run python code/fetch_loads.py && uv run python code/build_loads.py   # data/loads.npz, data/homes.csv
```

The other files can be rebuilt with `fetch_grid.py`/`build_grid.py` and
`fetch_weather.py`/`build_pv.py`; since the sources are revised over time, the
results may then differ slightly from the paper.

| File | Content |
|---|---|
| `data/grid.npz` | average and marginal emission factors and day-ahead price, 15 min, 2023–2024 |
| `data/pv.npz` | actual and forecast PV generation per kWp at Milan, Rome and Palermo |
| `data/mef_table.csv` | marginal emission factors by year, quarter and hour |
| `data/homes.csv` | households of the data set with type, data coverage and annual load; the 794 used have `selected = True` |
| `data/pv_validation.csv` | comparison of the PV model with PVGIS |
| `results/cases.csv` | the 20 households and configurations of the closed-loop comparison |

## Code

| File | Content |
|---|---|
| `code/lb/system.py` | battery, objectives, tariff, evaluation of a trajectory |
| `code/lb/rules.py` | self-consumption rule, with the setpoints of the hybrid policies |
| `code/lb/lp.py` | perfect-foresight LP and receding-horizon LPs (deterministic and two-stage) |
| `code/lb/scenarios.py` | PCA scenario model conditioned on observations and weather forecasts |
| `code/lb/data.py` | loading of the data and assembly of a case |
| `code/exp_headroom.py` | E1: rule vs perfect foresight for every household and configuration |
| `code/exp_sensitivity.py` | E1 sensitivity: efficiency, battery emissions, export accounting |
| `code/exp_closed_loop.py` | E2: closed-loop simulation of one case with one policy |
| `code/make_jobs.py`, `code/run_jobs.py` | job lists and a restartable parallel runner |
| `code/summarize.py`, `make_tables.py`, `key_numbers.py`, `plot_*.py` | tables, numbers and figures of the paper |

Policy names in the code and in the paper:

| Code | AB | EV-RT | 2S-RT | 2S-RT-nofc | 2S-RH | OR-RT | HY-EV-1H | HY-EV-DA | HY-OR-1H | PF | NB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper | SC | MPC-D | MPC-S | MPC-S0 | MPC-SL | MPC-P | HYB-1h | HYB-24h | HYB-P | PF | NB |

## Reproducing the results

A single run, e.g. the hybrid policy for one household with the cost objective
over the first two days:

```
uv run python code/exp_closed_loop.py --home 172262 --obj cost --policy HY-EV-1H --days 2
```

The full experiments (one job per line, run in parallel by `run_jobs.py` or by
a cluster scheduler; set `GRB_THREADS`, default 1, for the Gurobi threads per job):

```
uv run python code/make_jobs.py headroom        # E1: 794 households, ~20 min per shard of 4 households
uv run python code/run_jobs.py --jobs jobs/headroom.txt --workers 32
uv run python code/make_jobs.py sensitivity
uv run python code/run_jobs.py --jobs jobs/sensitivity.txt --workers 32
uv run python code/make_jobs.py closed-loop     # E2: 287 runs, uses results/cases.csv
uv run python code/run_jobs.py --jobs jobs/closed_loop2.txt --workers 32
```

`make_jobs.py cases` recomputes `results/cases.csv` from the results of E1.
On one core, a closed-loop run over the 318 days takes seconds for the rule
and the hybrid policies, about one minute for MPC-D and one to two hours for
the two-stage policies (up to about 2 h 15 min). All runs are restartable: E1
skips households already done and E2 resumes from daily checkpoints. Random
numbers use fixed seeds.

Results go to `results/headroom/`, `results/sensitivity/` and `results/cl2/`;
then

```
uv run python code/summarize.py
uv run python code/make_tables.py
uv run python code/key_numbers.py
cd code && uv run python plot_headroom.py && uv run python plot_closed_loop.py \
  && uv run python plot_day.py --date 2024-10-24 --case 180146_milan_pv4.5_b10-5_mef
```

writes the summaries to `results/summary/` and the figures to `figures/`.

## Licence

MIT (code). The data in `data/` are derived from the sources above and keep
their licences.
