# Linear battery models

Code and data for the paper

> A. Sacilotto, A. M. Bernardelli, S. Gualandi.
> *Linear battery policies for emission-aware operation of residential PV systems:
> the role of information structure in day-ahead and receding-horizon control.*

The code compares battery policies that minimize the equivalent carbon emissions
of a residential photovoltaic (PV) system with a battery, under uncertainty in
household load, PV generation and grid carbon intensity. All optimization models
are linear programs solved with Gurobi.

## Contents

| File | Content |
|---|---|
| `core.py` | Parameters, PCA scenario model, rule-based policy, linear programs, simulator |
| `exp_daily.py` | Experiment 1: day-ahead models on 14,000 training and 10,000 test scenarios |
| `exp_closed_loop.py` | Experiment 2: closed-loop operation on real data, 1 Feb – 19 Oct 2026 |
| `data/dataset.npz` | Daily household profiles of load, PV generation and carbon intensity (15-min, 1 Jan – 19 Oct 2026) |
| `data/build_dataset.py` | Downloads the raw data and builds `dataset.npz` |

## Installation

Python ≥ 3.10 and a Gurobi license (free for academic use) are required.

```
pip install -r requirements.txt
```

## Usage

**Experiment 1** (`season` in `winter`, `spring`, `summer`):

```
python exp_daily.py spring scen        # generate the training and test scenarios
python exp_daily.py spring NB          # no battery
python exp_daily.py spring AB          # rule-based self-consumption
python exp_daily.py spring PB          # programmed battery
python exp_daily.py spring OB          # omniscient battery (run before MB)
python exp_daily.py spring MB          # mean battery
python exp_daily.py spring FB3         # feedback battery: FB0, FB1, FB2 or FB3
python exp_daily.py spring collect     # summary of the season
```

The feedback models are solved on disjoint folds of the training scenarios; a
range of folds can be given (e.g. `python exp_daily.py spring FB0 0 9`) to run
them in parallel.

**Experiment 2**:

```
python exp_closed_loop.py <policy>
```

with `<policy>` in `NB`, `AB`, `PF` (perfect foresight), `EV-RH`, `PB-RH`, `2S-RH`
(receding horizon with expected-value, scenario-based and two-stage models), their
real-time variants `EV-RT`, `PB-RT`, `2S-RT`, and the day-ahead policies `PB-DA`,
`FB2-DA`, `FB3-DA`. Simulations are checkpointed every day and resume if interrupted.
For the sensitivity analysis, the battery capacity (kWh), the battery power (kW) and
a scaling factor of the household load can be changed, e.g.
`python exp_closed_loop.py 2S-RT --bmax 5 --pb 2.5 --load 1.5`
(results in `results_cl_sens/`).

Results are written to `results_daily/` and `results_cl/`. All random numbers use
fixed seeds. The number of Gurobi threads per process is set by the environment
variable `GRB_THREADS` (default 1).

## Data

The profiles are derived from Italian national data (load, PV generation and
generation per production type) from [Energy-Charts](https://www.energy-charts.info)
(Fraunhofer ISE), which redistributes data of the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu). They are scaled
to a single household with 6.4 kWp of PV and about 6.5 MWh/year of consumption;
the carbon intensity is computed from the generation mix.

`data/dataset.npz` is the exact dataset used in the paper. Since the source data
are revised over time, running `python data/build_dataset.py` (after removing
`dataset.npz`) may produce slightly different values.

## License

The code is released under the MIT License (see `LICENSE`). The data are derived
from Energy-Charts / Fraunhofer ISE and ENTSO-E and are provided for
non-commercial research use with attribution to the original providers.
