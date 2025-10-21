# Linear Battery Models

Python implementation of linear optimization and rule-based models for the operation of photovoltaic (PV) battery systems under uncertainty.  
The models were developed for the study *Comparative Analysis of Linear Battery Models for Carbon Emission Optimization in Solar Energy Systems*.

---

## Files

| File | Description |
|------|--------------|
| `scenario_generation.py` | PCA-based generation of synthetic load, solar, and emission scenarios. |
| `self_consumption.py` | Rule-based self-consumption algorithm (Automatic Battery model). |
| `battery_models.py` | Linear optimization models: OB (Omniscient Battery), PB (Programmed Battery), and feedback-based variants FB0–FB2. |

---

## Requirements

- Python ≥ 3.9  
- NumPy  
- Pandas  
- Gurobi Optimizer ≥ 10.0  

