# Technical Notes

This document records the model formulations, parameters, and data assumptions used across the scripts. See `README.md` for a high-level overview and usage instructions.

## Problem Setting

A wind power producer must submit a day-ahead (DA) energy offer before its actual production and the balancing price are realized. Each script solves an optimization model over a set of scenarios describing DA price, wind production, and system balancing state, and reports the optimal offer together with the resulting expected profit and risk.

## Offering Models (one- and two-price)

Files: `one_price_strategy.py`, `two_price_strategy.py`, `one_price_expost.py`, `two_price_expost.py`, `risk_analysis.py`

The core routine (`solve_offering_strategy_oneprice` / `solve_offering_strategy_twoprice`) takes:

- `capacity`  nominal wind farm capacity [MW]
- `lambda_DA`  DA market price per scenario over 24 h [EUR/MWh]
- `p_real`  realized wind production per scenario over 24 h [MW]
- `sys_status`  binary system balancing state per scenario
- `alpha`  CVaR confidence level (default 0.9)
- `beta`  CVaR weighting factor in the objective (default 0 = risk-neutral)

The model maximizes expected profit across scenarios subject to the offer being bounded by capacity, with balancing revenue defined by the one-price or two-price settlement rule. A CVaR term weighted by `beta` is added to the objective; sweeping `beta` from 0 upward traces the expected-profit vs. CVaR efficient frontier (produced by `risk_analysis.py`).

The `*_expost.py` scripts perform out-of-sample validation: the scenario set is split into in-sample and out-of-sample folds via cross-validation, the offer is optimized in-sample, and its performance is re-evaluated on the held-out scenarios.

## Reserve Capacity (P90) Sizing

Files: `reserve_alsox.py`, `reserve_cvar.py`, `reserve_out_of_sample.py`

These models size the largest reserve capacity bid `c_up` that still meets a P90 reliability requirement (epsilon = 0.1, i.e. 90% compliance) against a set of measured load/flexibility profiles. Two equivalent formulations are provided:

- ALSO-X: a chance-constrained MILP with a Big-M / binary indicator per (minute, scenario) pair, solved with Gurobi.
- CVaR: a convex relaxation using per-scenario slack variables and a VaR threshold, solved with the open-source PuLP/CBC solver.

`reserve_out_of_sample.py` sweeps the violation budget and evaluates the resulting capacity on out-of-sample profiles, reporting mean and total shortfall.

## Default Parameters

- Wind farm capacity: 500 MW
- Reference max wind (Finland): 8358 MW
- Time horizon: 24 hours (offering) / 60 minutes (reserve)
- Total scenarios: 1600, in-sample size 200, 8 folds
- Random seed: 42 (fixed for reproducibility)

## Data Files

- `Price_Scenarios.xlsx`  day-ahead price scenarios
- `Wind_Power_Scenarios.xlsx`  wind power production scenarios
- `p2data.xlsx`  load/flexibility profiles (300 profiles x 60 minutes) used by the reserve-sizing models

## Solvers

Most models run on the open-source PuLP/CBC solver. `reserve_alsox.py` uses Gurobi (`gurobipy`) and requires a valid Gurobi license.
