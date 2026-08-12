# Wind Farm Trading & Risk Optimization

A Python toolkit for optimal offering strategies and risk management of a wind power producer participating in a two-settlement electricity market. It formulates the producer's day-ahead bidding decision as a stochastic optimization problem and evaluates the trade-off between expected profit and financial risk using Conditional Value-at-Risk (CVaR).

## Overview

A wind farm must commit an energy offer to the day-ahead market before its actual production and the balancing price are known. This toolkit builds stochastic programming models that decide the optimal day-ahead offer across a large set of price/production/system-state scenarios, under both one-price and two-price balancing market designs. It then analyzes how the optimal strategy and profit change as the producer becomes more risk-averse.

The project covers:

- One-price balancing market offering strategy
- Two-price balancing market offering strategy
- Out-of-sample (ex-post) validation via cross-validation over scenario folds
- Risk-aware offering using a CVaR term with a tunable risk-aversion weight
- A reserve-capacity (P90) sizing model solved with both an ALSO-X and a CVaR formulation

## Method

Each offering model maximizes expected profit across scenarios subject to market and capacity constraints, using a linear / mixed-integer program. Risk aversion is introduced by adding a CVaR term to the objective, weighted by a factor that sweeps the producer from a risk-neutral to a strongly risk-averse strategy, tracing an efficient frontier of expected profit versus CVaR. Separate reserve-sizing models determine the largest capacity bid that still satisfies a probabilistic (P90) reliability requirement, comparing a chance-constrained ALSO-X approach with a CVaR approach.

## Repository Structure

| File | Description |
| --- | --- |
| `one_price_strategy.py` | Optimal day-ahead offer under a one-price balancing market. |
| `two_price_strategy.py` | Optimal day-ahead offer under a two-price balancing market. |
| `one_price_expost.py` | Out-of-sample (ex-post) validation of the one-price strategy. |
| `two_price_expost.py` | Out-of-sample (ex-post) validation of the two-price strategy. |
| `risk_analysis.py` | Expected-profit vs. CVaR trade-off analysis across risk-aversion levels. |
| `reserve_alsox.py` | Reserve-capacity (P90) sizing via the ALSO-X formulation. |
| `reserve_cvar.py` | Reserve-capacity (P90) sizing via the CVaR formulation. |
| `reserve_out_of_sample.py` | Out-of-sample evaluation of the reserve-sizing model. |
| `Price_Scenarios.xlsx` | Day-ahead price scenarios. |
| `Wind_Power_Scenarios.xlsx` | Wind power production scenarios. |
| `data.xlsx` | Load/measurement profiles used by the reserve-sizing models. |

## Requirements

- Python 3.9+
- `numpy`
- `pandas`
- `matplotlib`
- `pulp`
- `gurobipy` (for the ALSO-X reserve model)
- `tqdm`

Install the open-source dependencies with:

```bash
pip install numpy pandas matplotlib pulp tqdm
```

Note: `reserve_alsox.py` uses Gurobi via `gurobipy` and requires a valid Gurobi license. The CVaR variants run on the open-source PuLP/CBC solver.

## Usage

Run any script from the project root so the Excel scenario files resolve correctly. For example:

```bash
python one_price_strategy.py
```

Each script loads its scenario data, builds and solves the corresponding optimization model, and reports the optimal offer, expected profit, CVaR, and related plots.
