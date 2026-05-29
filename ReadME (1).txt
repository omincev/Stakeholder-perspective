Renewable Energy Markets - Assignment 2

Quick-start guide
This code requires Python to be installed, along with the numpy, matplotlib, pandas, time, gurobipy, and pulp packages.

To get started, set the working directory to the folder containing the code files. The following code files are included: - 1.1oneprice.py - 1.2twoprice.py - 1.3one-price_expost.py - 1.3two-price_expost.py - Task_1.4.py - part2.1and2.2alsox.py - part2.1and2.2cvar.py - part2.3.py 

The working directory should also include the following file: - p2data.xlsx: 300 load profiles for 60 minutes [kW]
The working directory should also include the following Excel files with case scenario data: "Price_Scenarios.xlsx" and "Wind_Power_Scenarios.xlsx" 

The general principle of the code is as follows: for each step, the main function takes the appropriate parameters, solves the linear problem, and outputs the results.
The input data in the excel files come from either online sources or are generated.

A more detailed description of the functioning of the code is shown in the pseudocode below.


1.1oneprice.py

This script is written to solve the offering strategy for a wind power producer participating in the electricity market under a one-price imbalance settlement scheme. It uses a risk-aware profit maximization model that accounts for both the Day-Ahead market revenue and imbalance settlement, incorporating a CVaR-based objective function. The goal is to find an optimal Day-Ahead bid profile that maximizes a weighted combination of expected profit and CVaR across multiple scenarios.

	Step 1: Import data and initialize parameters
		Load scenario data for Day-Ahead prices and wind power forecasts from Excel files.
		Simulate random system status scenarios (indicating power surplus or deficit).
		Define key parameters such as time horizon, wind farm capacity, and number of scenarios.
	Step 2: Generate in-sample scenarios
		Randomly sample a subset of possible scenario combinations (price, wind, and system status) to be used for optimization.
		Prepare data arrays for each scenario input.
	Step 3: Define and solve optimization model
		Formulate a linear programming model to maximize a weighted sum of expected profit and Conditional Value at Risk (CVaR).
		Include constraints for power balance, imbalance revenue/loss, and risk metrics. Solve the model using a standard LP solver.
	Step 4: Extract and analyze results
		Retrieve the optimal Day-Ahead bid per hour and calculate expected profit and profit per scenario.
		Present the results in a clear table showing hourly bid and expected earnings.
	Step 5: Plot results
		Generate visualizations including:
		Optimal bid profile over 24 hours
		Profit distribution across scenarios
		Cumulative distribution (CDF) of scenario profits

1.2twoprice.py

In Task 1.2, a two-price market offering strategy is optimized using a stochastic model that maximizes expected profit while managing risk via CVaR. Based on sampled price, wind, and system status scenarios, the model computes optimal hourly bids and evaluates profit distributions, enabling robust decision-making under market uncertainty.

	Step 1: Import data and initialize parameters
		Load forecasted wind power and Day-Ahead (DA) price scenarios from Excel files.
		Generate system status scenarios (power surplus or deficit).
		Set key parameters such as wind farm capacity, number of scenarios to sample, and number of hours (24).
	Step 2: Generate in-sample scenarios
		Randomly create combinations of wind, price, and system status scenarios (e.g., 20×20×4 = 1600 total).
		Randomly sample a subset of these to be used for optimization.
		Prepare three arrays: DA prices, wind generation forecasts, and system statuses per scenario.
	Step 3: Define and solve optimization model
		Formulate a risk-aware optimization problem for offering in a two-price imbalance settlement market.
		Define variables for DA bids, imbalance volumes, and binary up/down indicators.
		Include constraints for power imbalance, directional imbalances, and CVaR.
		The objective is a weighted combination of expected profit and CVaR. Solve using the PULP solver.
	Step 4: Extract and analyze results
		Obtain the optimal DA offer for each hour and compute expected profit, CVaR, and scenario-wise profit values.
		Construct and print a decision table showing expected bid and profit per hour. Calculate and print the total expected profit.
	Step 5: Plot results
		Generate the following plots:
		Optimal DA bid profile across 24 hours
		Profit per scenario (distribution plot)
		Cumulative Distribution Function (CDF) of scenario profits

1.3one-price_expost.py

Task 1.3 has two codes. In this code, a risk-aware day-ahead bidding strategy is optimized for a wind power producer using a stochastic model with Conditional Value at Risk (CVaR). The model integrates sampled scenarios of electricity prices, wind power generation, and system status to determine optimal hourly bids that maximize expected profit while controlling downside risk. Results are validated using an out-of-sample evaluation and saved for multiple cross-validation folds.

	Step 1: Import data and initialize parameters  
    		Load Day-Ahead price and wind power scenarios from Excel files.  
   		Generate all scenario combinations for wind, price, and system status.  
    		Set system parameters like wind farm capacity, scenario counts, risk level (alpha), and timing.  
	Step 2: Prepare scenario data  
    		For each sampled scenario subset, extract corresponding price, wind power, and system status data arrays.  
    		Normalize wind power by farm capacity relative to maximum historical wind.  
	Step 3: Formulate and solve optimization model  
    		Create a linear program maximizing expected profit under CVaR risk constraints.  
    		Define decision variables for hourly DA bids, imbalances, and risk parameters.  
    		Use scenario probabilities and balancing prices depending on system status to calculate revenues and penalties.  
    		Solve the model with the PULP solver to find optimal hourly bids.  
	Step 4: Evaluate profit out-of-sample  
    		Apply the optimized bids to out-of-sample scenarios to compute realized profits and estimate the 10% CVaR of losses.  
	Step 5: Save and summarize results  
    		Store optimal bids and profit distributions for each fold in Excel sheets.  
    		Aggregate and export summary statistics including average in-sample and out-of-sample profits and CVaR risk metrics.  
    		Print execution time for performance tracking.  

1.3two-price_expost.py

In this code, a two-price imbalance settlement market strategy is optimized for a wind power producer by formulating a mixed-integer linear program that maximizes expected profit under Conditional Value at Risk (CVaR) constraints. The model incorporates hourly day-ahead bids and explicitly models positive and negative imbalance volumes with binary variables to represent imbalance directions, adapting imbalance prices depending on system status. Results are cross-validated over multiple folds and saved for detailed out-of-sample performance evaluation.
 
	Step 1: Import data and initialize parameters  
    		Load Day-Ahead electricity price and wind power generation scenarios from Excel files.  
    		Generate all combinations of wind, price, and system status scenarios.  
    		Set key parameters such as wind farm capacity, scenario counts, risk aversion level (alpha), and timing.  
	Step 2: Prepare scenario data  
    		Extract and normalize scenario data subsets (price, wind power, system status) for in-sample and out-of-sample evaluation.  
	Step 3: Define and solve two-price optimization model  
    		Formulate a mixed-integer linear program maximizing expected profit and CVaR under two-price imbalance settlement rules.  
    		Define decision variables for hourly day-ahead bids, imbalance volumes split into upward and downward components,
		and binary variables for imbalance direction.  
    		Calculate revenues and penalties according to scenario-dependent imbalance prices and system status.  
    		Solve the model with the PULP CBC solver to find optimal bids.
	Step 4: Evaluate out-of-sample profits  
    		Apply optimized bids to out-of-sample scenarios to compute realized profits,
		explicitly considering positive/negative imbalances and corresponding two-price rules.  
    		Calculate the 10% CVaR of the out-of-sample profits for risk assessment.  
	Step 5: Save and summarize results  
    		Export optimal bids and detailed profit distributions per fold into Excel sheets.  
    		Aggregate fold-wise performance metrics including average profits and CVaR, and save to a summary sheet.  
    		Track and print progress with runtime information.  

Task_1.4.py

This script is written to solve the optimization problem using the CVaR technique. It solves optimization problem on selected in-sample scenarios for one-price and two-price schemes for different beta (CVaR weighting factor) values. It presents impact of beta factor on risk-averse offering strategy outcomes (expected profits, offer volume, distribution of profits). Then it runs optimization problem for one beta value on different number of in-sample scenarios to analyse changes in the risk-averse offering decisions.

	Step 1: Import data and sample scenarios
	Step 2: Define factors for risk-averse offering strategy 
	Step 3: Run the optimization for every beta
	Step 4: Run the optimization for different number of in-sample scenarios
	Step 5: Plot results


part2.1and2.2cvar.py

This script is written to solve the optimization problem using the CVaR technique. It starts by solving the optimization problem to find the optimal value for c for the in sample scenarios. It the compares the result value for c with the out of sample scenarios and generates a percentage value for the amount of minutes that the load is not met.

	Step 1: Import data
	Step 2: Constrain data to in sample scenarios, set up model and constraints
	Step 3: Solve the model and print results
	Step 4: Constrains data to out of sample scenarios and calculate how many minutes the load is not met
	Step 5: Plots

part2.1and2.2alsox.py

This script is written to solve the optimization problem using the ALSO - X technique. It starts by solving the optimization problem to find the optimal value for c for the in sample scenarios. It the compares the result value for c with the out of sample scenarios and generates a percentage value for the amount of minutes that the load is not met.

	Step 1: Import data
	Step 2: Constrain data to in sample scenarios, set up model and constraints
	Step 3: Solve the model and print results
	Step 4: Constrains data to out of sample scenarios and calculate how many minutes the load is not met
	Step 5: Plots

part2.3.py

This script varies the value q (the budget for load not being met) in the ALSO - X optimization. It takes the in sample scenarios and runs the optimization problem 10 times. q = (number of minutes x number of scenarios x epsilon) where epsilon is the percentage of times the load can not be met. The optimization is then run in a loop where epsilon starts a 0 and is increased my 0.02 every time the loop is run and the results are stored. 

	Step 1: Import data
	Step 2: Constrain data to in sample and out of sample
	Step 3: Set open arrays to collect data from model results and set values for epsilon
	Step 4: Run the model 10 times for increasing values of epsilon
	Step 5L Plots