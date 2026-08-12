import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD, lpDot, LpBinary

# Start timing
start_time = time.time()

def plot_exp_vs_cvar(results, scheme, num_betas_disp=4):
    if not isinstance(results, list):
        results = [results]

    exp = []
    cvar = []
    betas = []
    for result in results:
        exp.append(result["Expected Profit"])
        cvar.append(result["CVaR"])
        betas.append(result["Beta"])

    _, ax = plt.subplots(figsize=(10, 5)) 

    # 1. Plot the line
    ax.plot(cvar, exp, color='black')

    # 2. Select points evenly spaced
    if num_betas_disp < len(cvar):
        indices = np.linspace(0, len(cvar) - 1, num_betas_disp, dtype=int)
        selected_cvar = np.array(cvar)[indices]
        selected_exp = np.array(exp)[indices]
        selected_betas = np.array(betas)[indices]
    else:
        selected_cvar = cvar
        selected_exp = exp
        selected_betas = betas

    # 3. Plot only the selected points with markers
    ax.plot(selected_cvar, selected_exp, 'o', color='black', markersize=8)

    # 4. Add floating labels
    for idx, (x, y) in enumerate(zip(selected_cvar, selected_exp)):
        offset_x = 0.01 * (max(selected_cvar) - min(selected_cvar))  # 1% of cvar range
        offset_y = 0.01 * (max(selected_exp) - min(selected_exp))    # 1% of exp range
        ax.text(x + offset_x, y + offset_y, f"$\\beta$ = {selected_betas[idx]:.2f}", fontsize=12, ha='left', va='bottom')
        
    # 5. Expand axis limits to make room for text
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min, x_max + 0.1 * (x_max - x_min))  # expand x-axis 10%
    ax.set_ylim(y_min, y_max + 0.05 * (y_max - y_min))  # expand y-axis 5%

    # Plot formatting
    ax.set_title(f"Expected Profit vs CVaR ({scheme})", fontsize=16)
    ax.set_xlabel("CVaR [€]", fontsize=14)
    ax.set_ylabel("Expected Profit [€]", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_outcomes_vs_beta(results_one, results_two):
    if not isinstance(results_one, list):
        results_one = [results_one]
    if not isinstance(results_two, list):
        results_two = [results_two]

    # First set of results
    betas = []
    profits = []
    p_DA = []
    for result in results_one:
        betas.append(result["Beta"])
        profits.append(result["Expected Profit"])
        p_DA.append(result["Optimal Offer"])

    profits = np.array(profits)
    profits = profits / 1000

    # Second set of results
    betas_two = []
    profits_two = []
    p_DA_two = []
    for result in results_two:
        betas_two.append(result["Beta"])
        profits_two.append(result["Expected Profit"])
        p_DA_two.append(result["Optimal Offer"])

    profits_two = np.array(profits_two)
    profits_two = profits_two / 1000

    # --- Plot Offer Outcomes for each beta ---
    offers = []
    for offer in p_DA:
        offers.append(sum(offer) / 24)

    offers_two = []
    for offer in p_DA_two:
        offers_two.append(sum(offer) / 24)

    _, ax = plt.subplots(1, figsize=(8, 5))
    ax.plot(betas, offers, color='black', label="One-price Scheme")
    ax.plot(betas_two, offers_two, color='blue', label="Two-price Scheme")

    ax.set_title("Average offers for different $\\beta$", fontsize=16)
    ax.set_xlabel("$\\beta$", fontsize=14)
    ax.set_ylabel("Average Offer [MW]", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # --- Plot Expected Profits in thousands of EUR for each beta ---
    _, ax = plt.subplots(1, figsize=(8, 5))
    ax.plot(betas, profits, color='black', label="One-price Scheme")
    ax.plot(betas_two, profits_two, color='blue', label="Two-price Scheme")

    ax.set_title("Expected Profits for different $\\beta$", fontsize=16)
    ax.set_xlabel("$\\beta$", fontsize=14)
    ax.set_ylabel("Expected Profit [k€]", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_pdf(results, scheme, num_betas_disp=3):
    if not isinstance(results, list):
        results = [results]
    results = np.array(results)

    n_betas = len(results)

    # Create a subset of betas
    if n_betas > num_betas_disp:
        betas_subset = np.linspace(0, n_betas - 1, num_betas_disp, dtype=int)
    else:
        betas_subset = np.arange(0, n_betas, 1)

    results = results[betas_subset]

    _, ax = plt.subplots(figsize=(10, 6))

    # Colormap for distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for idx, result in enumerate(results):
        profits = np.array(result["Scenario Profits"])
        profits_kEUR = profits / 1000
        ax.hist(
            profits_kEUR,
            bins=20,
            #density=True,  # Normalize to probability
            histtype='step',  # Step histogram
            color=colors[idx],
            label=f'$\\beta$ = {result["Beta"]:.2f}',
            linewidth=2
        )

    # Plot formatting
    ax.set_title(f"Probability Distribution of Profits ({scheme})", fontsize=16)
    ax.set_xlabel("Profit [k€]", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(title="Scenarios", fontsize=10, title_fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def prepare_sample_scenarios(W, price_sc_filename, wind_sc_file_name, sys_status_sc):
    """Prepare random sample scenarios.

    Args:
        W (int): Number of scenarios to sample.
        price_sc_filename (str): File with price scenarios.
        wind_sc_file_name (str): File with wind power production scenarios.
        sys_status_sc (list): System status scenarios (generated before).

    Returns:
        tuple: DA forecasted prices for each sampled scenario; power forecast; system status.
    """
    T = 24

    DA_price_all = pd.read_excel(price_sc_filename).iloc[:, 1:].to_numpy()
    p_real_all = pd.read_excel(wind_sc_file_name).iloc[:, 1:].to_numpy() # Total wind generation forecast in Finland (reference wind power forecast)
    p_real_all = capacity * p_real_all / max_wind_finland # Share of given capacity in the total wind capacity in Finland

    # Create ALL scenario combinations (20 x 20 x 4 = 1600) based on the indices
    # (wind power production scenario, DA price scenario, system status scenario)
    all_combinations = [(i_w, i_p, i_s)
                        for i_w in range(20)
                        for i_p in range(20)
                        for i_s in range(4)]

    # Randomly select (W) combinations
    sampled_combinations = random.sample(all_combinations, W)
    # sampled_combinations = all_combinations  # Use all 1600 combinations
    out_of_sample_combinations = list(set(all_combinations) - set(sampled_combinations))

    # Preparing arrays for lambda_DA, p_real and sys_status
    lambda_DA = np.empty((W,T))
    p_real = np.empty((W,T))
    sys_status = np.ones((W,T))

    # Unpack scenarios data for lambda_DA, p_real and sys_status (filling previous arrays)
    for idx, (i_w, i_p, i_s) in enumerate(sampled_combinations):
        for t in range(T):
            lambda_DA[idx, t] = DA_price_all[t, i_p]
            p_real[idx, t] = p_real_all[t, i_w]
            sys_status[idx, t] = sys_status_sc[t, i_s]

    return lambda_DA, p_real, sys_status

def solve_offering_strategy_oneprice(capacity, lambda_DA, p_real, sys_status, alpha=0.9, beta=0):
    """
    Args:
        capacity (float): Nominal wind farm capacity [MW].
        lambda_DA (numpy.ndarray): DA Market Price forecast for each scenario (over 24h) [EUR/MWh].
        p_real (numpy.ndarray): Wind Power Production forecast for each scenario (over 24h) [MW].
        sys_status (numpy.ndarray): System Status for each scenario (over 24h).
        alpha (float): Confidence level. Value between 0 and 1, excluding 1. Default is 0.9.
        beta (float): CVaR weighting factor (0-1). Default is 0 - risk-neutral offering strategy.

    Returns:
        dict: "Objective" [EUR], "Optimal Offer" [MW], "Expected Profit" [EUR], "CVaR" [EUR],
            "Scenario Profits" [EUR].
    """
    # Number of sampled scenarios
    W = len(lambda_DA)
    T = 24

    model = LpProblem(name="maximize-profit", sense=LpMaximize)
    p_DA = [LpVariable(f"p_DA_{t}", lowBound=0, upBound=capacity) for t in range(T)] # Generation volume DA [MW]
    I_B = [[LpVariable(f"I_B_{w}_{t}", lowBound=None) for t in range(T)] for w in range(W)] # Imbalance cost/profit [EUR]
    delta_t_w = [[LpVariable(f"delta_{w}_{t}", lowBound=None) for t in range(T)] for w in range(W)] # Wind farm imbalance [MW]
    eta_w = [LpVariable(f"eta_{w}", lowBound=0) for w in range(W)] # Auxiliary variable for CVaR [EUR]
    zeta = LpVariable("zeta", lowBound=None) # VaR [EUR]

    # Scenario Weighting Factors (assuming equiprobable scenarios)
    pi_w = (np.ones(W) / W).tolist()

    # Array for balancing prices
    balancing_price = np.empty((W, T))

    for w in range(W):
        for t in range(T):
            # Balancing price
            if sys_status[w, t] == 0: # If power deficit
                balancing_price[w, t] = lambda_DA[w, t] * 1.25
            else:
                balancing_price[w, t] = lambda_DA[w, t] * 0.85

            model += delta_t_w[w][t] == p_real[w, t] - p_DA[t], f"imbalance_def_{w}_{t}" # Constraint 1
            model += (I_B[w][t] == balancing_price[w][t] * delta_t_w[w][t], f"Imbalance_cost_{w}_{t}") # Constraint 2

    # Constraint 3: Constraint for zeta
    for w in range(W):
        expr = lpSum(
            lambda_DA[w, t] * p_DA[t]
            + I_B[w][t]
            for t in range(T)
        )
        model += -expr + zeta - eta_w[w] <= 0, f"dual_feasibility_{w}"

    CVaR = zeta - (1/(1 - alpha)) * lpDot(eta_w, pi_w)

    # Expected Profit formula
    expected_profit = lpSum(
        pi_w[w] * (
            lambda_DA[w, t] * p_DA[t]
            + I_B[w][t]
        )
        for w in range(W)
        for t in range(T)
    )

    # Main objective function
    objective = (1-beta) * expected_profit + beta * CVaR

    # Run the solver for objective function maximization
    model += objective, "Objective"
    model.solve(PULP_CBC_CMD(msg=False))

    objective_value = value(model.objective)
    p_DA_opt = [value(var) for var in p_DA]

    # Calculating Profits by scenario
    profit_w = []
    for w in range(W):
        profit_n = lpSum(
            lambda_DA[w, t] * p_DA[t]
            + I_B[w][t]
        for t in range(T)
        )

        profit_w.append(value(profit_n))

    return {
        "Objective": objective_value,
        "Optimal Offer": p_DA_opt,
        "Expected Profit": value(expected_profit),
        "CVaR": value(CVaR),
        "Scenario Profits": profit_w,
        "Beta": beta
    }

def solve_offering_strategy_twoprice(capacity, lambda_DA, p_real, sys_status, alpha=0.9, beta=0):
    """
    Args:
        capacity (float): Nominal wind farm capacity [MW].
        lambda_DA (numpy.ndarray): DA Market Price forecast for each scenario (over 24h) [EUR/MWh].
        p_real (numpy.ndarray): Wind Power Production forecast for each scenario (over 24h) [MW].
        sys_status (numpy.ndarray): System Status for each scenario (over 24h).
        alpha (float): Confidence level. Value between 0 and 1, excluding 1. Default is 0.9.
        beta (float): CVaR weighting factor (0-1). Default is 0 - risk-neutral offering strategy.

    Returns:
        dict: "Objective" [EUR], "Optimal Offer" [MW], "Expected Profit" [EUR], "CVaR" [EUR],
            "Scenario Profits" [EUR].
    """
    # Number of sampled scenarios
    W = len(lambda_DA)
    T = 24
    M = capacity

    model = LpProblem(name="maximize-profit", sense=LpMaximize)
    p_DA = [LpVariable(f"p_DA_{t}", lowBound=0, upBound=capacity) for t in range(T)] # Generation volume DA [MW]
    delta_t_w = [[LpVariable(f"delta_{w}_{t}", lowBound=None) for t in range(T)] for w in range(W)] # Wind farm imbalance [MW]
    delta_t_w_up = [[LpVariable(f"delta_up_{t}_{w}", lowBound=0) for t in range(T)] for w in range(W)] # Power excess [MW]
    delta_t_w_down = [[LpVariable(f"delta_down_{t}_{w}", lowBound=0) for t in range(T)] for w in range(W)] # Power deficit [MW]
    binary_up = [[LpVariable(f"binary_up_{w}_{t}", cat=LpBinary) for t in range(T)] for w in range(W)] # Binary variable if delta is positive or negative
    eta_w = [LpVariable(f"eta_{w}", lowBound=0) for w in range(W)] # Auxiliary variable for CVaR [EUR]
    zeta = LpVariable("zeta", lowBound=None) # VaR [EUR]

    # Scenario Weighting Factors (assuming equiprobable scenarios)
    pi_w = (np.ones(W) / W).tolist()

    CVaR = zeta - (1/(1 - alpha)) * lpDot(eta_w, pi_w)

    # Expected Profit formula
    expected_profit = lpSum(
        pi_w[w] * (
            lambda_DA[w, t] * p_DA[t]
            + 0.85 * lambda_DA[w, t] * delta_t_w_up[w][t] * sys_status[w, t]
            + lambda_DA[w, t] * delta_t_w_up[w][t] * (1 - sys_status[w, t])
            - 1.25 * lambda_DA[w, t] * delta_t_w_down[w][t] * (1 - sys_status[w, t])
            - lambda_DA[w, t] * delta_t_w_down[w][t] * sys_status[w, t]
        )
        for w in range(W)
        for t in range(T)
    )

    # Main objective function
    objective = (1-beta) * expected_profit + beta * CVaR

    for w in range(W):
        for t in range(T):
            model += delta_t_w[w][t] == p_real[w, t] - p_DA[t], f"imbalance_def_{w}_{t}" # Constraint 1: delta_t_w = p_real - p_DA
            model += delta_t_w[w][t] == delta_t_w_up[w][t] - delta_t_w_down[w][t], f"delta_split_{w}_{t}" # Constraint 2: delta_t_w = delta_t_w_up - delta_t_w_down
            model += delta_t_w_up[w][t] <= M * binary_up[w][t] # Additional constraint: Positive excess 
            model += delta_t_w_down[w][t] <= M*(1 - binary_up[w][t]) # Additional constraint: Positive deficit

    # Constraint 3: Constraint for zeta
    for w in range(W):
        expr = lpSum(
            lambda_DA[w, t] * p_DA[t]
            + 0.85 * lambda_DA[w, t] * delta_t_w_up[w][t] * sys_status[w, t]
            + lambda_DA[w, t] * delta_t_w_up[w][t] * (1 - sys_status[w, t])
            - 1.25 * lambda_DA[w, t] * delta_t_w_down[w][t] * (1 - sys_status[w, t])
            - lambda_DA[w, t] * delta_t_w_down[w][t] * sys_status[w, t]
            for t in range(T)
        )
        model += -expr + zeta - eta_w[w] <= 0, f"dual_feasibility_{w}"

    # Run the solver for objective function maximization
    model += objective, "Objective"
    model.solve(PULP_CBC_CMD(msg=False))

    objective_value = value(model.objective)
    p_DA_opt = [value(var) for var in p_DA]

    # Calculating Profits by scenario
    profit_w = []
    for w in range(W):
        profit_n = lpSum(
            lambda_DA[w, t] * p_DA[t]
            + 0.85 * lambda_DA[w, t] * delta_t_w_up[w][t] * sys_status[w, t]
            + lambda_DA[w, t] * delta_t_w_up[w][t] * (1 - sys_status[w, t])
            - 1.25 * lambda_DA[w, t] * delta_t_w_down[w][t] * (1 - sys_status[w, t])
            - lambda_DA[w, t] * delta_t_w_down[w][t] * sys_status[w, t]
            for t in range(T)
        )

        profit_w.append(value(profit_n))

    return {
        "Objective": objective_value,
        "Optimal Offer": p_DA_opt,
        "Expected Profit": value(expected_profit),
        "CVaR": value(CVaR),
        "Scenario Profits": profit_w,
        "Beta": beta
    }

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Parameters
T = 24                      # Hours
capacity = 500              # Maximum wind farm capacity [MW]
max_wind_finland = 8358     # Reference max wind capacity in Finland [MW]
sys_scenarios = 4           # Number of system status scenarios
# Generate system status scenarios (random 0/1 for 4 scenarios)
system_status_all = np.random.binomial(n=1, p=0.5, size=(T, sys_scenarios))

num_scenarios = 200         # Number of SCENARIOS TO SAMPLE

# Load and sample scenario data
price_sc_filename = "Price_Scenarios.xlsx"
wind_sc_filename = "Wind_Power_Scenarios.xlsx"
lambda_DA, p_real, sys_status = prepare_sample_scenarios(num_scenarios, price_sc_filename, wind_sc_filename, system_status_all)

# Factors for risk-averse offering strategy
step = 10                         # Number of steps (more = slower compute!)
betas = np.linspace(0, 1, step+1)  # Beta from 0 to 1, spaced evenly

# Run the optimization for every beta
results_one = []
for beta in betas:
    result = solve_offering_strategy_oneprice(capacity, lambda_DA, p_real, sys_status, beta=beta)
    results_one.append(result)
results_two = []
for beta in betas:
    result = solve_offering_strategy_twoprice(capacity, lambda_DA, p_real, sys_status, beta=beta)
    results_two.append(result)

# Run the optimization for different number of in-sample scenarios
beta = 0.5
num_scenarios = [5, 50, 100, 200, 300, 400]    # Number of scenarios in a list (more = slower compute!)
for W in num_scenarios:
    lambda_DA, p_real, sys_status = prepare_sample_scenarios(W, price_sc_filename, wind_sc_filename, system_status_all)
    result = solve_offering_strategy_oneprice(capacity, lambda_DA, p_real, sys_status, beta=beta)
    print("Average Daily Offer for W =",len(result["Scenario Profits"]), "is", round(sum(result["Optimal Offer"])/T,2), "MW. (one-price scheme)")
    result = solve_offering_strategy_twoprice(capacity, lambda_DA, p_real, sys_status, beta=beta)
    print("Average Daily Offer for W =",len(result["Scenario Profits"]), "is", round(sum(result["Optimal Offer"])/T,2), "MW. (two-price scheme)")

# End timing
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} s")

# Plot Expected Profit vs CVaR
plot_exp_vs_cvar(results_one, "One-price Scheme", num_betas_disp=3)
plot_exp_vs_cvar(results_two, "Two-price Scheme")

# Plot offer outcomes and Expected profit vs Betas
plot_outcomes_vs_beta(results_one, results_two)

# Plot profit distribution
plot_pdf(results_one, "One-price Scheme")
plot_pdf(results_two, "Two-price Scheme")
