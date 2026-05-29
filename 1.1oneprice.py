import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD, lpDot, LpBinary
import time

# Start timing
start_time = time.time()

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
    zeta = LpVariable(f"zeta", lowBound=None) # VaR [EUR]

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

    # Constraint 3 (risk-averse): Constraint for zeta
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

def plot_outcomes(result):
    p_DA = result["Optimal Offer"]
    T = np.arange(1,25,1)
    # --- Plot Offer Outcomes ---

    _, ax = plt.subplots(1, figsize=(8, 5))
    ax.plot(T, p_DA, color='black')

    ax.set_title(f"Offering outcomes (one-price scheme)", fontsize=16)
    ax.set_xlabel(f"Hour", fontsize=14)
    ax.set_ylabel("Offer [MW]", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

random.seed(42)
np.random.seed(42)

T = 24                      # Hours
capacity = 500              # Maximum wind farm capacity [MW]
max_wind_finland = 8358     # Reference max wind capacity in Finland [MW]
sys_scenarios = 4           # Number of system status scenarios
system_status_all = np.random.binomial(n=1, p=0.5, size=(T, sys_scenarios)) # Random system status

num_scenarios = 200         # Number of SCENARIOS TO SAMPLE

# Load and sample scenario data
price_sc_filename = "Price_Scenarios.xlsx"
wind_sc_filename = "Wind_Power_Scenarios.xlsx"
lambda_DA, p_real, sys_status = prepare_sample_scenarios(num_scenarios, price_sc_filename, wind_sc_filename, system_status_all)

result = solve_offering_strategy_oneprice(capacity, lambda_DA, p_real, sys_status)
plot_outcomes(result)

# End timing
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} s")

# Extract results
p_DA = result["Optimal Offer"]
W, T = lambda_DA.shape
balancing_price = np.where(sys_status == 0, lambda_DA * 1.25, lambda_DA * 0.85)

# Compute expected hourly profits
hourly_profits = np.zeros((W, T))
for w in range(W):
    for t in range(T):
        imbalance = p_real[w, t] - p_DA[t]
        imbalance_profit = balancing_price[w, t] * imbalance
        da_revenue = lambda_DA[w, t] * p_DA[t]
        hourly_profits[w, t] = da_revenue + imbalance_profit

expected_profit_hourly = np.mean(hourly_profits, axis=0)


# Create table
table_df = pd.DataFrame({
    "Hour": np.arange(0, 24),
    "Expected Bid [MW]": p_DA,
    "Expected Profit [EUR]": expected_profit_hourly
})

# Print formatted table
print("Final Decision: Expected Day-Ahead Bid and Profit per Hour")
print("-----------------------------------------------------------------")
print("Hour | Expected Bid [MW] | Expected Profit [EUR]")
print("-----------------------------------------------------------------")
for _, row in table_df.iterrows():
    print(f"{int(row['Hour']):>4} | {row['Expected Bid [MW]']:.2f}{' ':>10} | {row['Expected Profit [EUR]']:.2f}")

print("\nTotal Expected Profit (Objective Function Value): {:.2f} EUR".format(result["Expected Profit"]))

#graph: Profit Distribution over Scenarios
# Create DataFrame of profits per scenario
df_profits = pd.DataFrame({
    "Scenario": np.arange(1, len(result["Scenario Profits"]) + 1),
    "Profit": result["Scenario Profits"]
})

# Plot profit distribution over scenarios
plt.figure(figsize=(8, 5)) 
plt.plot(df_profits["Scenario"], df_profits["Profit"], label="Profit per Scenario", color='deepskyblue')
plt.xlabel("Scenario", fontsize=12)
plt.ylabel("Profit [EUR]", fontsize=12)
plt.title("Profit Distribution over Scenarios (one-price)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Profit_Distribution_OnePrice.png", dpi=300)
plt.show()


# Plot cumulative distribution (CDF) of scenario profits
sorted_profits = np.sort(result["Scenario Profits"])
cdf = np.arange(1, len(sorted_profits)+1) / len(sorted_profits)

plt.figure(figsize=(8, 5))
plt.plot(sorted_profits, cdf, drawstyle='steps-post', color='darkorange')
plt.xlabel("Profit [EUR]", fontsize=12)
plt.ylabel("Cumulative Probability", fontsize=12)
plt.title("Cumulative Distribution of Profits (CDF) one-price", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("Profit_CDF_OnePrice.png", dpi=300)
plt.show()