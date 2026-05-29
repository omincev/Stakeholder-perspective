import pandas as pd
import numpy as np
import random
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD, lpDot
import matplotlib.pyplot as plt
import os

# -------------------- CONFIG --------------------
capacity = 500
max_wind_finland = 8358
T = 24
total_scenarios = 1600
in_sample_size = 200
folds = 8
alpha = 0.9
random.seed(42)
np.random.seed(42)

output_dir = "expost_results"
os.makedirs(output_dir, exist_ok=True)

# -------------------- LOAD DATA --------------------
price_data = pd.read_excel("Price_Scenarios.xlsx").iloc[:, 1:].to_numpy()
wind_data = pd.read_excel("Wind_Power_Scenarios.xlsx").iloc[:, 1:].to_numpy()
all_combinations = [(i_w, i_p, i_s) for i_w in range(20) for i_p in range(20) for i_s in range(4)]
system_status_all = np.random.binomial(n=1, p=0.5, size=(T, 4))
random.shuffle(all_combinations)

# -------------------- FUNCTIONS --------------------
def get_scenarios(combos):
    W = len(combos)
    lambda_DA = np.zeros((W, T))
    p_real = np.zeros((W, T))
    sys_status = np.zeros((W, T))
    for idx, (i_w, i_p, i_s) in enumerate(combos):
        lambda_DA[idx, :] = price_data[:, i_p]
        p_real[idx, :] = wind_data[:, i_w]
        sys_status[idx, :] = system_status_all[:, i_s]
    p_real *= capacity / max_wind_finland
    return lambda_DA, p_real, sys_status

def solve_offering_strategy(lambda_DA, p_real, sys_status, alpha=0.9):
    W, T = lambda_DA.shape
    model = LpProblem(name="maximize-profit", sense=LpMaximize)

    p_DA = [LpVariable(f"p_DA_{t}", lowBound=0, upBound=capacity) for t in range(T)]
    I_B = [[LpVariable(f"I_B_{w}_{t}") for t in range(T)] for w in range(W)]
    delta = [[LpVariable(f"delta_{w}_{t}") for t in range(T)] for w in range(W)]
    eta = [LpVariable(f"eta_{w}", lowBound=0) for w in range(W)]
    zeta = LpVariable("zeta")

    pi_w = np.ones(W) / W
    balancing_price = np.where(sys_status == 0, lambda_DA * 1.25, lambda_DA * 0.85)

    for w in range(W):
        for t in range(T):
            model += delta[w][t] == p_real[w, t] - p_DA[t]
            model += I_B[w][t] == balancing_price[w, t] * delta[w][t]
        model += -lpSum(lambda_DA[w, t] * p_DA[t] + I_B[w][t] for t in range(T)) + zeta - eta[w] <= 0

    expected_profit = lpSum(
        pi_w[w] * lpSum(lambda_DA[w, t] * p_DA[t] + I_B[w][t] for t in range(T))
        for w in range(W)
    )
    CVaR = zeta - (1 / (1 - alpha)) * lpDot(eta, pi_w)
    model += expected_profit, "Objective"
    model.solve(PULP_CBC_CMD(msg=0))

    return np.array([value(p_DA[t]) for t in range(T)]), value(expected_profit)

def evaluate_profit(p_DA, lambda_DA, p_real, sys_status):
    balancing_price = np.where(sys_status == 0, lambda_DA * 1.25, lambda_DA * 0.85)
    imbalance = p_real - p_DA
    da_revenue = lambda_DA * p_DA
    imbalance_profit = imbalance * balancing_price
    return np.sum(da_revenue + imbalance_profit, axis=1)

# -------------------- MAIN LOOP --------------------
results = []
excel_path = os.path.join(output_dir, "expost1_all_results.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for fold in range(folds):
        fold_start = fold * in_sample_size
        fold_end = fold_start + in_sample_size
        in_sample = all_combinations[fold_start:fold_end]
        out_of_sample = all_combinations[:fold_start] + all_combinations[fold_end:]

        lambda_in, p_real_in, sys_in = get_scenarios(in_sample)
        lambda_out, p_real_out, sys_out = get_scenarios(out_of_sample)

        p_DA_opt, profit_in = solve_offering_strategy(lambda_in, p_real_in, sys_in)
        profit_out = evaluate_profit(p_DA_opt, lambda_out, p_real_out, sys_out)
        CVaR_10 = np.mean(np.sort(profit_out)[:int(0.1 * len(profit_out))])

        # Save to Excel
        bid_df = pd.DataFrame({"Hour": np.arange(1, T+1), "Optimal Bid [MW]": p_DA_opt})
        profits_df = pd.DataFrame({"Scenario": np.arange(1, len(profit_out)+1), "Profit [EUR]": profit_out})
        bid_df.to_excel(writer, sheet_name=f"Fold_{fold+1}_Bid", index=False)
        profits_df.to_excel(writer, sheet_name=f"Fold_{fold+1}_Profits", index=False)

        results.append({
            "Fold": fold + 1,
            "In-sample Profit [EUR]": profit_in,
            "Out-of-sample Profit [EUR]": np.mean(profit_out),
            "Out-of-sample CVaR (10%) [EUR]": CVaR_10
        })

    # Final summary sheet
    df_summary = pd.DataFrame(results)
    df_summary.to_excel(writer, sheet_name="Summary", index=False)

print(f"\n✅ All results saved to {excel_path}")

# -------------------- PLOT CDFs --------------------
plt.figure(figsize=(10, 6))
for fold in range(folds):
    out_set = all_combinations[:fold*in_sample_size] + all_combinations[(fold+1)*in_sample_size:]
    lambda_out, p_real_out, sys_out = get_scenarios(out_set)
    p_DA_opt, _ = solve_offering_strategy(*get_scenarios(all_combinations[fold*in_sample_size:(fold+1)*in_sample_size]))
    profits = evaluate_profit(p_DA_opt, lambda_out, p_real_out, sys_out)
    sorted_p = np.sort(profits)
    cdf = np.arange(1, len(sorted_p)+1) / len(sorted_p)
    plt.step(sorted_p, cdf, label=f"Fold {fold+1}")

plt.xlabel("Profit [EUR]")
plt.ylabel("Cumulative Probability")
plt.title("Ex-Post Profit CDFs for Each Fold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "expost_cdf_all_folds.png"), dpi=300)
plt.show()
