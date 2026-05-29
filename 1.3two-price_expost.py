import pandas as pd
import numpy as np
import random
import os
import time
from tqdm import tqdm
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD, lpDot, LpBinary
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
T = 24
capacity = 500
max_wind_finland = 8358
alpha = 0.9
folds = 8
in_sample_size = 200
total_scenarios = 1600
output_dir = "twoprice_results"
os.makedirs(output_dir, exist_ok=True)

# ------------- DATA LOAD ----------------
price_data = pd.read_excel("Price_Scenarios.xlsx").iloc[:, 1:].to_numpy()
wind_data = pd.read_excel("Wind_Power_Scenarios.xlsx").iloc[:, 1:].to_numpy()
all_combinations = [(i_w, i_p, i_s) for i_w in range(20) for i_p in range(20) for i_s in range(4)]
system_status_all = np.random.binomial(n=1, p=0.5, size=(T, 4))
random.seed(42)
np.random.seed(42)
random.shuffle(all_combinations)

# ----------- FUNCTIONS ------------
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

def solve_offering_strategy_twoprice(capacity, lambda_DA, p_real, sys_status, alpha=0.9, beta=0):
    W = len(lambda_DA)
    T = 24
    M = capacity
    model = LpProblem(name="maximize-profit", sense=LpMaximize)

    p_DA = [LpVariable(f"p_DA_{t}", lowBound=0, upBound=capacity) for t in range(T)]
    delta_t_w = [[LpVariable(f"delta_{w}_{t}") for t in range(T)] for w in range(W)]
    delta_t_w_up = [[LpVariable(f"delta_up_{t}_{w}", lowBound=0) for t in range(T)] for w in range(W)]
    delta_t_w_down = [[LpVariable(f"delta_down_{t}_{w}", lowBound=0) for t in range(T)] for w in range(W)]
    binary_up = [[LpVariable(f"binary_up_{w}_{t}", cat=LpBinary) for t in range(T)] for w in range(W)]
    eta_w = [LpVariable(f"eta_{w}", lowBound=0) for w in range(W)]
    zeta = LpVariable(f"zeta")

    pi_w = (np.ones(W) / W).tolist()

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
    CVaR = zeta - (1 / (1 - alpha)) * lpDot(eta_w, pi_w)
    model += (1 - beta) * expected_profit + beta * CVaR, "Objective"

    for w in range(W):
        for t in range(T):
            model += delta_t_w[w][t] == p_real[w, t] - p_DA[t]
            model += delta_t_w[w][t] == delta_t_w_up[w][t] - delta_t_w_down[w][t]
            model += delta_t_w_up[w][t] <= M * binary_up[w][t]
            model += delta_t_w_down[w][t] <= M * (1 - binary_up[w][t])

        profit_expr = lpSum(
            lambda_DA[w, t] * p_DA[t]
            + 0.85 * lambda_DA[w, t] * delta_t_w_up[w][t] * sys_status[w, t]
            + lambda_DA[w, t] * delta_t_w_up[w][t] * (1 - sys_status[w, t])
            - 1.25 * lambda_DA[w, t] * delta_t_w_down[w][t] * (1 - sys_status[w, t])
            - lambda_DA[w, t] * delta_t_w_down[w][t] * sys_status[w, t]
            for t in range(T)
        )
        model += -profit_expr + zeta - eta_w[w] <= 0

    model.solve(PULP_CBC_CMD(msg=0))
    p_DA_opt = np.array([value(var) for var in p_DA])
    return p_DA_opt, value(expected_profit)

def evaluate_profit_twoprice(p_DA, lambda_DA, p_real, sys_status):
    W, T = lambda_DA.shape
    profits = np.zeros(W)
    for w in range(W):
        profit = 0
        for t in range(T):
            delta = p_real[w, t] - p_DA[t]
            DA = lambda_DA[w, t] * p_DA[t]
            if delta >= 0:
                if sys_status[w, t] == 1:
                    imbalance = 0.85 * lambda_DA[w, t] * delta
                else:
                    imbalance = lambda_DA[w, t] * delta
            else:
                if sys_status[w, t] == 1:
                    imbalance = lambda_DA[w, t] * delta
                else:
                    imbalance = 1.25 * lambda_DA[w, t] * delta
            profit += DA + imbalance
        profits[w] = profit
    return profits

# ------------------ MAIN LOOP ------------------
results = []
excel_path = os.path.join(output_dir, "twoprice_expost_all_results.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Write summary first
    df_summary = pd.DataFrame(columns=["Fold", "In-sample Profit [EUR]", "Out-of-sample Profit [EUR]", "Out-of-sample CVaR (10%) [EUR]"])
    df_summary.to_excel(writer, sheet_name="Summary", index=False)

    for fold in tqdm(range(folds), desc="Ex-Post Analysis Progress"):
        fold_start_time = time.time()

        fold_start = fold * in_sample_size
        fold_end = fold_start + in_sample_size
        in_sample = all_combinations[fold_start:fold_end]
        out_of_sample = all_combinations[:fold_start] + all_combinations[fold_end:]

        lambda_in, p_real_in, sys_in = get_scenarios(in_sample)
        lambda_out, p_real_out, sys_out = get_scenarios(out_of_sample)

        p_DA_opt, profit_in = solve_offering_strategy_twoprice(capacity, lambda_in, p_real_in, sys_in)
        profit_out = evaluate_profit_twoprice(p_DA_opt, lambda_out, p_real_out, sys_out)
        CVaR_10 = np.mean(np.sort(profit_out)[:int(0.1 * len(profit_out))])

        pd.DataFrame({"Hour": np.arange(1, T+1), "Optimal Bid [MW]": p_DA_opt}).to_excel(writer, sheet_name=f"Fold_{fold+1}_Bid", index=False)
        pd.DataFrame({"Scenario": np.arange(1, len(profit_out)+1), "Profit [EUR]": profit_out}).to_excel(writer, sheet_name=f"Fold_{fold+1}_Profits", index=False)

        results.append({
            "Fold": fold + 1,
            "In-sample Profit [EUR]": profit_in,
            "Out-of-sample Profit [EUR]": np.mean(profit_out),
            "Out-of-sample CVaR (10%) [EUR]": CVaR_10
        })

        # print(f"✔️ Fold {fold+1} completed in {time.time() - fold_start_time:.2f} seconds")

    pd.DataFrame(results).to_excel(writer, sheet_name="Summary", index=False)

print(f"\n✅ All results saved to {excel_path}")

# ------------------ CDF PLOT ------------------
plt.figure(figsize=(10, 6))
for fold in range(folds):
    out_set = all_combinations[:fold*in_sample_size] + all_combinations[(fold+1)*in_sample_size:]
    lambda_out, p_real_out, sys_out = get_scenarios(out_set)
    p_DA_opt, _ = solve_offering_strategy_twoprice(capacity, *get_scenarios(all_combinations[fold*in_sample_size:(fold+1)*in_sample_size]))
    profits = evaluate_profit_twoprice(p_DA_opt, lambda_out, p_real_out, sys_out)
    sorted_p = np.sort(profits)
    cdf = np.arange(1, len(sorted_p)+1) / len(sorted_p)
    plt.step(sorted_p, cdf, label=f"Fold {fold+1}")

plt.xlabel("Profit [EUR]")
plt.ylabel("Cumulative Probability")
plt.title("Two-Price Ex-Post Profit CDFs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "twoprice_cdf_all_folds.png"), dpi=300)
plt.show()
