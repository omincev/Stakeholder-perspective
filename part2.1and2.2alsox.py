import time

start_time = time.time()

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
IMPORT DATA
"""

# === Load the Excel data ===
p2data = 'p2data.xlsx'
df = pd.read_excel(p2data, index_col=0)


"""
TASK 2.1 CONSTRAIN DATA TO IN SAMPLE SCENARIOS AND SET UP MODEL AND CONSTRAINTS
"""

# Use first 100 profiles
F = df.iloc[:100].values.T  # Shape (60, 100), where rows = minutes (m), cols = scenarios (ω)

# Problem dimensions
minutes, scenarios = F.shape
M = 1e6   # Big-M constant
q = 100*60*0.1   # Max number of selected y_{m,ω}

# === Set up Gurobi Model ===
model = gp.Model("Maximize_c_up")

# Variables
c_up = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="c_up")
y = model.addVars(minutes, scenarios, vtype=GRB.BINARY, name="y")

# Constraint 1: c_up - F_{m,ω} ≤ y_{m,ω} * M
for m in range(minutes):
    for w in range(scenarios):
        model.addConstr(c_up - F[m, w] <= y[m, w] * M, name=f"cons1_{m}_{w}")

# Constraint 2: sum of y_{m,ω} ≤ q
model.addConstr(gp.quicksum(y[m, w] for m in range(minutes) for w in range(scenarios)) <= q, name="sum_y_leq_q")

# Objective: Maximize c_up
model.setObjective(c_up, GRB.MAXIMIZE)

"""
SOLVE MODEL AND PRINT RESULTS
"""

# Solve
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal c↑: {c_up.X:.2f}")
    selected = sum(y[m, w].X for m in range(minutes) for w in range(scenarios))
    print(f"Total y[m,ω] == 1: {int(selected)}")
else:
    print("No optimal solution found.")

"""
TASK 2.2 CONSTRAIN DATA TO OUT OF SAMPLE SCENARIOS AND CALCULATE HOW MANY MINUTES THE LOAD IS NOT MET
"""

#Select out of sample profiles
F_remaining = df.iloc[100:300].values.T  # (60, 200)

# === Compute % of minutes exceeding optimal c_up ===
c_opt = c_up.X

percent_exceed = []

for scenario_idx in range(F_remaining.shape[1]):  # 0 to 199
    profile_data = F_remaining[:, scenario_idx]
    count_exceed = np.sum(profile_data < c_opt)
    percent = (count_exceed / 60) * 100
    profile_number = scenario_idx + 101
    percent_exceed.append((profile_number, percent))

# Assume df has shape (300, 60)
plt.figure(figsize=(14, 7))

# Plot each profile (each row is one profile)
for i in range(F.shape[0]):
    plt.plot(range(60), df.iloc[i, :], linewidth=0.6, alpha=0.4)

# Flatten into a single 1D array for easy comparison
flattened = F.flatten()

# Count how many values are below the optimal value
num_below = np.sum(flattened <= c_up.X)

# Total number of values
total_values = F.size

# Compute percentage
percent_below = (num_below / total_values) * 100

print(f"Percentage of minutes below c↑ In sample = {c_up.X:.2f}: {percent_below:.2f}%")


# Flatten into a single 1D array for easy comparison
flattened = F_remaining.flatten()

# Count how many values are below the optimal value
num_below = np.sum(flattened <= c_up.X)

# Total number of values
total_values = F_remaining.size

# Compute percentage
percent_below = (num_below / total_values) * 100

print(f"Percentage of minutes below c↑ Out of sample = {c_up.X:.2f}: {percent_below:.2f}%")

"""
PLOTS
"""

# Labels and formatting
plt.title("100 In-Sample Stochastic Profiles Over One Hour")
plt.xlabel("Minutes")
plt.ylabel("Power [kW]")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save to file
save_path = '/Users/jaredbutler/Documents/Masters/Winter 2025/1) Renewables in Electricity markets/Assignment 2/in-samplestochastic_profiles_1hr.png'
plt.savefig(save_path, dpi=300)
plt.show()

# Assume percent_exceed is a list of (profile_number, percent) tuples
profile_numbers = [p[0] for p in percent_exceed]
percentages = [p[1] for p in percent_exceed]

plt.figure(figsize=(14, 6))
plt.bar(profile_numbers, percentages, color='skyblue', edgecolor='k')
plt.axhline(y=10, color='r', linestyle='--', label='10% Threshold')
plt.axhline(y=15, color='orange', linestyle='--', label='15% Threshold')

plt.xlabel('Profile Number')
plt.ylabel('% of Minutes Below c↑')
plt.title('Reserve Shortfall % per Profile ALSO - X')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Save the figure
save_path = '/Users/jaredbutler/Documents/Masters/Winter 2025/1) Renewables in Electricity markets/Assignment 2/shortfall_bar_graph_also.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

end_time = time.time()
print("Run time:", end_time - start_time, "seconds")

# Count user-defined variables
user_vars = [var for var in globals() if not var.startswith("__") and not callable(globals()[var])]
print("Number of user-defined variables:", len(user_vars))

