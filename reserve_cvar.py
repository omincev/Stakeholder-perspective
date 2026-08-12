import time

start_time = time.time()

import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt

"""
IMPORT DATA
"""

# === Load the data ===
p2data = 'p2data.xlsx'
df = pd.read_excel(p2data, index_col=0)

"""
TASK 2.1 CONSTRAIN DATA TO IN SAMPLE SCENARIOS AND SET UP MODEL AND CONSTRAINTS
"""

F = df.iloc[:100].values.T  # Shape: (60, 100)

# === Parameters ===
minutes, scenarios = F.shape  # 60, 100
epsilon = 0.1  # P90 compliance: epsilon = 0.1 means 90% confidence

# === Define PuLP Model ===
model = pulp.LpProblem("CVaR_MaxC", pulp.LpMaximize)

# === Variables ===
c_up = pulp.LpVariable("c_up", lowBound=0.0, cat='Continuous')
beta = pulp.LpVariable("beta", upBound=0.0, cat='Continuous')
zeta = pulp.LpVariable.dicts("zeta",
                             ((m, w) for m in range(minutes) for w in range(scenarios)),
                             cat='Continuous')

# === Constraints ===

# Constraint 1: c_up - F[m, w] ≤ zeta[m, w]
for m in range(minutes):
    for w in range(scenarios):
        model += c_up - F[m, w] <= zeta[(m, w)], f"shortfall_{m}_{w}"

# Constraint 2: Average zeta must be ≤ (1 - epsilon) * beta
avg_zeta_expr = pulp.lpSum(zeta[(m, w)] for m in range(minutes) for w in range(scenarios)) / (minutes * scenarios)
model += avg_zeta_expr <= (1 - epsilon) * beta, "CVaR_constraint"

#Constraint 3: β ≤ zeta[m, w]
for m in range(minutes):
    for w in range(scenarios):
        model += beta <= zeta[(m, w)], f"VaR_bound_{m}_{w}"

# === Objective: Maximize c_up ===
model += c_up, "Maximize_Reserve"


"""
SOLVE MODEL AND PRINT RESULTS
"""

# === Solve ===
model.solve()

# === Output ===
if pulp.LpStatus[model.status] == 'Optimal':
    print(f"Optimal c↑ (Reserve Offered): {c_up.value():.2f} kW")
    print(f"Optimal β (VaR Threshold): {beta.value():.2f}\n")

#     print("ζ (Shortfall) values > 0:")
#     for m in range(minutes):
#         for w in range(scenarios):
#             z = zeta[(m, w)].value()
#             if z > 1e-6:
#                 print(f"ζ[{m},{w}] = {z:.2f}")
else:
    print("No optimal solution found.")


"""
TASK 2.2 CONSTRAIN DATA TO OUT OF SAMPLE SCENARIOS AND CALCULATE HOW MANY MINUTES THE LOAD IS NOT MET
"""

#Select out of sample profiles
F_remaining = df.iloc[100:300].values.T  # (60, 200)

# Assume c_up is a PuLP variable you've already solved
c_opt = c_up.varValue  # This is how you get the value in PuLP

# F_remaining = remaining 200 profiles (60 x 200 matrix)
percent_exceed = []

for scenario_idx in range(F_remaining.shape[1]):
    profile_data = F_remaining[:, scenario_idx]
    count_exceed = np.sum(profile_data < c_opt)
    percent = (count_exceed / 60) * 100
    profile_number = scenario_idx + 101  # Because the first 100 were used
    percent_exceed.append((profile_number, percent))


# Assume percent_exceed is a list of (profile_number, percent) tuples
profile_numbers = [p[0] for p in percent_exceed]
percentages = [p[1] for p in percent_exceed]




# Assuming c_up is a PuLP variable
c_opt = c_up.varValue

# Flatten and compare
flattened = F.flatten()
num_below = np.sum(flattened <= c_opt)
total_values = F.size

# Compute percentage
percent_below = (num_below / total_values) * 100

print(f"Percentage of minutes below c↑ In sample = {c_opt:.2f}: {percent_below:.2f}%")


# Assuming c_up is a PuLP variable
c_opt = c_up.varValue

# Flatten and compare
flattened = F_remaining.flatten()
num_below = np.sum(flattened <= c_opt)
total_values = F_remaining.size

# Compute percentage
percent_below = (num_below / total_values) * 100

print(f"Percentage of minutes below c↑ Out of sample = {c_opt:.2f}: {percent_below:.2f}%")

"""
PLOTS
"""

plt.figure(figsize=(14, 6))
plt.bar(profile_numbers, percentages, color='skyblue', edgecolor='k')
plt.axhline(y=10, color='r', linestyle='--', label='10% Threshold')
plt.axhline(y=15, color='orange', linestyle='--', label='15% Threshold')

plt.xlabel('Profile Number')
plt.ylabel('% of Minutes Below c↑')
plt.title('Reserve Shortfall % per Profile CVAR')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Save the figure
save_path = '/Users/jaredbutler/Documents/Masters/Winter 2025/1) Renewables in Electricity markets/Assignment 2/shortfall_bar_graph_CVAR.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

end_time = time.time()
print("Run time:", end_time - start_time, "seconds")

# Count user-defined variables
user_vars = [var for var in globals() if not var.startswith("__") and not callable(globals()[var])]
print("Number of user-defined variables:", len(user_vars))
