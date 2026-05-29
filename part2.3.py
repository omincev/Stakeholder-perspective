import time

start_time = time.time()

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

"""
IMPORT DATA
"""


# === Load the Excel data ===

p2data = 'p2data.xlsx'
df = pd.read_excel(p2data, index_col=0)


"""
CONSTRAIN DATA TO IN SAMPLE AND OUT OF SAMPLE DATA
"""

# Use first 100 profiles
F = df.iloc[:100].values.T  # Shape (60, 100)

# === Load out-of-sample data ===
F_remaining = df.iloc[100:300].values.T  # Shape (60, 200)

# Problem dimensions
minutes, scenarios = F.shape
M = 1e6  # Big-M constant

"""
SET OPEN ARRAY TO COLLECT DATA FROM MODEL RESULTS AND SET VALUES FOR EPSILON IN VARIABLE x_lables
"""

# Store results
results = []
meanShortfall = []
totalShortfall = []
# Loop over different values of x
x_values = np.arange(0, 0.22, 0.02)  # From 0.2 to 1.0 in steps of 0.02

"""
RUN MODEL 10 TIMES INCREASING THE VALUE FOR EPSILON (x_value) WITH EACH LOOP
"""

for x in x_values:
    q = int(100 * 60 * x)

    model = gp.Model("Maximize_c_up")
    c_up = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="c_up")
    y = model.addVars(minutes, scenarios, vtype=GRB.BINARY, name="y")

    for m in range(minutes):
        for w in range(scenarios):
            model.addConstr(c_up - F[m, w] <= y[m, w] * M)

    model.addConstr(gp.quicksum(y[m, w] for m in range(minutes) for w in range(scenarios)) <= q)
    model.setObjective(c_up, GRB.MAXIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        c_opt = c_up.X
        shortfalls = []

        for m in range(F_remaining.shape[0]):
            for w in range(F_remaining.shape[1]):
                if F_remaining[m, w] < c_opt:
                    shortfalls.append(c_opt - F_remaining[m, w])

        expected_shortfall = np.mean(shortfalls) if shortfalls else 0.0
        total_shortfall = sum(shortfalls)

        results.append((x, c_opt))
        meanShortfall.append((x, np.round(expected_shortfall, 1)))
        totalShortfall.append((x, total_shortfall))

        print(f"x = {x:.2f}, Optimal c↑ = {c_opt:.2f}, Mean shortfall = {expected_shortfall:.2f}")
    else:
        results.append((x, None))
        meanShortfall.append((x, None))
        totalShortfall.append((x, None))
        print(f"x = {x:.2f}, No optimal solution found.")

"""
PLOTS
"""

import matplotlib.pyplot as plt

# Unpack the data
x_vals = [x for x, _ in results]
c_up_vals = [c for _, c in results]
mean_shortfall_vals = [s for _, s in meanShortfall]
total_shortfall_vals = [t for _, t in totalShortfall]

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Optimal c↑ and Mean Shortfall on left y-axis
color1 = 'tab:blue'
ax1.set_xlabel("ε (epsilon)")
ax1.set_ylabel("Optimal c↑ and Mean Shortfall [kW]", color=color1)
ax1.plot(x_vals, c_up_vals, label="Optimal c↑", color="tab:blue", marker='o')
ax1.plot(x_vals, mean_shortfall_vals, label="Mean Shortfall", color="tab:green", marker='s')
ax1.tick_params(axis='y', labelcolor=color1)

# Create second y-axis for total shortfall
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel("Total Shortfall [kW]", color=color2)
ax2.plot(x_vals, total_shortfall_vals, label="Total Shortfall", color=color2, marker='^')
ax2.tick_params(axis='y', labelcolor=color2)

# Title and legend
fig.suptitle("ε varying from 0 to 0.2 using ALSO - X", fontsize=14)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/jaredbutler/Documents/Masters/Winter 2025/1) Renewables in Electricity markets/Assignment 2/q_sweep.png", dpi=300)
plt.show()

end_time = time.time()
print("Run time:", end_time - start_time, "seconds")

# Count user-defined variables
user_vars = [var for var in globals() if not var.startswith("__") and not callable(globals()[var])]
print("Number of user-defined variables:", len(user_vars))
    