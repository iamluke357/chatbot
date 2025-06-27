from pulp import (
    LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, PULP_CBC_CMD
)

# ----- Data -----
tables  = ['T1', 'T2', 'T3', 'T4', 'T5']
waiters = ['W1', 'W2', 'W3']
service_times = {'T1': 20, 'T2': 10, 'T3': 25, 'T4': 15, 'T5': 30}
max_tables_per_waiter = 2

# ----- Model -----
model = LpProblem("Minimize_Max_Load", LpMinimize)

# Decision variables: assign waiter w to table t
assign = LpVariable.dicts("assign", [(w, t) for w in waiters for t in tables], 0, 1, LpBinary)

# Max workload variable
L = LpVariable("MaxLoad", lowBound=0)

# ----- Objective -----
model += L  # Minimize the worst load across all waiters

# ----- Constraints -----

# 1. Each table is assigned to exactly one waiter
for t in tables:
    model += lpSum(assign[w, t] for w in waiters) == 1

# 2. No waiter handles more than allowed number of tables
for w in waiters:
    model += lpSum(assign[w, t] for t in tables) <= max_tables_per_waiter

# 3. Each waiter's workload must be ≤ L
for w in waiters:
    model += lpSum(assign[w, t] * service_times[t] for t in tables) <= L

# ----- Solve -----
solver = PULP_CBC_CMD(msg=False)
model.solve(solver)

# ----- Results -----
print("Balanced Table Assignments:")
for w in waiters:
    for t in tables:
        if assign[w, t].value() == 1:
            print(f"  {w} → {t}")

print(f"\nMaximum individual workload: {L.value()} minutes")
