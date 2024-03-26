from ortools.linear_solver import pywraplp
import time
import matplotlib.pyplot as plt
import numpy as np

solver = pywraplp.Solver.CreateSolver('GLOP')
#Exercise 1
# Define the variables
xA = solver.NumVar(0, solver.infinity(), 'xA')
xB = solver.NumVar(0, solver.infinity(), 'xB')

# Define the constraints
solver.Add(2*xA + 3*xB <= 960)
solver.Add(8*xA + 10*xB <= 3400)


# Define the objective function
solver.Maximize(22 * xA + 28*xB)

# Invoke the solver
start_time1 = time.time()
status = solver.Solve()
end_time1 = time.time()
print('\nExercise 1')
# Retrieve and display the solution
if status == pywraplp.Solver.OPTIMAL:
    print('Solution :') 
    print('Objective value =', solver.Objective().Value())
    print('xA =', xA.solution_value())
    print('xB =', xB.solution_value())
else:
    print('The problem does not have an optimal solution.')
print("Solving time for exercise 1:", end_time1 - start_time1, "seconds")


#Graphic of the feasible region 
x = np.linspace(0, 500, 400)  # Range of values for x-axis
y1 = (960 - 2*x) / 3  # First constraint
y2 = (3400 - 8*x) / 10  # Second constraint

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label="2xA + 3xB <= 960")
plt.plot(x, y2, label="8xA + 10xB <= 3400")
plt.fill_between(x, np.maximum(0, y1), np.minimum(y1, y2), where=y1>y2, color='darkgreen', alpha=0.5, label='Feasible Region')
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.xlabel('xA')
plt.ylabel('xB')
plt.title('Feasible Region for Exercise 1')
plt.legend()
#plt.show() You will see the graphic in the word attach


#Exercise 2
# Define the variables
x = [solver.NumVar(0, cap, f'x{i}') for i, cap in enumerate([800, 700, 600, 800], start=1)]
A = [solver.NumVar(50, solver.infinity(), f'A{i}') for i in range(1, 5)]

# Initial inventory
initial_inventory = 250
demands = [900, 600, 800, 600]
production_cost = 15
inventory_cost = 3
min_inventory = 50

# Define the constraints
# Inventory balance constraints
solver.Add(A[0] == initial_inventory + x[0] - demands[0])
for i in range(1, 4):
    solver.Add(A[i] == A[i - 1] + x[i] - demands[i])

# Define the objective function
objective_terms = [production_cost * x[i] for i in range(4)] + \
                  [inventory_cost * A[i] for i in range(4)]
solver.Minimize(solver.Sum(objective_terms))

# Invoke the solver
start_time2 = time.perf_counter()
status_2 = solver.Solve()
end_time2 = time.perf_counter()

# Retrieve and display the solution
print('\nExercise 2')
if status_2 == pywraplp.Solver.OPTIMAL:
    print('Solution :')
    print('Objective value =', solver.Objective().Value())
    for i in range(4):
        print(f'Week {i+1} production =', x[i].solution_value())
        print(f'Week {i+1} inventory =', A[i].solution_value())
else:
    print('The problem does not have an optimal solution.')
print("Solving time for exercise 2:", end_time2 - start_time2, "seconds")


# Assuming x[i].solution_value() and A[i].solution_value() are available from the solver
production_values = [x[i].solution_value() for i in range(4)]
inventory_values = [A[i].solution_value() for i in range(4)]
weeks = range(1, 5)

plt.figure(figsize=(10, 6))

# Plotting production levels
plt.plot(weeks, production_values, '-o', label='Production')
# Plotting inventory levels
plt.plot(weeks, inventory_values, '-s', label='Inventory')

plt.xlabel('Week')
plt.ylabel('Quantity')
plt.title('Production and Inventory Levels Over Time')
plt.xticks(weeks)
plt.grid(True)
plt.legend()
#plt.show()  You will see the graphic in the word attach


#Exercise 3
# Define the variables
x1_3 = solver.NumVar(0, solver.infinity(), 'x1_3')
x2_3 = solver.NumVar(0, solver.infinity(), 'x2_3')

# Define the constraints
solver.Add(x2_3 <= 5)
solver.Add(x1_3 + x2_3 <= 10)
solver.Add(-x1_3 + x2_3 >= -2)

# Define the objective function
solver.Maximize(3 * x1_3 + x2_3)

# Invoke the solver
start_time3 = time.perf_counter()
status_3 = solver.Solve()
end_time3 = time.perf_counter()

print('\nExercise 3')
# Retrieve and display the solution
if status_3 == pywraplp.Solver.OPTIMAL:
    print('Solution :')
    print('Objective value =', solver.Objective().Value())
    print('x1 =', x1_3.solution_value())
    print('x2 =', x2_3.solution_value())
else:
    print('The problem does not have an optimal solution.')
print("Solving time for exercise 3:", end_time3 - start_time3, "seconds")

#Graphic of the feasible region
x = np.linspace(0, 10, 400)
y1 = 5 * np.ones_like(x)
y2 = 10 - x
y3 = x - 2

# Plotting the constraints
plt.figure(figsize=(8, 6))

# Re-evaluating the feasible region accurately considering all constraints
plt.plot(x, y1, label=r'$x_2 \leq 5$', color="blue")
plt.plot(x, y2, label=r'$x_1 + x_2 \leq 10$', color="red")
plt.plot(x, y3, label=r'$x_2 \geq x_1 - 2$', color="green")

# Fill the true feasible region
y3_corrected = np.maximum(x - 2, 0)
y2_corrected = 10 - x
y_fill = np.minimum(y1, y2_corrected)
plt.fill_between(x, y3_corrected, y_fill, where=(y3_corrected<=y_fill), color='darkgreen', alpha=0.5, label='True Feasible Region')

# Labeling axes
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

# Adding other plot elements
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.title('Feasible Region for Exercise 3')
#plt.show() You will see the graphic in the word attach


#Exercise 4
# Define the variables
x1_4 = solver.NumVar(0, solver.infinity(), 'x1_4')
x2_4 = solver.NumVar(0, solver.infinity(), 'x2_4')
solver.Add(3*x1_4 + x2_4 >= 6)
solver.Add(x2_4 >= 3)
solver.Add(x1_4<=4)

# Define the objective function
solver.Minimize(x1_4 + x2_4)

# Invoke the solver
start_time4 = time.perf_counter()
status_4 = solver.Solve()
end_time4 = time.perf_counter()

print('\nExercise 4')
# Retrieve and display the solution
if status_4 == pywraplp.Solver.OPTIMAL:
    print('Solution :')
    print('Objective value =', solver.Objective().Value())
    print('x1 =', x1_4.solution_value())
    print('x2 =', x2_4.solution_value())
else:
    print('The problem does not have an optimal solution.')
print("Solving time for exercise 4:", end_time4 - start_time4, "seconds")

# Graphic of the feasible region 
x = np.linspace(0, 10, 400)

# Defining the constraints
y4 = np.maximum(6 - 3*x, 3)  
y5 = 3 * np.ones_like(x)  

plt.figure(figsize=(8, 6))

# Plotting the constraints
plt.plot(x, y4, label=r'$3x_1 + x_2 \geq 6$', color="blue")
plt.plot(x, y5, label=r'$x_2 \geq 3$', color="red")
plt.axvline(x=4, color="green", label=r'$x_1 \leq 4$')

# Highlighting the feasible region
plt.fill_between(x, np.maximum(y4, y5), 10, where=(x<=4), color='darkgreen', alpha=0.5, label='Feasible Region')

# Labeling axes
plt.xlim(0, 5)
plt.ylim(0, 10)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.title('Feasible Region for Exercise 4')
#plt.show() You will see the graphic in the word attach

# Exercise 5
# Définir les variables
x1_5 = solver.NumVar(0, solver.infinity(), 'x1_5')
x2_5 = solver.NumVar(0, solver.infinity(), 'x2_5')

# Définir les contraintes
solver.Add(-x1_5 + x2_5 <= 2)
solver.Add(x1_5 + 2 * x2_5 <= 8)
solver.Add(x1_5 <= 6)

# Définir la fonction objectif
solver.Maximize(x1_5 + 2 * x2_5)

# Invoquer le solveur
start_time5 = time.perf_counter()
status_5 = solver.Solve()
end_time5 = time.perf_counter()

# Récupérer et afficher la solution
print('\nExercise 5')
if status_5 == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x1 =', x1_5.solution_value())
    print('x2 =', x2_5.solution_value())
else:
    print("The problem does not have an optimal solution.")
print("Solving time for exercise 5:", end_time5 - start_time5, "seconds")

# Graphic of the feasible region 
x1 = np.linspace(0, 10, 400)

# Translate constraints to equations
y1 = x1 - 2  
y2 = (8 - x1) / 2 

plt.figure(figsize=(8, 6))
plt.plot(x1, y1, label=r'$-x_1 + x_2 \leq 2$', color='blue')
plt.plot(x1, y2, label=r'$x_1 + 2x_2 \leq 8$', color='red')
plt.axvline(x=6, label=r'$x_1 \leq 6$', color='green')


plt.fill_between(x1, np.maximum(y1, 0), np.minimum(y2, 6), where=(x1<=6), color='darkgreen', alpha=0.5, label='Feasible Region')

plt.xlim(0, 7)
plt.ylim(-3, 5)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.title('Feasible Region for Exercise 5')
#plt.show() You will see the graphic in the word attach


# Exercise 6
# Définir les variables
x1_6 = solver.NumVar(0, solver.infinity(), 'x1_6')
x2_6 = solver.NumVar(0, solver.infinity(), 'x2_6')

# Définir les contraintes
solver.Add(x1_6 + x2_6 >= 4)
solver.Add(-x1_6 + x2_6 <= 4)
solver.Add(-x1_6 + 2 * x2_6 >= -4)

# Définir la fonction objectif
solver.Maximize(3 * x1_6 + x2_6)

# Invoquer le solveur
start_time6 = time.perf_counter()
status_6 = solver.Solve()
end_time6 = time.perf_counter()
# Récupérer et afficher la solution
print('\nExercise 6')
if status_6 == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x1 =', x1_6.solution_value())
    print('x2 =', x2_6.solution_value())
else:
    print("The problem does not have an optimal solution.")
print("Solving time for exercise 6:", end_time6 - start_time6, "seconds")

x1_6 = np.linspace(-2, 10, 400)

y1_6 = 4 - x1_6  
y2_6 = x1_6 + 4  
y3_6 = 2 + 0.5 * x1_6 

plt.figure(figsize=(8, 6))

# Plotting the constraints
plt.plot(x1_6, y1_6, label=r'$x_1 + x_2 \geq 4$', color='blue')
plt.plot(x1_6, y2_6, label=r'$-x_1 + x_2 \leq 4$', color='red')
plt.plot(x1_6, y3_6, label=r'$-x_1 + 2x_2 \geq -4$', color='green')

# Filling the feasible region
plt.fill_between(x1_6, np.maximum(y1_6, y3_6), y2_6, where=(y2_6>=np.maximum(y1_6, y3_6)), color='darkgreen', alpha=0.5, label='Feasible Region')

# Setting up the plot
plt.xlim(-2, 10)
plt.ylim(-2, 10)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.title('Feasible Region for Exercise 6')
#plt.show() You will see the graphic in the word attach



# Exercise 7
# Définir les variables
x1_7 = solver.NumVar(0, solver.infinity(), 'x1_7')
x2_7 = solver.NumVar(0, solver.infinity(), 'x2_7')

# Définir les contraintes
solver.Add(-x1_7 + x2_7 >= 4)
solver.Add(-x1_7 + 2 * x2_7 <= -4)

# Définir la fonction objectif
solver.Maximize(3 * x1_7 + x2_7)

# Invoquer le solveur
start_time7 = time.perf_counter()
status_7 = solver.Solve()
end_time7 = time.perf_counter()

# Récupérer et afficher la solution
print('\nExercise 7')
if status_7 == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x1 =', x1_7.solution_value())
    print('x2 =', x2_7.solution_value())
else:
    print("The problem does not have an optimal solution.")
print("Solving time for exercise 7:", end_time7 - start_time7, "seconds")

x1_7 = np.linspace(-10, 10, 400)


y1_7 = x1_7 + 4  
y2_7 = (x1_7 - 2) / 2  

plt.figure(figsize=(8, 6))
plt.plot(x1_7, y1_7, label=r'$-x_1 + x_2 \geq 4$', color='blue')
plt.plot(x1_7, y2_7, label=r'$-x_1 + 2x_2 \leq -4$', color='red')
plt.fill_between(x1_7, y1_7, np.maximum(y1_7, y2_7), where=(y1_7>=y2_7), color='darkgreen', alpha=0.5, label='Feasible Region')


plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.title('Feasible Region for Exercise 7')
#plt.show() You will see the graphic in the word attach