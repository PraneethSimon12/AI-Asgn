'''Consider the following 2-SAT problem with 4 Boolean variables a, b, c, d: F=(¬a∨d)𝖠(c∨b) 𝖠 ( ¬c∨¬d) 𝖠 ( ¬d∨¬b) 𝖠 (¬a∨¬d)
The MOVEGEN function to generate new solution be arbitrary changing value of any one variable
Let the candidate solution be of the order (abcd) and the initial candidate solution be (1111). Let heuristic to evaluate each solution be number of clauses satisfied in the formula.
Apply Simulated Annealing (Consider T= 500 and cooling function = T-50) (Assume the following 3 random numbers:0.655,0.254.0.432)
Accept every good move and accept a bad move if probability is greater than 50%.'''

import random
import math

# Define the clauses of the 2-SAT formula
clauses = [(-1, 4), (3, 2), (-3, -4), (-4, -2), (-1, -4)]

# Define the initial candidate solution
candidate_solution = [1, 1, 1, 1]
num_variables = len(candidate_solution)

# Define the random numbers for the simulation
random_numbers = [0.655, 0.254, 0.432]
random_index = 0

# Define the Simulated Annealing parameters
T = 500
cooling_factor = 50

def evaluate_solution(solution):
    """Evaluate the solution by counting the number of satisfied clauses"""
    satisfied_clauses = 0
    for clause in clauses:
        literal1, literal2 = clause
        value1 = solution[abs(literal1) - 1] if literal1 > 0 else not solution[abs(literal1) - 1]
        value2 = solution[abs(literal2) - 1] if literal2 > 0 else not solution[abs(literal2) - 1]
        if value1 or value2:
            satisfied_clauses += 1
    return satisfied_clauses

def generate_neighbor(solution):
    """Generate a neighbor solution by flipping the value of a random variable"""
    neighbor = solution.copy()
    variable_index = random.randint(0, num_variables - 1)
    neighbor[variable_index] = 1 - neighbor[variable_index]
    return neighbor

def acceptance_probability(current_score, new_score, temperature):
    """Calculate the acceptance probability for a bad move"""
    if new_score > current_score:
        return 1.0
    else:
        return math.exp((new_score - current_score) / temperature)

# Initialize the best solution and its score
best_solution = candidate_solution
best_score = evaluate_solution(best_solution)

# Simulated Annealing loop
while T > 0:
    # Generate a neighbor solution
    neighbor_solution = generate_neighbor(candidate_solution)

    # Evaluate the neighbor solution
    neighbor_score = evaluate_solution(neighbor_solution)

    # Calculate the acceptance probability for a bad move
    if neighbor_score < best_score:
        random_num = random_numbers[random_index]
        random_index = (random_index + 1) % len(random_numbers)
        acceptance_prob = acceptance_probability(best_score, neighbor_score, T)
        if acceptance_prob > random_num:
            candidate_solution = neighbor_solution
            best_score = neighbor_score

    # Update the best solution if the neighbor is better
    if neighbor_score > best_score:
        best_solution = neighbor_solution
        best_score = neighbor_score

    # Cool down the temperature
    T -= cooling_factor

# Print the best solution and its score
print("Best solution:", best_solution)
print("Number of satisfied clauses:", best_score)