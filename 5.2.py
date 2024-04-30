'''A thief enters a house for robbing it. He can carry a maximal weight of 9 kg into his bag. There are 4 items in the house with the following weights and values. The thief has to plan the items he should take to maximize the total value if he either takes the item completely or leaves it completely?
Item Item Name Weight (in Kg) Value (in $) A Mirror 2 3
B Silver Nugget 3 5
C Painting 4 7
D Vase 5 9
The problem is solved using Genetic Algorithm with population size 4 and each individual encoded as {XA, XB, XC, XD} where Xi ={0,1 } and i=A, B, C, D.
Consider initial population as 1111, 1000, 1010, and 1001.
Generate the population for next iteration as follows: Select the 1st and 2nd fittest individual as it is in the next iteration. Apply 1-point crossover in the middle between 3rd and 4th fittest chromosome followed by single bit mutation of first offspring (produced through crossover). Bit chosen for mutation follows this cyclic'''

import random

# Problem parameters
items = [
    {"name": "A", "weight": 2, "value": 3},
    {"name": "B", "weight": 3, "value": 5},
    {"name": "C", "weight": 4, "value": 7},
    {"name": "D", "weight": 5, "value": 9}
]
max_weight = 9
population_size = 4
chromosome_length = len(items)

# Initial population
population = [
    [1, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1]
]

def fitness(chromosome):
    weight = sum(items[i]["weight"] * chromosome[i] for i in range(chromosome_length))
    value = sum(items[i]["value"] * chromosome[i] for i in range(chromosome_length))
    if weight > max_weight:
        return 0
    return value

def selection(population):
    sorted_population = sorted(population, key=lambda x: fitness(x), reverse=True)
    return sorted_population[:2] + crossover(sorted_population[2], sorted_population[3])

def crossover(parent1, parent2):
    crossover_point = chromosome_length // 2
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return [mutation(offspring1), offspring2]

def mutation(chromosome):
    mutation_point = random.randint(0, chromosome_length - 1)
    chromosome[mutation_point] = 1 - chromosome[mutation_point]
    return chromosome

for iteration in range(10):
    population = selection(population)

best_chromosome = max(population, key=fitness)
best_items = [items[i]["name"] for i in range(chromosome_length) if best_chromosome[i] == 1]
best_value = fitness(best_chromosome)
best_weight = sum(items[i]["weight"] * best_chromosome[i] for i in range(chromosome_length))

print(f"Best items: {best_items}")
print(f"Maximum value: {best_value}")
print(f"Total weight: {best_weight}")