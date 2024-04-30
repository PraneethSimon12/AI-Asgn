'''Solve the given 0/1 knapsack problem by considering the following points: 
Name Weight Value A 45 3 B 40 5 C 50 8 D 90 10 Chromosome is a 4-bit string. 
– {xA xB xC xD} Population size = 4, Maximum Capacity of the bag (W) = 100. 
First two fittest chromosomes selected as it is. 3rd and 4th fittest use for 
one-point crossover in the middle followed by single bit mutation of first offspring. 
Bits chosen for mutation follows this cyclic order (xD, xC, xB, xA). 
Initial population: {1 1 1 1, 1 0 0 0, 1 0 1 0, 1 0 0 1}. 
Output the result after 10 iterations.
'''

import random

# Problem parameters
items = [
    {"name": "A", "weight": 45, "value": 3},
    {"name": "B", "weight": 40, "value": 5},
    {"name": "C", "weight": 50, "value": 8},
    {"name": "D", "weight": 90, "value": 10}
]
max_weight = 100
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
    return [mutation(offspring1), mutation(offspring2)]

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