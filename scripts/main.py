from GeneticAlgorithm import GeneticAlgorithm as GA
import csv


target = 55
# Size of the population
max_population_count = 100
# Number of genes in an individual
individual_length = 6
# Min and max possible values for each gene
individual_min_value = 0
individual_max_value = 10
# Number of generations
max_num_generations = 10000

fileName = "./results/population_count_log.csv"
with open(fileName, "w", newline="") as csvFile:
    csvWriter = csv.DictWriter(
        csvFile, fieldnames=["population_count", "num_generations", "fitness"]
    )
    csvWriter.writeheader()

population_counts = [base * 10 ** exp for exp in range(1, individual_length + 1) for base in range(1, 10)]

for population_count in population_counts:
    ga = GA(
        population_count,
        individual_length,
        individual_min_value,
        individual_max_value,
    )

    current_fitness = ga.grade(target)
    fitness_history = [current_fitness]

    while current_fitness > 0 and len(fitness_history) < max_num_generations:
        ga.evolve(target)
        current_fitness = ga.grade(target)
        fitness_history.append(current_fitness)

    print(f"Population Count: {population_count}\tFitness: {current_fitness}\tGenerations: {len(fitness_history)}")

    with open(fileName, "a", newline="") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=["population_count", "num_generations", "fitness"]
        )
        csvWriter.writerow(
            {
                "population_count": population_count,
                "num_generations": len(fitness_history),
                "fitness": current_fitness,
            }
        )
