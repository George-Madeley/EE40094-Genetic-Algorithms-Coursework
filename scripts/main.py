from GeneticAlgorithm import GeneticAlgorithm as GA
import csv


def vary_population(
    target,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    fileName,
):
    with open(fileName, "w", newline="") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=["population_count", "num_generations", "min fitness", "avg fitness"]
        )
        csvWriter.writeheader()

    # Define the population counts to test
    population_counts = [base * 10 ** exp for exp in range(1, 6) for base in range(1, 10)]
    population_counts.append(1000000)

    # Run the GA for each population count
    for population_count in population_counts:
        # Create a new GA instance
        ga = GA(
            population_count,
            individual_length,
            individual_min_value,
            individual_max_value,
        )

        # Run the GA
        min_fitness = ga.calculateMinGrade(target)
        avg_fitness = ga.calculateAverageGrade(target)
        fitness_history = [{"min": min_fitness, "avg": avg_fitness}]

        # Evolve the population until we reach the target or the max number of generations
        while min_fitness > 0 and len(fitness_history) < max_num_generations:
            ga.evolve(target)
            min_fitness = ga.calculateMinGrade(target)
            avg_fitness = ga.calculateAverageGrade(target)
            fitness_history.append({"min": min_fitness, "avg": avg_fitness})

        # Print the results
        min_fitness_str = "{:.2f}".format(min_fitness)
        avg_fitness_str = "{:.2f}".format(avg_fitness)
        print(f"Population Count: {population_count}\tMin Fitness: {min_fitness_str}\tAvg Fitness:\t{avg_fitness_str}\tGenerations: {len(fitness_history)}")

        # Write the results to a CSV file
        with open(fileName, "a", newline="") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile, fieldnames=["population_count", "num_generations", "min fitness", "avg fitness"]
            )
            csvWriter.writerow(
                {
                    "population_count": population_count,
                    "num_generations": len(fitness_history),
                    "min fitness": min_fitness,
                    "avg fitness": avg_fitness,
                }
            )

def vary_mutation(
    target,
    population_count,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    fileName,
):
    with open(fileName, "w", newline="") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=["mutate", "num_generations", "min fitness", "avg fitness"]
        )
        csvWriter.writeheader()

    mutations = [mutate/100 for mutate in range(0, 100)]
    mutations.append(1)

    # Run the GA for each population count
    for mutate in mutations:
        # Create a new GA instance
        ga = GA(
            population_count,
            individual_length,
            individual_min_value,
            individual_max_value,
        )

        # Run the GA
        min_fitness = ga.calculateMinGrade(target)
        avg_fitness = ga.calculateAverageGrade(target)
        fitness_history = [{"min": min_fitness, "avg": avg_fitness}]

        # Evolve the population until we reach the target or the max number of generations
        while min_fitness > 0 and len(fitness_history) < max_num_generations:
            ga.evolve(target, mutate=mutate)
            min_fitness = ga.calculateMinGrade(target)
            avg_fitness = ga.calculateAverageGrade(target)
            fitness_history.append({"min": min_fitness, "avg": avg_fitness})

        # Print the results
        min_fitness_str = "{:.2f}".format(min_fitness)
        avg_fitness_str = "{:.2f}".format(avg_fitness)
        print(f"Mutate: {mutate}\tMin Fitness: {min_fitness_str}\tAvg Fitness:\t{avg_fitness_str}\tGenerations: {len(fitness_history)}")

        # Write the results to a CSV file
        with open(fileName, "a", newline="") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile, fieldnames=["mutate", "num_generations", "min fitness", "avg fitness"]
            )
            csvWriter.writerow(
                {
                    "mutate": mutate,
                    "num_generations": len(fitness_history),
                    "min fitness": min_fitness,
                    "avg fitness": avg_fitness,
                }
            )

def vary_retain(
    target,
    population_count,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    fileName,
):
    with open(fileName, "w", newline="") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=["retain", "num_generations", "min fitness", "avg fitness"]
        )
        csvWriter.writeheader()

    retains = [retain/100 for retain in range(2, 98, 2)]

    # Run the GA for each population count
    for retain in retains:
        # Create a new GA instance
        ga = GA(
            population_count,
            individual_length,
            individual_min_value,
            individual_max_value,
        )

        # Run the GA
        min_fitness = ga.calculateMinGrade(target)
        avg_fitness = ga.calculateAverageGrade(target)
        fitness_history = [{"min": min_fitness, "avg": avg_fitness}]

        # Evolve the population until we reach the target or the max number of generations
        while min_fitness > 0 and len(fitness_history) < max_num_generations:
            ga.evolve(target, retain=retain)
            min_fitness = ga.calculateMinGrade(target)
            avg_fitness = ga.calculateAverageGrade(target)
            fitness_history.append({"min": min_fitness, "avg": avg_fitness})

        # Print the results
        min_fitness_str = "{:.2f}".format(min_fitness)
        avg_fitness_str = "{:.2f}".format(avg_fitness)
        print(f"Retain: {retain}\tMin Fitness: {min_fitness_str}\tAvg Fitness:\t{avg_fitness_str}\tGenerations: {len(fitness_history)}")

        # Write the results to a CSV file
        with open(fileName, "a", newline="") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile, fieldnames=["retain", "num_generations", "min fitness", "avg fitness"]
            )
            csvWriter.writerow(
                {
                    "retain": retain,
                    "num_generations": len(fitness_history),
                    "min fitness": min_fitness,
                    "avg fitness": avg_fitness,
                }
            )

def vary_random_select(
    target,
    population_count,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    fileName,
):
    with open(fileName, "w", newline="") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=["random_select", "num_generations", "min fitness", "avg fitness"]
        )
        csvWriter.writeheader()

    random_selects = [random_select/100 for random_select in range(0, 100)]

    # Run the GA for each population count
    for random_select in random_selects:
        # Create a new GA instance
        ga = GA(
            population_count,
            individual_length,
            individual_min_value,
            individual_max_value,
        )

        # Run the GA
        min_fitness = ga.calculateMinGrade(target)
        avg_fitness = ga.calculateAverageGrade(target)
        fitness_history = [{"min": min_fitness, "avg": avg_fitness}]

        # Evolve the population until we reach the target or the max number of generations
        while min_fitness > 0 and len(fitness_history) < max_num_generations:
            ga.evolve(target, random_select=random_select)
            min_fitness = ga.calculateMinGrade(target)
            avg_fitness = ga.calculateAverageGrade(target)
            fitness_history.append({"min": min_fitness, "avg": avg_fitness})

        # Print the results
        min_fitness_str = "{:.2f}".format(min_fitness)
        avg_fitness_str = "{:.2f}".format(avg_fitness)
        print(f"Random Select: {random_select}\tMin Fitness: {min_fitness_str}\tAvg Fitness:\t{avg_fitness_str}\tGenerations: {len(fitness_history)}")

        # Write the results to a CSV file
        with open(fileName, "a", newline="") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile, fieldnames=["random_select", "num_generations", "min fitness", "avg fitness"]
            )
            csvWriter.writerow(
                {
                    "random_select": random_select,
                    "num_generations": len(fitness_history),
                    "min fitness": min_fitness,
                    "avg fitness": avg_fitness,
                }
            )

target = 550
# Size of the population
population_count = 100
# Number of genes in an individual
individual_length = 6
# Min and max possible values for each gene
individual_min_value = 0
individual_max_value = 100
# Number of generations
max_num_generations = 10000

fileName = "./results/population_count_log.csv"

print("Varying population count")

vary_population(
    target,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    "./results/population_count_log.csv",
)

print("Varying mutation rate")

vary_mutation(
    target,
    population_count,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    "./results/mutation.csv",
)

print("Varying retain rate")

vary_retain(
    target,
    population_count,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    "./results/retain.csv",
)

print("Varying random select rate")

vary_random_select(
    target,
    population_count,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    "./results/random_select.csv",
)

