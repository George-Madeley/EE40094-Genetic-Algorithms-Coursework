from GeneticAlgorithm import GeneticAlgorithm as GA
import csv

def test_with_population_count(
    target,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    fileName,
    variation_name,
    test_function
):
    with open(fileName, "w", newline="") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=["population_count", variation_name, "num_generations", "min fitness", "avg fitness"]
        )
        csvWriter.writeheader()

    population_counts = [base * 10 ** exp for exp in range(1, 6) for base in range(1, 10)]
    population_counts.append(1000000)

    for population_count in population_counts:
        test_function(
            target,
            population_count,
            individual_length,
            individual_min_value,
            individual_max_value,
            max_num_generations,
            fileName,
            variation_name
        )

    
    

def test_function(
    target,
    population_count,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    fileName,
    variation_name
):
    if variation_name == "mutation":
        variation_values = [mutate/100 for mutate in range(0, 100)]
        variation_values.append(1)
    elif variation_name == "retain":
        variation_values = [retain/100 for retain in range(2, 98)]
    elif variation_name == "random_select":
        variation_values = [random_select/100 for random_select in range(0, 90)]


    # Run the GA for each population count
    for variation_value in variation_values:
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
            if variation_name == "mutation":
                ga.evolve(target, mutate=variation_value)
            elif variation_name == "retain":
                ga.evolve(target, retain=variation_value)
            elif variation_name == "random_select":
                ga.evolve(target, random_select=variation_value)
            else:
                ga.evolve(target)
            min_fitness = ga.calculateMinGrade(target)
            avg_fitness = ga.calculateAverageGrade(target)
            fitness_history.append({"min": min_fitness, "avg": avg_fitness})

        # Print the results
        min_fitness_str = "{:.2f}".format(min_fitness)
        avg_fitness_str = "{:.2f}".format(avg_fitness)
        print(f"population_count: {population_count}\t{variation_name}: {variation_value}\tMin Fitness: {min_fitness_str}\tAvg Fitness: {avg_fitness_str}\tGenerations: {len(fitness_history)}")

        # Write the results to a CSV file
        with open(fileName, "a", newline="") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile, fieldnames=["population_count", variation_name, "num_generations", "min fitness", "avg fitness"]
            )
            csvWriter.writerow(
                {
                    "population_count": population_count,
                    variation_name: variation_value,
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

tests = [
    {
        "fileName": "results/random_select_with_population.csv",
        "variation_name": "random_select"
    }
]

for test in tests:
    test_with_population_count(
        target,
        individual_length,
        individual_min_value,
        individual_max_value,
        max_num_generations,
        test["fileName"],
        test["variation_name"],
        test_function
    )

