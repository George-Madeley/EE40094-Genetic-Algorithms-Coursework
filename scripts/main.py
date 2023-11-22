from GeneticAlgorithm import GeneticAlgorithm as GA
import csv
import numpy as np

def generateListOfSchema(schema_type=0, base_schema=r'00011001'):
    """
    Generate a list of schemata to test the schema theorem.
    schema_type: 0 - Vary the defining length
                 1 - Vary the order
                 2 - Vary the schema
    """
    if schema_type == 0:
        # return a schema varying the defining length
        schemata = [
            r'0......1',
            r'0.....0.',
            r'0....0..',
            r'0...1...',
            r'0..1....',
            r'0.0.....',
            r'00......',
        ]
        return schemata
    elif schema_type == 1:
        # return a schema varying the order
        schemata = [
            r'0......1',
            r'0.....01',
            r'0....001',
            r'0...1001',
            r'0..11001',
            r'0.011001',
            r'00011001',
        ]
        return schemata
    else:
        schemata = []
        values = np.arange(1, 2**8)
        for value in values:
            binary_value = np.binary_repr(value, width=8)
            schema = []
            for idx, bit in enumerate(binary_value):
                if bit == '0':
                    schema.append('.')
                else:
                    schema.append(base_schema[idx])
            schema = r"".join(schema)
            schemata.append(schema)
        return schemata

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
            csvFile,
            fieldnames=[
                "population_count",
                variation_name,
                "num_generations",
                "min fitness",
                "avg fitness"])
        csvWriter.writeheader()

    population_counts = [base * 10 **
                         exp for exp in range(1, 6) for base in range(1, 10)]
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
    variation_name,
):
    if variation_name == "mutation":
        variation_values = [mutate / 100 for mutate in range(10, 100)]
        variation_values.append(1)
    elif variation_name == "retain":
        variation_values = [retain / 100 for retain in range(2, 98)]
    elif variation_name == "random_select":
        variation_values = [random_select /
                            100 for random_select in range(0, 100)]
        
    num_of_repeats = 10

    # Run the GA for each population count
    for variation_value in variation_values:
        total_min_fitness = 0
        total_avg_fitness = 0
        total_num_generations = 0
        for rep in range(num_of_repeats):
            # Create a new GA instance
            ga = GA(
                population_count,
                individual_length,
                individual_min_value,
                individual_max_value,
            )

            # Run the GA
            min_fitness = ga.calculateMinFitness(target)
            avg_fitness = ga.calculateAverageFitness(target)
            generation = 0

            # Evolve the population until we reach the target or the max number of
            # generations
            while min_fitness > 0 and generation < max_num_generations:
                if variation_name == "mutation":
                    ga.evolve(target, mutate=variation_value)
                elif variation_name == "retain":
                    ga.evolve(target, retain=variation_value)
                elif variation_name == "random_select":
                    ga.evolve(target, random_select=variation_value)
                else:
                    ga.evolve(target)
                min_fitness = ga.calculateMinFitness(target)
                avg_fitness = ga.calculateAverageFitness(target)
                generation += 1

            total_avg_fitness += avg_fitness
            total_min_fitness += min_fitness
            total_num_generations += generation

        avg_min_fitness = total_min_fitness / num_of_repeats
        avg_avg_fitness = total_avg_fitness / num_of_repeats
        avg_num_generations = total_num_generations / num_of_repeats

        # Print the results
        min_fitness_str = "{:.2f}".format(avg_min_fitness)
        avg_fitness_str = "{:.2f}".format(avg_avg_fitness)
        print(f"population_count: {population_count}\t{variation_name}: {variation_value}\tMin Fitness: {min_fitness_str}\tAvg Fitness: {avg_fitness_str}\tGenerations: {avg_num_generations}")

        # Write the results to a CSV file
        with open(fileName, "a", newline="") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile,
                fieldnames=[
                    "population_count",
                    variation_name,
                    "num_generations",
                    "min fitness",
                    "avg fitness"])
            csvWriter.writerow(
                {
                    "population_count": population_count,
                    variation_name: variation_value,
                    "num_generations":avg_num_generations,
                    "min fitness": avg_min_fitness,
                    "avg fitness": avg_avg_fitness,
                }
            )

def vary_population_count(
    target,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    num_of_repeats,
    fileName,
):
    with open(fileName, "w", newline="") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile,
            fieldnames=[
                "population_count",
                "num_generations",
                "min fitness",
                "avg fitness"])
        csvWriter.writeheader()

    population_counts = [base * 10 **
                         exp for exp in range(5, 6) for base in range(1, 10)]
    population_counts.append(1000000)

    for population_count in population_counts:
        total_min_fitness = 0
        total_avg_fitness = 0
        total_num_generations = 0
        for repeat in range(num_of_repeats):
            ga = GA(
                population_count,
                individual_length,
                individual_min_value,
                individual_max_value,
            )

            # Run the GA
            min_fitness = ga.calculateMinFitness(target)
            avg_fitness = ga.calculateAverageFitness(target)
            fitness_history = [{"min": min_fitness, "avg": avg_fitness}]

            # Evolve the population until we reach the target or the max number
            # of generations
            while min_fitness > 0 and len(
                    fitness_history) < max_num_generations:
                ga.evolve(target)
                min_fitness = ga.calculateMinFitness(target)
                avg_fitness = ga.calculateAverageFitness(target)
                fitness_history.append(
                    {"min": min_fitness, "avg": avg_fitness})

            total_min_fitness += min_fitness
            total_avg_fitness += avg_fitness
            total_num_generations += len(fitness_history)

        min_fitness = total_min_fitness / num_of_repeats
        avg_fitness = total_avg_fitness / num_of_repeats
        avg_num_generations = total_num_generations / num_of_repeats

        # Print the results
        min_fitness_str = "{:.2f}".format(min_fitness)
        avg_fitness_str = "{:.2f}".format(avg_fitness)
        print(
            f"population_count: {population_count}\tMin Fitness: {min_fitness_str}\tAvg Fitness: {avg_fitness_str}\tGenerations: {avg_num_generations}")

        # Write the results to a CSV file
        with open(fileName, "a", newline="") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile,
                fieldnames=[
                    "population_count",
                    "num_generations",
                    "min fitness",
                    "avg fitness"])
            csvWriter.writerow(
                {
                    "population_count": population_count,
                    "num_generations": len(fitness_history),
                    "min fitness": min_fitness,
                    "avg fitness": avg_fitness,
                }
            )

def test_schema_theorem(
    target,
    individual_length,
    individual_min_value,
    individual_max_value,
    population_count,
    max_num_generations,
    fileName,
):
    fieldNames = [
        "schema",
        "order",
        "defining_length",
        "num_generations",
        "num_matches",
        "num_expected_matches",
        "schema_fitness",
        "population_fitness",
    ]
    with open(fileName, "w", newline="") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=fieldNames
        )
        csvWriter.writeheader()

    schmemata = [r'0...1001']

    for schema in schmemata:
        if len(schema) > 8*individual_length:
            print(f"Invalid schema length: {schema} {len(schema)}")
            continue

        ga = GA(
            population_count,
            individual_length,
            individual_min_value,
            individual_max_value,
        )

        generation_number = 0
        num_expected_matches = 0
        min_fitness = ga.calculateMinFitness(target)
        avg_fitness = ga.calculateAverageFitness(target)
        order = ga.calculateSchemaOrder(schema)
        defining_length = ga.calculateSchemaDefiningLength(schema)
        num_matches = ga.calculateNumIndividualsInSchema(schema)
        schema_fitness = ga.calculateAverageSchemaFitness(schema, target)

        with open(fileName, "a", newline="") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile, fieldnames=fieldNames
            )
            csvWriter.writerow(
                {
                    "schema": schema,
                    "order": order,
                    "defining_length": defining_length,
                    "num_generations": generation_number,
                    "num_matches": num_matches,
                    "num_expected_matches": num_expected_matches,
                    "schema_fitness": schema_fitness,
                    "population_fitness": avg_fitness,
                }
            )
        num_expected_matches_str = "{:.2f}".format(num_expected_matches)
        print(f"schema: {schema}\tgeneration_number: {generation_number}\tnum_matches: {num_matches}\tnum_expected_matches: {num_expected_matches_str}\tmin_fitness: {min_fitness}")
        num_expected_matches = ga.calculateExpectedNumIndividualsInSchema(
            schema,
            target
        )
        # Evolve the population until we reach the target or the max number of
        # generations
        while min_fitness > 0 and generation_number < max_num_generations and num_matches < population_count:
            ga.evolve(target)
            min_fitness = ga.calculateMinFitness(target)
            avg_fitness = ga.calculateAverageFitness(target)
            generation_number += 1
            num_matches = ga.calculateNumIndividualsInSchema(schema)
            schema_fitness = ga.calculateAverageSchemaFitness(schema, target)

            with open(fileName, "a", newline="") as csvFile:
                csvWriter = csv.DictWriter(
                    csvFile, fieldnames=fieldNames
                )
                csvWriter.writerow(
                    {
                        "schema": schema,
                        "order": order,
                        "defining_length": defining_length,
                        "num_generations": generation_number,
                        "num_matches": num_matches,
                        "num_expected_matches": num_expected_matches,
                        "schema_fitness": schema_fitness,
                        "population_fitness": avg_fitness,
                    }
                )

            num_expected_matches_str = "{:.2f}".format(num_expected_matches)
            print(f"schema: {schema}\tgeneration_number: {generation_number}\tnum_matches: {num_matches}\tnum_expected_matches: {num_expected_matches_str}\tmin_fitness: {min_fitness}")
            num_expected_matches = ga.calculateExpectedNumIndividualsInSchema(
                schema,
                target
            )

def singleTest(
    target,
    individual_length,
    individual_min_value,
    individual_max_value,
    population_count,
    max_num_generations,
):
    ga = GA(
        population_count,
        individual_length,
        individual_min_value,
        individual_max_value,
    )

    # Run the GA
    min_fitness = ga.calculateMinFitness(target)
    avg_fitness = ga.calculateAverageFitness(target)

    # Evolve the population until we reach the target or the max number
    # of generations
    generation = 0
    while min_fitness > 0 and generation < max_num_generations:
        ga.evolve(target)
        min_fitness = ga.calculateMinFitness(target)
        avg_fitness = ga.calculateAverageFitness(target)
        # Print the results
        min_fitness_str = "{:.2f}".format(min_fitness)
        avg_fitness_str = "{:.2f}".format(avg_fitness)
        print(
            f"Min Fitness: {min_fitness_str}\tAvg Fitness: {avg_fitness_str}\tGenerations: {generation}")
        generation += 1



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

# test_schema_theorem(
#     target,
#     individual_length,
#     individual_min_value,
#     individual_max_value,
#     population_count,
#     max_num_generations,
#     "./results/schema_theorem_5.csv",
# )

fileName = "./results/random_select.csv"

with open(fileName, "w", newline="") as csvFile:
    csvWriter = csv.DictWriter(
        csvFile,
        fieldnames=[
            "population_count",
            "random_select",
            "num_generations",
            "min fitness",
            "avg fitness"])
    csvWriter.writeheader()

test_function(
    target,
    population_count,
    individual_length,
    individual_min_value,
    individual_max_value,
    max_num_generations,
    fileName,
    "random_select",
)
