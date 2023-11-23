from random import randint, random
import numpy as np
import re
import struct


class GeneticAlgorithm:
    """
    A genetic algorithm that evolves a population of individuals to find the best solution to a problem.
    """

    def __init__(
            self,
            population_size,
            individual_length,
            min,
            max,
            retain=0.2,
            random_select=0.05,
            mutate=0.01,
            fitness_method="scalar",
            x_value_size=100):
        """
        Create a genetic algorithm.

        :param population_size: the number of individuals in the population
        :param individual_length: the number of values per individual
        :param min: the minimum possible value in an individual's list of values
        :param max: the maximum possible value in an individual's list of values
        :param retain: the percentage of individuals to retain from the previous generation
        :param random_select: the probability of retaining a non-fittest individual
        :param mutate: the probability of mutation for each individual gene
        :param fitness_method: the method to use to calculate the fitness of each individual
        :param x_value_size: the number of x values to use when calculating the fitness of each individual polynomially
        """
        # Store constructor variables
        self.population_size = population_size
        self.individual_length = individual_length
        self.min = min
        self.max = max
        self.retain = retain
        self.random_select = random_select
        self.mutation_probability = mutate
        self.population = self.generatePopulation()
        self.fitness = 0
        self.fitness_method = fitness_method
        self.x_value_size = x_value_size

##########################################################################
# Population Evolution
##########################################################################

    def generatePopulation(self):
        """
        Create a number of individuals (i.e. a population).

        :returns: a population of individuals
        """
        # Create a population of random individuals.
        return np.random.randint(
            self.min,
            self.max,
            (self.population_size,
             self.individual_length),
            dtype=np.int8)

    def evolve(
            self,
            target,
            selection_method="rank",
            crossover_method="bitwise",
            elitism=False):
        """
        Evolve a population some number of generations.

        :param target: the sum of numbers that individuals are aiming for
        :param selection_method: the method to use to select parents
        :param crossover_method: the method to use to crossover parents
        :param elitism: whether or not to use elitism when evolving the population
        """
        # Select parents from the population.
        parents = self.selectParents(target, selection_method)
        # Add random individuals to the parents list.
        parents = self.addRandomIndividuals(parents)
        # Crossover parents to create children.
        children = self.crossover(parents, crossover_method)

        # If elitism is enabled, preserve the parents.
        if elitism:
            # Mutate the children.
            mutated_children = self.mutate(children)
            # Create the new population by combining the parents and children.
            new_population = np.append(parents, mutated_children, axis=0)
        else:
            # Create the new population by combining the parents and children.
            new_population = np.append(parents, children, axis=0)
            # Mutate the new population.
            new_population = self.mutate(new_population)

        # Update the population.
        self.population = new_population
        # Reset the fitness of the population.
        self.fitness = 0

    def selectParents(self, target, selection_method):
        """
        Select parents from the population.

        :param target: the sum of numbers that individuals are aiming for
        :param selection_method: the method to use to select parents

        :returns: the parents selected from the population

        :raises Exception: if an invalid selection method is given
        """
        if selection_method == "rank":
            parents = self.rankPopulation(target)
        elif selection_method == "roulette":
            parents = self.rouletteWheelSelection(target)
        else:
            # If an invalid selection method is given, raise an exception.
            raise Exception(
                "Invalid selection method: {}".format(selection_method))
        return parents

    def rankPopulation(self, target):
        """
        Select parents using rank selection.

        :param target: the sum of numbers that individuals are aiming for

        :returns: the parents selected using rank selection
        """
        # Calculate the fitness of each individual in the population.
        fitnesss = self.getFitness(target)
        # Sort the fitnesss in ascending order.
        sortedfitnesssIndices = np.argsort(fitnesss)
        sortedIndividuals = self.population[sortedfitnesssIndices]
        # Calculate the number of individuals to retain, based on the retain
        # parameter. Using ceil() prevents us from retaining 0 individuals
        # when the retain parameter is low. I.e. it makes sure we always
        # retain at least one individual.
        retain_length = np.ceil(self.population_size * self.retain)
        retain_length = int(retain_length) if retain_length > 1 else 2

        # Retain the best individuals
        parents = sortedIndividuals[:retain_length]
        return parents

    def rouletteWheelSelection(self, target):
        """
        Select parents using roulette wheel selection.

        :param target: the sum of numbers that individuals are aiming for

        :returns: the parents selected using roulette wheel selection
        """
        # Calculate the number of individuals to retain, based on the retain
        # parameter. Using ceil() prevents us from retaining 0 individuals
        # when the retain parameter is low. I.e. it makes sure we always
        # retain at least two individuals to create children.
        retain_length = np.ceil(self.population_size * self.retain)
        retain_length = int(retain_length) if retain_length > 1 else 2

        # Calculate the fitness of each individual in the population.
        fitnesses = self.getFitness(target)
        # Calculate the probability of each individual being selected for
        # retention. As lower fitnesses are better, we need to invert the
        # fitnesses to get the probabilities.
        fitness_sum = np.sum(fitnesses)
        inverse_fitnesses = fitness_sum / fitnesses
        probabilities = inverse_fitnesses / np.sum(inverse_fitnesses)

        # Select the parents using roulette wheel selection.
        parents = np.random.choice(
            self.population,
            size=retain_length,
            p=probabilities)
        return parents

    def getFitness(self, target):
        """
        Return the fitness of each individual in the population.

        :param target: the sum of numbers that individuals are aiming for

        :returns: the fitness of each individual in the population
        """

        # Check if the fitness of the population has already been calculated.
        # If it has, return the fitness. Otherwise, calculate the fitness.
        self.fitness = self.fitness if isinstance(
            self.fitness, np.ndarray) else self.calculateFitnesses(target)
        return self.fitness

    def calculateFitnesses(self, target, population=None):
        """
        Calculate the fitness of each individual in the population.

        :param target: the sum of numbers that individuals are aiming for
        :param population: the population to calculate the fitness of

        :returns: the fitness of each individual in the population

        :raises Exception: if an invalid fitness method is given
        """
        # If no population is given, use the current population.
        if population is None:
            population = self.population

        # Calculate the fitness of each individual in the population.
        if self.fitness_method == "scalar":
            fitnesses = self.calculateFitnessesScalar(target, population)
        elif self.fitness_method == "elementwise":
            fitnesses = self.calculateFitnessesElementwise(target, population)
        elif self.fitness_method == "polynomial":
            fitnesses = self.calculateFitnessesPolynomially(target, population)
        else:
            # If an invalid fitness method is given, raise an exception.
            raise Exception(
                "Invalid fitness method: {}".format(
                    self.fitness_method))
        return fitnesses

    def calculateFitnessesScalar(self, target, population):
        """
        Calculate the fitness of each individual in the population.

        :param target: the sum of numbers that individuals are aiming for
        :param population: the population to calculate the fitness of

        :returns: the fitness of each individual in the population
        """
        # Calculate the fitness of each individual in the population by summing
        # an individual's values and subtracting the sum from the target.
        target = np.full(population.shape[0], target)
        fitness = np.abs(target - np.sum(population, axis=1))
        return fitness

    def calculateFitnessesElementwise(self, target, population):
        """
        Calculate the fitness of each individual in the population.

        :param target: the sum of numbers that individuals are aiming for
        :param population: the population to calculate the fitness of

        :returns: the fitness of each individual in the population
        """

        # Calculate the fitness of each individual in the population by summing
        # the absolute difference between each value in an individual and the
        # target value.
        target_array = np.zeros(
            (population.shape[0],
             self.individual_length), dtype=np.float32) + target
        fitness = np.abs(target_array - population)
        fitness = np.sum(fitness, axis=1)
        return fitness

    def calculateFitnessesPolynomially(
            self,
            target,
            population,
            min_x=-100,
            max_x=100):
        """
        Calculate the fitness of each individual in the population.

        :param target: the sum of numbers that individuals are aiming for
        :param population: the population to calculate the fitness of
        :param min_x: the minimum x value to calculate the fitness for
        :param max_x: the maximum x value to calculate the fitness for

        :returns: the fitness of each individual in the population
        """
        # calculate the x values to use when calculating the fitness
        step = (abs(min_x) + abs(max_x)) / self.x_value_size
        x_values = np.arange(min_x, max_x, step, dtype=np.float64)

        # calculate the y values for each x value using a polynomial equation
        # where the coefficients are the elements in target
        expected_y_values = np.polyval(target, x_values)

        # calculate the y values for each x value using a polynomial equation
        # where the coefficients are the elements in the individual
        actual_y_values = np.zeros(
            (population.shape[0],
             len(x_values)),
            dtype=np.float64)
        for individualIndex in range(population.shape[0]):
            individual = population[individualIndex]
            actual_y_values[individualIndex] = np.polyval(individual, x_values)

        # calculate the fitness of each individual by summing the absolute
        # difference between the expected y values and the actual y values
        fitness = np.abs(expected_y_values - actual_y_values)
        fitness = np.sum(fitness, axis=1)
        fitness = fitness / len(x_values)
        return fitness

    def addRandomIndividuals(self, parents):
        """
        Add random individuals to the population.

        :param parents: the parents to add random individuals to

        :returns: the parents to be used for the next generation
        """

        # randomly add individuals to the parents list
        for individual in self.population:
            # random_select is the probability that the individual will
            # be added to the parents list (i.e. retained for the next
            # generation).
            if self.random_select > random() and len(parents) < self.population_size - 1:
                parents = np.append(
                    parents, individual.reshape(
                        (1, 6)), axis=0)
        return parents

    def crossover(self, parents, crossover_method="bitwise"):
        """
        Crossover parents to create children.

        :param parents: the parents to create children from
        :param crossover_method: the method to use to crossover parents

        :returns: the children created from the parents

        :raises Exception: if an invalid crossover method is given
        """
        if crossover_method == "elementwise":
            children = self.crossoverElementwise(parents)
        elif crossover_method == "bitwise":
            children = self.crossoverBitwise(parents)
        else:
            # If an invalid crossover method is given, raise an exception.
            raise Exception(
                "Invalid crossover method: {}".format(crossover_method))
        return children

    def crossoverElementwise(self, parents):
        """
        Crossover parents to create children.

        :param parents: the parents to create children from

        :returns: the children created from the parents
        """
        # calculate the number of parents and children
        numParents = len(parents)
        numDesiredChildren = self.population_size - numParents
        children = []
        while len(children) < numDesiredChildren:
            # Get a random mom and dad.
            dadIndex = randint(0, numParents - 1)
            momIndex = randint(0, numParents - 1)

            # Make sure the mom and dad are not the same parent.
            if dadIndex != momIndex:
                # Get the mom and dad.
                dad = parents[dadIndex]
                mom = parents[momIndex]
                halfIndex = int(self.individual_length / 2)
                # Create a child by taking the first half of the dad and the
                # second half of the mom.
                child = np.concatenate((dad[:halfIndex], mom[halfIndex:]))
                children.append(child)
        return np.array(children)

    def crossoverBitwise(self, parents):
        """
        Crossover parents to create children.

        :param parents: the parents to create children from

        :returns: the children created from the parents
        """
        # calculate the number of parents and children
        numParents = len(parents)
        numDesiredChildren = self.population_size - numParents
        children = []
        while len(children) < numDesiredChildren:
            # Get a random mom and dad index.
            dadIndex = randint(0, numParents - 1)
            momIndex = randint(0, numParents - 1)

            # Make sure the mom and dad are not the same parent.
            if dadIndex != momIndex:
                # Get the mom and dad.
                dad = parents[dadIndex]
                mom = parents[momIndex]
                # Convert the mom and dad to binary.
                binary_dad = ''.join(
                    [self.getBinaryRepresentationOfGene(gene) for gene in dad])
                binary_mom = ''.join(
                    [self.getBinaryRepresentationOfGene(gene) for gene in mom])
                # Get the index to split the mom and dad at.
                halfIndex = int(len(binary_dad) / 2)
                # Create a child by taking the first half of the dad and the
                # second half of the mom.
                binary_child = binary_dad[:halfIndex] + binary_mom[halfIndex:]
                # Convert the child to a list of genes.
                child_binaries = [binary_child[i:i + 8]
                                  for i in range(0, len(binary_child), 8)]
                child = [self.getSignedIntegerRepresentationOfGene(
                    binary_gene) for binary_gene in child_binaries]
                children.append(child)
        return np.array(children)

    def getBinaryRepresentationOfGene(self, gene):
        """
        Return an 8-bit signed binary representation of the given gene.

        :param gene: the gene to get the binary representation of

        :returns: the binary representation of the given gene
        """
        # Check that the gene is between -127 and 127.
        if not (-127 <= gene <= 127):
            # If the gene is not between -127 and 127, raise an exception.
            raise Exception("Gene must be between -127 and 127")

        # Convert the gene to an 8-bit signed binary representation.
        binary_gene = np.binary_repr(gene, width=8)
        return binary_gene

    def getSignedIntegerRepresentationOfGene(self, gene):
        """
        Return the signed integer representation of the given gene.

        :param gene: the gene to get the signed integer representation of

        :returns: the signed integer representation of the given gene
        """
        # Convert the binary gene to a signed integer.
        signed_gene = struct.unpack('b', struct.pack('B', int(gene, 2)))[0]
        return signed_gene

    def mutate(self, population):
        """
        Mutate some individuals.

        :param population: the population to mutate

        :returns: the mutated parents
        """
        # mutate some individuals
        for individualIndex in range(population.shape[0]):
            # mutate is the probability that each gene will be randomly
            # changed. This probability will be higher for individuals with
            # a low fitness, so these individuals are more likely to be
            # mutated.
            if self.mutation_probability > random():
                mutateIndex = randint(0, self.individual_length - 1)
                # this mutation is not ideal, because it
                # restricts the range of possible values,
                # but the function is unaware of the min/max
                # values used to create the individuals,
                # so it'll have to do for now
                population[individualIndex, mutateIndex] = randint(
                    self.min, self.max)
        return population

##########################################################################
# Population Analysis
##########################################################################

    def calculateAverageFitness(self, target):
        """
        Find average fitness for a population.

        :param target: the sum of numbers that individuals are aiming for.

        :returns: the average fitness of the population
        """
        # Calculate the fitness of each individual in the population.
        fitnesses = self.getFitness(target)
        # Calculate the average fitness of the population.
        average = np.average(fitnesses)

        # Return the average fitness of the population
        return average

    def calculateMinFitness(self, target):
        """
        Find minimum fitness for a population.

        :param target: the sum of numbers that individuals are aiming for

        :returns: the minimum fitness of the population
        """
        # Calculate the fitness of each individual in the population.
        fitnesses = self.getFitness(target)
        # Calculate the minimum fitness of the population.
        minGrade = np.min(fitnesses)
        return minGrade

##########################################################################
# Schema Analysis
##########################################################################

    def calculateExpectedNumIndividualsInSchema(self, schema, target):
        """
        Calculate the expected number of individuals in the population that match the schema.

        :param schema: the schema to calculate the expected number of individuals in the population that match
        :param target: the sum of numbers that individuals are aiming for

        :returns: the expected number of individuals in the next generation that match the schema
        """
        # Calculate the number of individuals in the population that match
        # the schema.
        numOfIndividualsInSchema = self.calculateNumIndividualsInSchema(schema)
        # Calculate the average fitness of the schema.
        avgSchemaFitness = self.calculateAverageSchemaFitness(schema, target)
        # Calculate the average fitness of the population.
        avgPopulationFitness = self.calculateAverageFitness(target)
        # Calculate the effect of crossover on the schema.
        crossoverEffect = self.calculateCrossoverEffect(schema)
        # Calculate the effect of mutation on the schema.
        mutationEffect = self.calculateMutationEffect(schema)
        # Calculate the effect of selection on the schema. This number must be
        # larger than 1 if the schema is to survive to the next generation. As
        # lower fitnesses are better, we need to invert the fitnesses to get
        # the selection effect.
        selectionEffect = 1 / (avgSchemaFitness / avgPopulationFitness)
        # Calculate the expected number of individuals in the next generation
        expectedNumIndividualsInSchema = numOfIndividualsInSchema * \
            crossoverEffect * mutationEffect * selectionEffect
        return expectedNumIndividualsInSchema

    def calculateNumIndividualsInSchema(self, schema):
        """
        Return the number of individuals in the population that match the schema.

        :param schema: the schema to calculate the number of individuals in the population that match

        :returns: the number of individuals in the population that match the schema
        """
        # Calculate the number of individuals in the population that match
        # the schema.
        num_matches = 0
        for individual in self.population:
            # Check if the individual is a member of the schema.
            if self.isIndividualMemberOfSchema(individual, schema):
                num_matches += 1
        return num_matches

    def isIndividualMemberOfSchema(self, individual, schema):
        """
        Return True if individual is a member of schema, False otherwise.

        :param individual: the individual to check if it is a member of the schema
        :param schema: the schema to check if the individual is a member of

        :returns: True if individual is a member of schema, False otherwise
        """
        # Get the binary representation of the individual.
        binary_individual = ''.join(
            [self.getBinaryRepresentationOfGene(gene) for gene in individual])
        # Check if the individual is a member of the schema.
        isMember = self.isGeneMemberOfSchema(binary_individual, schema)
        return isMember

    def isGeneMemberOfSchema(self, gene, schema):
        """
        Return True if gene is a member of schema, False otherwise.

        :param gene: the gene to check if it is a member of the schema
        :param schema: the schema to check if the gene is a member of

        :returns: True if gene is a member of schema, False otherwise

        :raises Exception: if the schema contains invalid characters
        """
        # Check that the schema only contains 0s, 1s, and .
        if re.match("^[01.]+$", schema) is None:
            # If the schema contains invalid characters, raise an exception.
            raise Exception(
                "Invalid schema: {}. Use \'.\' instead of \'*\'".format(schema))
        # Check if the gene is a member of the schema.
        return re.match(schema, gene) is not None

    def calculateAverageSchemaFitness(self, schema, target):
        """
        Return the average fitness of the schema.

        :param schema: the schema to calculate the average fitness of
        :param target: the sum of numbers that individuals are aiming for

        :returns: the average fitness of the schema
        """
        # Calculate the number of individuals in the population that match
        # the schema.
        num_of_schema_matches = self.calculateNumIndividualsInSchema(schema)
        # Calculate the total fitness of the schema.
        total_schema_fitness = self.calculateTotalSchemaFitness(schema, target)
        # If there are no individuals in the population that match the schema,
        # return 0.
        if total_schema_fitness is None:
            return 0
        # Calculate the average fitness of the schema.
        schema_fitness = total_schema_fitness / num_of_schema_matches
        return schema_fitness

    def calculateTotalSchemaFitness(self, schema, target):
        """
        Return the total fitness of the schema.

        :param schema: the schema to calculate the total fitness of
        :param target: the sum of numbers that individuals are aiming for

        :returns: the total fitness of the schema
        """
        # Calculate the total fitness of the schema by summing the fitness of
        # each individual in the population that matches the schema.
        total_schema_fitness = 0
        individual_with_fitness = zip(self.population, self.getFitness(target))
        for individual, fitness in individual_with_fitness:
            # Check if the individual is a member of the schema.
            if self.isIndividualMemberOfSchema(individual, schema):
                total_schema_fitness += fitness
        return total_schema_fitness

    def calculateCrossoverEffect(self, schema):
        """
        Calculate the effect of crossover on the schema.

        :param schema: the schema to calculate the effect of crossover on

        :returns: the effect of crossover on the schema
        """
        # Calculate the defining length of the schema.
        defining_length = self.calculateSchemaDefiningLength(schema)
        # Calculate the probability of crossover for each gene.
        pc1 = defining_length / (len(schema) - 1)
        pc = 1 - self.retain
        crossover_effect = 1 - pc * pc1
        return crossover_effect

    def calculateSchemaDefiningLength(self, schema):
        """
        Return the defining length of the schema.

        :param schema: the schema to calculate the defining length of

        :returns: the defining length of the schema

        :raises Exception: if the schema contains invalid characters
        """
        # Check that the schema only contains 0s, 1s, and .
        if re.match("^[01.]+$", schema) is None:
            # If the schema contains invalid characters, raise an exception.
            raise Exception(
                "Invalid schema: {}. Use \'.\' instead of \'*\'".format(schema))

        # Calculate the defining length of the schema.
        defining_length = len(schema.strip(".")) - 1
        return defining_length

    def calculateMutationEffect(self, schema):
        """
        Calculate the effect of mutation on the schema.

        :param schema: the schema to calculate the effect of mutation on

        :returns: the effect of mutation on the schema
        """
        # Calculate the order of the schema.
        schema_order = self.calculateSchemaOrder(schema)
        # Calculate the effect of mutation on the schema.
        mutation_effect = (1 - self.mutation_probability) ** schema_order
        return mutation_effect

    def calculateSchemaOrder(self, schema):
        """
        Return the order of the schema.

        :param schema: the schema to calculate the order of

        :returns: the order of the schema

        :raises Exception: if the schema contains invalid characters
        """
        # Check that the schema only contains 0s, 1s, and .
        if re.match("^[01.]+$", schema) is None:
            # If the schema contains invalid characters, raise an exception.
            raise Exception(
                "Invalid schema: {}. Use \'.\' instead of \'*\'".format(schema))

        # Calculate the order of the schema.
        schema_order = len(schema) - schema.count(".")
        return schema_order
