from random import randint, random
import numpy as np
import re
import struct

class GeneticAlgorithm:
    """
    A genetic algorithm that evolves a population of individuals to find the best solution to a problem.
    """
    def __init__(self, population_size, individual_length, min, max, retain=0.2, random_select=0.05, mutate=0.01, fitness_method="scalar"):
        """
        Create a genetic algorithm.

        :param population_size: the number of individuals in the population
        :param individual_length: the number of values per individual
        :param min: the minimum possible value in an individual's list of values
        :param max: the maximum possible value in an individual's list of values
        """
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

    def addRandomIndividuals(self, parents):
        """
        Add random individuals to the population.

        :param parents: the parents to add random individuals to

        :returns: the parents to be used for the next generation
        """

        # randomly add other individuals to promote genetic diversity
        for individual in self.population:
            # random_select is the probability that the individual will
            # be added to the parents list (i.e. retained for the next
            # generation). This probability will be higher for individuals
            # with a high fitness, so these individuals are more likely to
            # be parents.
            if self.random_select > random() and len(parents) < self.population_size - 1:
                parents = np.append(
                    parents, individual.reshape(
                        (1, 6)), axis=0)
        return parents

    def calculateAverageFitness(self, target):
        """
        Find average fitness for a population.
        
        :param target: the sum of numbers that individuals are aiming for.

        :returns: the average fitness of the population
        """
        fitnesses = self.getFitness(target)
        average = np.average(fitnesses)

        # Return the average fitness of the population
        return average

    def calculateAverageSchemaFitness(self, schema, target):
        """
        Return the average fitness of the schema.

        :param schema: the schema to calculate the average fitness of
        :param target: the sum of numbers that individuals are aiming for

        :returns: the average fitness of the schema
        """
        num_of_schema_matches = self.calculateNumIndividualsInSchema(schema)
        total_schema_fitness = self.calculateTotalSchemaFitness(schema, target)
        if total_schema_fitness is None:
            return 0
        schema_fitness = total_schema_fitness / num_of_schema_matches
        return schema_fitness

    def calculateCrossoverEffect(self, schema):
        """
        Calculate the effect of crossover on the schema.

        :returns: the effect of crossover on the schema
        """
        defining_length = self.calculateSchemaDefiningLength(schema)
        pc1 = defining_length / (len(schema) - 1)
        pc = 1 - self.retain
        crossover_effect = 1 - pc * pc1
        return crossover_effect

    def calculateExpectedNumIndividualsInSchema(self, schema, target):
        """
        Calculate the expected number of individuals in the population that match the schema.
        
        :param schema: the schema to calculate the expected number of individuals in the population that match
        :param target: the sum of numbers that individuals are aiming for
        
        :returns: the expected number of individuals in the next generation that match the schema
        """
        numOfIndividualsInSchema = self.calculateNumIndividualsInSchema(schema)
        avgSchemaFitness = self.calculateAverageSchemaFitness(schema, target)
        avgPopulationFitness = self.calculateAverageFitness(target)
        crossoverEffect = self.calculateCrossoverEffect(schema)
        mutationEffect = self.calculateMutationEffect(schema)
        selectionEffect = 1 / (avgSchemaFitness / avgPopulationFitness)
        expectedNumIndividualsInSchema = numOfIndividualsInSchema * crossoverEffect * mutationEffect * selectionEffect
        return expectedNumIndividualsInSchema

    def calculateFitnesses(self, target, population=None):
        """
        Calculate the fitness of each individual in the population.

        :param target: the sum of numbers that individuals are aiming for
        :param population: the population to calculate the fitness of

        :returns: the fitness of each individual in the population

        :raises Exception: if an invalid fitness method is given
        """
        
        if population is None:
            population = self.population

        if self.fitness_method == "scalar":
            fitnesses = self.calculateFitnessesScalar(target, population)
        elif self.fitness_method == "elementwise":
            fitnesses = self.calculateFitnessesElementwise(target, population)
        elif self.fitness_method == "polynomial":
            fitnesses = self.calculateFitnessesPolynomially(target, population)
        else:
            raise Exception("Invalid fitness method: {}".format(self.fitness_method))
        return fitnesses
    
    def calculateFitnessesScalar(self, target, population):
        """
        Calculate the fitness of each individual in the population.

        :param target: the sum of numbers that individuals are aiming for
        :param population: the population to calculate the fitness of

        :returns: the fitness of each individual in the population
        """

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

        target_array = np.zeros(
            (population.shape[0],
             self.individual_length), dtype=np.float32) + target
        fitness = np.abs(target_array - population)
        fitness = np.sum(fitness, axis=1)
        return fitness

    def calculateFitnessesPolynomially(self, target, population, min_x=-100, max_x=100, step=1):
        """
        Calculate the fitness of each individual in the population.

        :param target: the sum of numbers that individuals are aiming for
        :param population: the population to calculate the fitness of
        :param min_x: the minimum x value to calculate the fitness for
        :param max_x: the maximum x value to calculate the fitness for
        :param step: the step size to use when calculating the fitness

        :returns: the fitness of each individual in the population
        """

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

    def calculateMinFitness(self, target):
        """
        Find minimum fitness for a population.
        
        :param target: the sum of numbers that individuals are aiming for
        
        :returns: the minimum fitness of the population
        """
        fitnesses = self.getFitness(target)
        minGrade = np.min(fitnesses)

        # Return the min fitness of the population
        return minGrade

    def calculateMutationEffect(self, schema):
        """
        Calculate the effect of mutation on the schema.

        :param schema: the schema to calculate the effect of mutation on
        :param mutate: the probability of mutation for each individual gene

        :returns: the effect of mutation on the schema
        """
        schema_order = self.calculateSchemaOrder(schema)
        return (1 - self.mutation_probability) ** schema_order

    def calculateNumIndividualsInSchema(self, schema):
        """
        Return the number of individuals in the population that match the schema.

        :param schema: the schema to calculate the number of individuals in the population that match

        :returns: the number of individuals in the population that match the schema
        """
        num_matches = 0
        for individual in self.population:
            if self.isIndividualMemberOfSchema(individual, schema):
                num_matches += 1
        return num_matches

    def calculateSchemaDefiningLength(self, schema):
        """
        Return the defining length of the schema.

        :param schema: the schema to calculate the defining length of

        :returns: the defining length of the schema

        :raises Exception: if the schema contains invalid characters
        """
        # Check that the schema only contains 0s, 1s, and .
        if re.match("^[01.]+$", schema) is None:
            raise Exception("Invalid schema: {}. Use \'.\' instead of \'*\'".format(schema))

        defining_length = len(schema.strip(".")) - 1
        return defining_length

    def calculateSchemaOrder(self, schema):
        """
        Return the order of the schema.
        
        :param schema: the schema to calculate the order of
        
        :returns: the order of the schema

        :raises Exception: if the schema contains invalid characters
        """
        # Check that the schema only contains 0s, 1s, and .
        if re.match("^[01.]+$", schema) is None:
            raise Exception("Invalid schema: {}. Use \'.\' instead of \'*\'".format(schema))

        schema_order = len(schema) - schema.count(".")
        return schema_order

    def calculateTotalSchemaFitness(self, schema, target):
        """
        Return the total fitness of the schema.

        :param schema: the schema to calculate the total fitness of
        :param target: the sum of numbers that individuals are aiming for

        :returns: the total fitness of the schema
        """
        # Stores the index of the individual in the population
        # that are in the schema.
        total_schema_fitness = 0
        individual_with_fitness = zip(self.population, self.getFitness(target))
        for individual, fitness in individual_with_fitness:
            if self.isIndividualMemberOfSchema(individual, schema):
                total_schema_fitness += fitness
        return total_schema_fitness

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
            raise Exception("Invalid crossover method: {}".format(crossover_method))
        return children

    def crossoverElementwise(self, parents):
        """
        Crossover parents to create children.
        
        :param parents: the parents to create children from

        :returns: the children created from the parents
        """
        # crossover parents to create children
        numParents = len(parents)
        numDesiredChildren = self.population_size - numParents
        children = []
        while len(children) < numDesiredChildren:
            # Get a random mom and dad.
            dadIndex = randint(0, numParents - 1)
            momIndex = randint(0, numParents - 1)

            # Since we don't want to use the same
            # mom and dad multiple times, make sure
            # they aren't the same parent.
            if dadIndex != momIndex:
                dad = parents[dadIndex]
                mom = parents[momIndex]
                halfIndex = int(self.individual_length / 2)
                # Create a child.
                child = np.concatenate((dad[:halfIndex], mom[halfIndex:]))
                children.append(child)
        return np.array(children)
    
    def crossoverBitwise(self, parents):
        """
        Crossover parents to create children.
        
        :param parents: the parents to create children from

        :returns: the children created from the parents
        """
        # crossover parents to create children
        numParents = len(parents)
        numDesiredChildren = self.population_size - numParents
        children = []
        while len(children) < numDesiredChildren:
            # Get a random mom and dad.
            dadIndex = randint(0, numParents - 1)
            momIndex = randint(0, numParents - 1)

            # Since we don't want to use the same
            # mom and dad multiple times, make sure
            # they aren't the same parent.
            if dadIndex != momIndex:
                dad = parents[dadIndex]
                mom = parents[momIndex]
                binary_dad = ''.join([self.getBinaryRepresentationOfGene(gene) for gene in dad])
                binary_mom = ''.join([self.getBinaryRepresentationOfGene(gene) for gene in mom])
                halfIndex = int(len(binary_dad) / 2)
                # Create a child.
                binary_child = binary_dad[:halfIndex] + binary_mom[halfIndex:]
                child_binaries = [binary_child[i:i+8] for i in range(0, len(binary_child), 8)]
                child = [self.getSignedIntegerRepresentationOfGene(binary_gene) for binary_gene in child_binaries]
                children.append(child)
        return np.array(children)

    def evolve(self, target, selection_method="rank", crossover_method="bitwise", elitism=False):
        """
        Evolve a population some number of generations.

        :param target: the sum of numbers that individuals are aiming for
        :param selection_method: the method to use to select parents
        :param crossover_method: the method to use to crossover parents
        :param elitism: whether or not to use elitism when evolving the population
        """
        # For each individual in the population, calculate the fitness
        # and sort the population in order of ascending fitness
        parents = self.selectParents(target, selection_method)
        parents = self.addRandomIndividuals(parents)
        children = self.crossover(parents, crossover_method)

        # If elitism is enabled, preserve the parents.
        if elitism:
            mutated_children = self.mutate(children)
            new_population = np.append(parents, mutated_children, axis=0)
        else:
            new_population = np.append(parents, children, axis=0)
            new_population = self.mutate(new_population)

        self.population = new_population
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
            raise Exception("Invalid selection method: {}".format(selection_method))
        return parents

    def rankPopulation(self, target):
        fitnesss = self.getFitness(target)
        sortedfitnesssIndices = np.argsort(fitnesss)
        sortedIndividuals = self.population[sortedfitnesssIndices]
        # Calculate the number of individuals to retain, based on the retain
        # parameter. Using ceil() prevents us from retaining 0 individuals
        # when the retain parameter is low. I.e. it makes sure we always
        # retain at least one individual.
        retain_length = np.floor(self.population_size * self.retain)
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
        parents = np.random.choice(self.population, size=retain_length, p=probabilities)
        return parents

    def generatePopulation(self):
        """
        Create a number of individuals (i.e. a population).

        :param count: the number of individuals in the population
        :param length: the number of values per individual
        :param min: the minimum possible value in an individual's list of values
        :param max: the maximum possible value in an individual's list of values

        :returns: a population of individuals
        """
        return np.random.randint(
            self.min,
            self.max,
            (self.population_size,
             self.individual_length),
            dtype=np.int8)

    def getBinaryRepresentationOfGene(self, gene):
        """
        Return an 8-bit signed binary representation of the given gene.

        :param gene: the gene to get the binary representation of

        :returns: the binary representation of the given gene
        """
        if not (-127 <= gene <= 127):
            raise Exception("Gene must be between -127 and 127")
        
        binary_gene = np.binary_repr(gene, width=8)
        return binary_gene

    def getSignedIntegerRepresentationOfGene(self, gene):
        """
        Return the signed integer representation of the given gene.

        :param gene: the gene to get the signed integer representation of

        :returns: the signed integer representation of the given gene
        """
        signed_gene = struct.unpack('b', struct.pack('B', int(gene, 2)))[0]
        return signed_gene

    def getFitness(self, target):
        """
        Return the fitness of each individual in the population.
        
        :param target: the sum of numbers that individuals are aiming for
        
        :returns: the fitness of each individual in the population
        """
        self.fitness = self.fitness if isinstance(
            self.fitness, np.ndarray) else self.calculateFitnesses(target)
        return self.fitness

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
            raise Exception("Invalid schema: {}. Use \'.\' instead of \'*\'".format(schema))
        
        return re.match(schema, gene) is not None

    def isIndividualMemberOfSchemaGenewise(self, individual, schema):
        """
        Return True if individual is a member of schema, False otherwise.

        :param individual: the individual to check if it is a member of the schema
        :param schema: the schema to check if the individual is a member of

        :returns: True if individual is a member of schema, False otherwise
        """
        for gene in individual:
            binary_gene = self.getBinaryRepresentationOfGene(gene)
            if self.isGeneMemberOfSchema(binary_gene, schema):
                return True
        return False
    
    def isIndividualMemberOfSchema(self, individual, schema):
        """
        Return True if individual is a member of schema, False otherwise.

        :param individual: the individual to check if it is a member of the schema
        :param schema: the schema to check if the individual is a member of

        :returns: True if individual is a member of schema, False otherwise
        """
        binary_individual = ''.join([self.getBinaryRepresentationOfGene(gene) for gene in individual])
        isMember = self.isGeneMemberOfSchema(binary_individual, schema)
        return isMember

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
