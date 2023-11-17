from random import randint, random
import numpy as np


class GeneticAlgorithm:
    def __init__(self, population_size, individual_length, min, max):
        self.population_size = population_size
        self.individual_length = individual_length
        self.min = min
        self.max = max
        self.population = self.generatePopulation()
        self.errors = 0

    def generatePopulation(self):
        """
        Create a number of individuals (i.e. a population).

        count: the number of individuals in the population
        length: the number of values per individual
        min: the minimum possible value in an individual's list of values
        max: the maximum possible value in an individual's list of values

        """
        return np.random.randint(self.min, self.max, (self.population_size, self.individual_length))

    def calculateAverageGrade(self, target):
        "Find average error for a population."
        # Calculate the error of each individual in the population
        # and sum them together. The error is the sum of the differences
        # between each individual's sum and the target
        errors = self.errors if type(self.errors) == np.ndarray else self.calculateErrors(target)
        average = np.average(errors)

        # Return the average error of the population
        return average
    
    def calculateMinGrade(self, target):
        "Find minimum error for a population."
        # Calculate the error of each individual in the population
        # and sum them together. The error is the sum of the differences
        # between each individual's sum and the target
        errors = self.errors if type(self.errors) == np.ndarray else self.calculateErrors(target)
        minGrade = np.min(errors)

        # Return the min error of the population
        return minGrade

    def calculateErrors(self, target):
        target = np.full(self.population_size, target)
        self.errors = np.abs(target - np.sum(self.population, axis=1))
        return self.errors

    def evolve(self, target, retain=0.2, random_select=0.05, mutate=0.01):
        """
        Evolve a population some number of generations.
        
        population: the population to evolve
        target: the sum of numbers that individuals are aiming for
        retain: the portion of the population that should be retained without change between generations
        random_select: the portion of the population that should randomly selected for retention
        mutate: the probability of mutation for each individual gene
        """
        # For each individual in the population, calculate the fitness
        # and sort the population in order of ascending fitness
        errors = self.errors if type(self.errors) == np.ndarray else self.calculateErrors(target)
        sortedErrorsIndices = np.argsort(errors)
        sortedIndividuals = self.population[sortedErrorsIndices]
        # Calculate the number of individuals to retain, based on the retain
        # parameter. Using ceil() prevents us from retaining 0 individuals
        # when the retain parameter is low. I.e. it makes sure we always
        # retain at least one individual.
        retain_length = np.floor(self.population_size * retain)
        retain_length = int(retain_length) if retain_length > 1 else 2

        # Retain the best individuals
        parents = sortedIndividuals[:retain_length]

        # randomly add other individuals to promote genetic diversity
        for individual in sortedIndividuals[retain_length:]:
            # random_select is the probability that the individual will
            # be added to the parents list (i.e. retained for the next
            # generation). This probability will be higher for individuals
            # with a high fitness, so these individuals are more likely to
            # be parents.
            if random_select > random() and len(parents) < self.population_size - 1:
                parents = np.append(parents, individual.reshape((1, 6)), axis=0)

        # mutate some individuals
        for individualIndex in range(parents.shape[0]):
            # mutate is the probability that each gene will be randomly
            # changed. This probability will be higher for individuals with
            # a low fitness, so these individuals are more likely to be mutated.
            if mutate > random():
                mutateIndex = randint(0, self.individual_length - 1)
                # this mutation is not ideal, because it
                # restricts the range of possible values,
                # but the function is unaware of the min/max
                # values used to create the individuals,
                # so it'll have to do for now
                parents[individualIndex, mutateIndex] = randint(self.min, self.max)

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

        # Add children to the parents to create the next generation.
        new_population = np.append(parents, children, axis=0)
        self.population = new_population
        self.errors = 0
