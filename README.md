# Genetic Algorithms | George Madeley

|University of Bath|
| :- |
|Genetic Algorithms|
|EE40098: Computational Intelligence|

|<p>George Madeley</p><p>11-24-2023</p>|
| :- |

## <a name="_toc151734468"></a>**ABSTRACT**

An investigation of genetic algorithms was conducted varying the population size, probability of mutation, probability of random selection, percentage retained, fitness functions, and selection methods. Results showed that larger populations reduce the number of generations compared to smaller populations. Probability of mutations values between 0.2 and 0.9 are preferred, same with probability of retain and percentage of random selection. There is negligible difference between using ranking selection compared to roulette wheel selection however, using elitism shows improvement of near 10% due to protection of individuals with better fitness values.

Hollands Schema Theorem was investigated by using multiple different schemata, varying their order, and defining length. Results proved Holland’s Schema Theorem that the frequency of schema with above average fitness will increase exponentially. However, results also show the limitations of Holland’s Schema Theorem with regards to finite population size.

## **CONTENTS**

[Abstract](#_toc151734468)

[Table of Equations](#_toc151734469)

[Table of Figures](#_toc151734470)

[Table of tables](#_toc151734471)

[Introduction](#_toc151734472)

[Design](#_toc151734473)

[Results](#_toc151734474)

- [Exercise 2: Varying Parameters](#_toc151734475)

  - [Varying Population Size](#_toc151734476)

  - [Varying Probability of Mutation](#_toc151734477)

  - [Varying Percentage of Retained Individuals](#_toc151734478)

  - [Varying Probability of Random Selection](#_toc151734479)

  - [Varying Selection Methods](#_toc151734480)

- [Exercise 3: Algorithm Termination](#_toc151734481)

- [Exercise 4: Optimising Parameters for a 5<sup>th</sup>-order Polynomial](#_toc151734482)

  - [Optimising Population Size](#_toc151734483)

  - [Optimising Probability of Mutation](#_toc151734484)

  - [Optimising Percentage of Retained Individuals](#_toc151734485)

  - [Optimising Probability of Random Selection](#_toc151734486)

  - [Varying Fitness Function](#_toc151734487)

- [Exercise 5: Proving Hollands Schema Theorem](#_toc151734488)

  - [Varying Defining Length](#_toc151734489)

  - [Varying Order](#_toc151734490)

  - [Proving Hollands Schema Theorem.](#_toc151734491)

[Conclusion](#_toc151734492)

[References](#_toc151734493)

## <a name="_toc151734469"></a>**TABLE OF EQUATIONS**

[Equation 1 Probability of selection of an individual for roulette wheel selection.](#_toc151734496)

[Equation 2 Fifth-order polynomial used to calculate the fitness of an individual.](#_toc151734497)

[Equation 3 Sum of the absolute elementwise difference between two vectors.](#_toc151734498)

[Equation 4 Sum of the absolute difference between a series of target y-values and actual y-values.](#_toc151734499)

[Equation 5 Hollands Schema Theorem.](#_toc151734500)

## <a name="_toc151734470"></a>**TABLE OF FIGURES**

[Figure 1 Number of generations to produce an individual with a fitness of zero over each population value.](#_toc151734501)

[Figure 2 Number of generations to produce an individual with a fitness of zero over each probability of mutation.](#_toc151734502)

[Figure 3 Number of generations to produce an individual with a fitness of zero over percentage of retained individuals.](#_toc151734503)

[Figure 4 Number of generations to produce an individual with a fitness of zero over the probability of random selection.](#_toc151734504)

[Figure 5 Bar chart comparing performance of ranking and roulette selection with and without elitism.](#_toc151734505)

[Figure 6 Number of generations to produce an individual with a fitness of zero over each population value for fifth-order polynomial.](#_toc151734506)

[Figure 7 Big O notation of the genetic algorithm over population size for the fifth-order polynomial.](#_toc151734507)

[Figure 8 Number of generations to produce an individual with a fitness of zero over the probability of mutation for the fifth-order polynomial.](#_toc151734508)

[Figure 9 Number of generations to produce an individual with a fitness of zero over percentage of retained individuals for the fifth-order polynomial.](#_toc151734509)

[Figure 10 Number of generations to produce an individual with a fitness of zero over percentage of retained individuals for the fifth-order polynomial where retain is less than 0.7.](#_toc151734510)

[Figure 11 Number of generations to produce an individual with a fitness of zero over probability of random selection for the fifth-order polynomial.](#_toc151734511)

[Figure 12 Number of generations to produce an individual with a fitness of zero over probability of random selection for the fifth-order polynomial where probability of random selection is less than 0.5.](#_toc151734512)

[Figure 13 Comparing how many generations the algorithm takes to produce an individual with a fitness of zero for each fitness calculation method varying the number of x-value used.](#_toc151734513)

[Figure 14 Number of individuals that match the schemata with varying defining length over the number of generations to stabilise within 1% of the population size.](#_toc151734514)

[Figure 15 Fitness of each schema with varying defining length over the number of generations to stabilise within 1% of the population size.](#_toc151734515)

[Figure 16 Number of individuals that match the schemata with varying order over the number of generations to stabilise within 1% of the population size.](#_toc151734516)

[Figure 17 Fitness of each schema with varying order over the number of generations to stabilise within 1% of the population size.](#_toc151734517)

[Figure 18 Number of individuals that match the schema 0***1001 over the expected number of schema matches.](#_toc151734518)

## <a name="_toc151734471"></a>**TABLE OF TABLES**

[Table 1 List of schemata that match the binary representation of 25 with constant order of 2 and varying defining length.](#_toc151734519)

[Table 2 List of schemata that match the binary representation of 25 with varying order and constant defining length of 7.](#_toc151734520)

## <a name="_toc151734472"></a>**INTRODUCTION**

A genetic algorithm is an optimization algorithm, inspired by the process of natural selection and evolution, which searches for optimal or near-optimal solutions to a problem [1]. The algorithm starts off with a collection of individuals (termed the population) where each individual contains a collection of binary represented numbers (termed chromosomes)  with each bit representing a gene [2]. These genes determine how well the individual performs at solving a problem. The genetic algorithm ranks the individuals based on their fitness to the problem where a fitness of zero means the individuals gene are the optimum solution to the problem. Once the fitness of each individual is calculated, the genetic algorithm evolves the population which consists of retaining a number of individuals, mutating the retained individuals, then making the kept individuals reproduce by means of gene crossover, to produce a collection of children to maintain the population size [3,4]. Evolution attempts to produce a new population better suited to problem to solve. Evolution repeats until an individual with a fitness of zero is found or until one of the termination criteria, discussed in [Exercise 3: Algorithm Termination](#_ref151567022), is met.

## <a name="_toc151734473"></a>**DESIGN**

A genetic algorithm class was designed with a series of different methods to accommodate varying genetic algorithms permutations. The user needs to construct an instance of the genetic algorithm class. The constructor automatically creates an initial population for the user based on the given parameters.

The user then needs to run the *evolve()* method to evolve the population once. If the user wishes to evolve the population until a termination condition is met, they need to wrap the method in a loop. The user can use the *calculateMinFitness()* or *calculateAverageFitness()* to return the minimum fitness and average fitness of the population.

The class also comes fit with methods relating the schemas and Holland’s Schema Theorem. These are separate from the *evolve()* method and need to be called by the user if they wish to gain an understanding of schemas. Any schema used needs to have a maximum length of eight and instead of stars, ‘\*’, bullets need to be used instead, ‘.’, a regular expression was used to implement the schema matching methods.

The user can vary the parameters of the class constructor and the evolve method to use:

- different mutation methods,
- selection methods,
- crossover methods,
- fitness functions,
- probability of mutations,
- probability of random select,
- percentage retained,
- population size,
- individual size,
- minimum chromosome value,
- and maximum chromosome value.

## <a name="_toc151734474"></a>**RESULTS**

### <a name="_toc151734475"></a>**Exercise 2: Varying Parameters**

A genetic algorithm has the following parameters that can be varied:

- **Population size -** the number of individuals in the population.   N ϵ {1≤N<∞} where N is the probability of mutation.
- **Probability of mutation -** the probability that an individual has one of its genes mutated.<a name="_hlk151565301"></a>  Pm ϵ {0≤Pm≤1} where Pm is the probability of mutation.
- **Percentage retained -** percentage of the population retained for the next generations. This prioritises individuals with the best fitnesses (fitnesses close to zero).   R ϵ {0≤R≤1} where R is the percentage of individuals retained.
- **Probability of random selection -** the probability that an individual not selected to be retained is kept for the next generation.   Prs ϵ {0≤Prs≤1} where Prs is the probability of random selection.

To understand how these parameters impacts how many generations the genetic algorithm takes to produce an individual with a fitness of zero, each parameter was varied within their given ranges. Each individual contained six chromosomes ranging from {0≤c≤100}. An individual’s fitness was calculated by finding the absolute difference between the target value of 550 and the sum of the chromosome’s values. The genetic algorithm evolved the population until an individual with a fitness of zero was produced or until the number of evolutions exceeded the maximum number of generations; 10,000.

The results were plotted to graphs detailing how many generations the algorithm took to produce an individual with a fitness of zero. Tests were repeated ten times to accommodate for the random values used to initialise the population.

#### <a name="_toc151734476"></a>**Varying Population Size**

![Number of generations to produce an individual with a fitness of zero over each population value.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.001.png)

<a name="_ref151568683"></a><a name="_toc151734501"></a>**Figure 1** Number of generations to produce an individual with a fitness of zero over each population value.

The population of the genetic algorithm was varied from 10 to 10<sup>6</sup> logarithmically. [Figure 1](#_ref151568683) shows that with small populations, the number of generations it takes to produce an individual with a fitness of zero is considerably large (>100) whilst larger populations had a smaller number of generations. This is because the larger populations can accommodate more gene combinations than the smaller populations and therefore require less evolutions to produce an individual with a fitness of zero.

The number of generations the genetic algorithm takes to produce an individual with a fitness of zero flatlines between one and four once the population reaches 10<sup>3</sup> but as the population increases, the number of generations never stabilises to one. This is because the individuals in the population are not unique. Even if population had a size of 10<sup>12</sup> (the total number of gene variations), the individual(s) with a fitness of zero may not be in the population as there could be multiple identical individuals therefore requiring another generation to produce the individual(s) with a fitness of zero.

#### <a name="_toc151734477"></a>**Varying Probability of Mutation**

![Number of generations to produce an individual with a fitness of zero over each probability of mutation.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.002.png)

<a name="_ref151568697"></a><a name="_toc151734502"></a>**Figure 2** Number of generations to produce an individual with a fitness of zero over each probability of mutation.

[Figure 2](#_ref151568697) shows that as the probability of mutation increases, the number of generations to produce an individual with a fitness of zero decreases. As the probability of mutation increase, the number of retained individuals[^1] that are mutated increases. When probability of mutation is zero, the number of generations it takes the genetic algorithm never produces an individual with a fitness of zero[^2]. This is because the genetic algorithm relies of the probability of mutation to generate new genes which were not seen in the populations history which is an issue if the population does not contain an individual with the required gene to produce an individual with a fitness of zero when the probability of mutation is zero as the genetic algorithm will therefore never be able to produce an individual with a fitness of zero.

However, high probability of mutation increases the change that the individuals with a fitness close to zero are mutated before crossover resulting in more evolutions to produce an individual with a fitness of zero.

#### <a name="_toc151734478"></a>**Varying Percentage of Retained Individuals**

![Number of generations to produce an individual with a fitness of zero over percentage of retained individuals.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.003.png)

<a name="_ref151568711"></a><a name="_toc151734503"></a>**Figure 3** Number of generations to produce an individual with a fitness of zero over percentage of retained individuals.

Retain is the percentage of the population that is retained each generation and used a parent. The genetic algorithm always retains the individuals with the smallest fitnesses. [Figure 3](#_ref151568711) shows that when retain is small (<0.2), the number of generations the genetic algorithms take to produce an individual with a fitness of zero is large (>1000). This is because the limited number of parent individuals produce les varied individuals resulting in many children with similar genetics. This reduced diversity of genetics causes the genetic algorithm to rely more on the probability of mutation for new genes and take more generations to reach an individual with a fitness of zero.

When retain is high (>0.6), the number of generations the genetic algorithm takes to reach an individual is…

#### <a name="_toc151734479"></a>**Varying Probability of Random Selection**

![Number of generations to produce an individual with a fitness of zero over the probability of random selection.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.004.png)

<a name="_ref151568729"></a><a name="_toc151734504"></a>**Figure 4** Number of generations to produce an individual with a fitness of zero over the probability of random selection.

Random selection is the probability that each individual will be randomly selected to be retained. Random selection promotes diversity within the population but retaining individuals with large fitnesses as they genes differ largely compared with the individuals to smaller fitnesses. These randomly selected individuals pass down their genes to their children via crossover with the hopes that they have provided a gene that they child uses to obtain a fitness closer to zero compared with the previous generation.

[Figure 4](#_ref151568729) shows that when the probability of random selection is small (<0.2) the number of generations the genetic algorithm takes to reach an individual with a fitness of zero is large (>400). This is because of the limited diversity as the genetic algorithm is limited to similar genes provides by the parents and the mutation therefore showing that larger probabilities of random selection reduce the number of evolutions.

However, [Figure 4](#_ref151568729) shows that when the probability of random selection is large (>0.9), the number of evolutions the genetic algorithm takes to produce an individual with a fitness of zero is also large (>400). This is because most of the population is made up of individuals from the previous generation therefore the genetic algorithm is not prioritising individuals with small fitnesses. When the probability of random selection is 100%, the difference between the current and the previous generation is due solely to the probability of mutation which promotes diversity and not converging to an individual with a fitness of zero. As a result, the genetic algorithm could produce an individual with a fitness of zero but the number of generations to produce that individual could be extremely large as the mutations are random.

#### <a name="_toc151734480"></a>**Varying Selection Methods**

There are three types of methods that can be used to select parent individuals:

- **Rank –** This method ranks the individuals from best to worse fitness. Then retains a given number of individuals with the best fitnesses. How many individuals retained is determined by the retain parameter [5,6,7]. A positive on rank selection is that it also gives individuals with poor fitness values a chance to reproduce and thus improve [8].
- **Roulette Wheel –** This method picks individuals based on probability determined by [Equation 1](#_ref151650542). Individuals with the best fitness have a higher probability of being chosen than individuals with poor fitnesses [5,6,7].

  Probability=fitness of the individualsum of fitnesses for the population

<a name="_ref151650542"></a><a name="_toc151734496"></a>**Equation 1** Probability of selection of an individual for roulette wheel selection.

- **Elitism –** This method works in conjunction with the other two selection methods. Elitism preserves the individuals with the best fitnesses by preventing them from being mutated or destroyed during crossover. Several versions of the individuals with the best fitnesses are copied verbatim to the next generation [5,9].

All three methods were programmed in the genetic algorithm class and tested using a population of 100, with each individual having six chromosomes varying from 0 to 100. Probability of mutation set to 1%, probability of random select set to 5%, and percentages retained set to 20%.

![Bar chart comparing performance of ranking and roulette selection with and without elitism.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.005.png)

<a name="_ref151650375"></a><a name="_toc151734505"></a>**Figure 5** Bar chart comparing performance of ranking and roulette selection with and without elitism.

[Figure 5](#_ref151650375) shows that when the genetic algorithm uses roulette selection, the algorithm uses 1.45% less generations to produce an individual with a fitness of zero without elitism and 0.43% with elitism when compared to ranking selection. Due to how small the differences between ranking and roulette, one can come the conclusion that either selection method can be used for this problem.

However, [Figure 5](#_ref151650375) shows when the genetic algorithm uses ranking selection with elitism, the algorithm uses 9.63% less generations to produce an individual with a fitness of zero compared to when the algorithm uses ranking selection without elitism. This pattern can also be seen when the genetic algorithm uses roulette selection with elitism. The algorithm uses 8.53% less generations to produce an individual with a fitness of zero compared to when the algorithm uses a roulette selection without elitism. Based on these results, one can determine that using elitism is the preferred method when optimising a genetic algorithm for this problem.

### <a name="_ref151567022"></a><a name="_toc151734481"></a>**Exercise 3: Algorithm Termination**

There are multiple ways in which to terminate the execution of the algorithm:

1. <a name="_ref151568770"></a>When the genetic algorithm has evolved up to a set number of generations. i.e., the algorithm will terminate once it has reached a given number of generations. This is to stop excessive computation [1,10,11].
2. When the best fitness value is less than or equal to a given fitness limit. This is used to stop excessive computation when an acceptable fitness value has been reached by an individual [1,12,11].
3. When the change between the current and previous best fitness values has not changed for a given number of generations. This is used to stop excessive runtimes with the assumption that the algorithm has reached a local minima and will not reach the acceptable level of fitness [1,11].
4. When the difference between the current and previous best fitness values has been equal to or less than a given difference value for the previous t number of generations. This is used to stop excessive runtimes with the assumption that the changes between each fitness value is too small and will take too long to reach the acceptable level of fitness [1,11].

Option [1](#_ref151568770) is the only option which terminates the genetic algorithm once a solution has been found whilst the other options only terminate the algorithm if it gets stuck in a local minima or take too long to produce an individual whose fitness is less than or equal to a given fitness limit.

### <a name="_toc151734482"></a>**Exercise 4: Optimising Parameters for a 5<sup>th</sup>-order Polynomial**

The genetic algorithm was tasked with producing an individual whose genes would match up with the coefficients seen in [Equation 2](#_ref151568793).

y=25x5+18x4+31x3-14x2+7x-19

<a name="_ref151568793"></a><a name="_toc151734497"></a>**Equation 2** Fifth-order polynomial used to calculate the fitness of an individual.

200 hundred x-values ranging from -100 to 100 were created and used to calculate the expected y-values (the y-values for each x-value where the coefficients in the equation seen in [Equation 2](#_ref151568793) are from the target array) and the actual y-values (the y-values for each x-value where the coefficients in the equation seen in [Equation 2](#_ref151568793) are from the individuals genes). The fitness of an individual was calculated by finding the absolute difference between the expected y-values and the individuals actual y-values.

The parameters: population size, probability of mutation, percentage of population retained, and probability of random selection were varied in that order to find the parameters of the genetic algorithm that cause the algorithm to produce an individual with a fitness of zero is the smallest number of generations. The starting parameters for population size, probability of mutation, percentage of population retained, and probability of random selection were 100, 0.01, 0.2, and 0.05 respectively. Once an optimum parameter value had been calculated, it was then used as a constant parameter value for the next parameter tests. I.e., when the optimum population size was calculated, that population size was used when calculating the proceeding optimum parameter values.

#### <a name="_toc151734483"></a>**Optimising Population Size**

![Number of generations to produce an individual with a fitness of zero over each population value for fifth-order polynomial.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.006.png)

<a name="_ref151568842"></a><a name="_toc151734506"></a>**Figure 6** Number of generations to produce an individual with a fitness of zero over each population value for fifth-order polynomial.

[Figure 6](#_ref151568842) shows that the optimum population size to produce an individual with a fitness of zero in the least number of generations is 70,000. However, though 70,000 is the optimum population size, the big O notation for this population size is very large (>1e7), shown in [Figure 7](#_ref151568855), causing large runtimes. These large runtimes would cause exceedingly large amount of time to calculate the optimum values for the other parameters. Therefore, a different population size with a smaller big O notation must be chosen.

![Big O notation of the genetic algorithm over population size for the fifth-order polynomial.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.007.png)

<a name="_ref151568855"></a><a name="_toc151734507"></a>**Figure 7** Big O notation of the genetic algorithm over population size for the fifth-order polynomial.

The population with the smallest big O notation, 10, was used. However, [Figure 7](#_ref151568855) also shows that for populations equal to and less than 200 never produced an individual with a fitness of zero because the genetic algorithm exceeded the max number of evolutions.

Therefore, the population value that is larger than 200 and has the smallest big O notation value was chosen. The optimum population value is 10,000.

#### <a name="_toc151734484"></a>**Optimising Probability of Mutation**

![Number of generations to produce an individual with a fitness of zero over the probability of mutation for the fifth-order polynomial.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.008.png)

<a name="_ref151568862"></a><a name="_toc151734508"></a>**Figure 8** Number of generations to produce an individual with a fitness of zero over the probability of mutation for the fifth-order polynomial.

[Figure 8](#_ref151568862) shows the number of generations the genetic algorithm takes to produce an individual with a fitness of zero for each probability of mutation varying from 0.1 to 1.0 in steps of 0.01. [Figure 8](#_ref151568862) show many fluctuations between each value. This is because the probability of mutation is just the probability that a parent individual is mutated not the proportion of parents that are mutated. As a result, the number of mutations that occur at different generations can vary therefore varying the diversity of genes in the next generation leading to a varying number of generations to genetic algorithms takes to produce an individual with a fitness of zero.

Despite this, the least number of generations the algorithm takes to produce an individual with a fitness of zero belongs the probability of mutation of 0.46.

#### <a name="_toc151734485"></a>**Optimising Percentage of Retained Individuals**

![Number of generations to produce an individual with a fitness of zero over percentage of retained individuals for the fifth-order polynomial](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.009.png)

<a name="_ref151568869"></a><a name="_toc151734509"></a>**Figure 9** Number of generations to produce an individual with a fitness of zero over percentage of retained individuals for the fifth-order polynomial.

[Figure 9](#_ref151568869) shows that as the percentage of the population that is retained increases, the number of generations the genetic algorithm takes to produce an individual with a fitness of zero. Once the retain reaches 0.74, the algorithm never produces an individual with a fitness of zero and instead reaches the max number of generations; 10,000.

![Number of generations to produce an individual with a fitness of zero over percentage of retained individuals for the fifth-order polynomial where retain is less than 0.7.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.010.png)

<a name="_ref151568875"></a><a name="_toc151734510"></a>**Figure 10** Number of generations to produce an individual with a fitness of zero over percentage of retained individuals for the fifth-order polynomial where retain is less than 0.7.

[Figure 10](#_ref151568875) shows the same results from [Figure 9](#_ref151568869) but only the results where retain in less than 0.7. [Figure 9](#_ref151568869) shows that the smallest number of evolutions the algorithm took to produce an individual with a fitness of zero is 21, which corresponds to the retain value of 0.15.

#### <a name="_toc151734486"></a>**Optimising Probability of Random Selection**

![Number of generations to produce an individual with a fitness of zero over probability of random selection for the fifth-order polynomial.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.011.png)

<a name="_ref151650837"></a><a name="_toc151734511"></a>**Figure 11** Number of generations to produce an individual with a fitness of zero over probability of random selection for the fifth-order polynomial.

[Figure 11](#_ref151650837) shows how many generations the genetic algorithm took to produce an individual with a fitness of zero for each probability of random selection. The maximum probability of random selection tested was 0.65 as the computation time exceed an hour for each test once the probability of random selection was larger than 0.6.

![Number of generations to produce an individual with a fitness of zero over probability of random selection for the fifth-order polynomial where probability of random selection is less than 0.5.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.012.png)

<a name="_ref151650848"></a><a name="_toc151734512"></a>**Figure 12** Number of generations to produce an individual with a fitness of zero over probability of random selection for the fifth-order polynomial where probability of random selection is less than 0.5.

[Figure 12](#_ref151650848) shows that the probability of random selection with the smallest number of generations is 0.01.

#### <a name="_toc151734487"></a>**Varying Fitness Function**

There are two methods to calculate the fitness of an individual for this problem:

- **Elementwise –** Sum the absolute differences between an individual’s chromosomes and their corresponding target.

  fx=i=0L|ti-xi|

  <a name="_toc151734498"></a>**Equation 3 Sum of the absolute elementwise difference between two vectors.**

  Where x is an individual, xi is a chromosome in the individual, L is the number of chromosomes in the individual, and t is the target vector.

- **Polynomial-wise –** Create a range of x-values from -100 to 100. For each x-value, calculate its corresponding y-value using the target vector and again for the individual’s chromosomes, named the target y-value and the actual y-value respectively. Sum the absolute difference between each x-values corresponding target y-value and actual y-value.

  ga, b=i=0caibc-i

  fx=jϵmgt, j-gx,j

  <a name="_toc151734499"></a>**Equation 4** Sum of the absolute difference between a series of target y-values and actual y-values.

  Where g(a,b) is the generic polynomial equation. c is the order of the polynomial. t is the target vector and x is the individual. m is a list of values, of size n, which varies from -100 to 100.

Both methods for calculating fitness were compared varying n, the number of x-values used to calculate fitness using the polynomial method[^3].

![Comparing how many generations the algorithm takes to produce an individual with a fitness of zero for each fitness calculation method varying the number of x-value used.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.013.png)

<a name="_ref151734207"></a><a name="_toc151734513"></a>**Figure 13** Comparing how many generations the algorithm takes to produce an individual with a fitness of zero for each fitness calculation method varying the number of x-value used.

[Figure 13](#_ref151734207) shows that small number of x-values (<30) causes an increase in the number of generations the genetic algorithms need to produce an individual with a fitness of 0. However, once the number of x-values exceeds 30, the number of generations fluctuates which is due to the random initialization of the population.

To conclude this investigation, the optimum parameters for the genetic algorithm to produce an individual with fitness of zero for the fifth order polynomial, seen Equation 2, in 20 generations is:

- Population size: 10,000.
- Probability of mutation: 46%.
- Percentage retained: 15%.
- Probability of random select: 1%.

Each optimum parameter was calculated then used to find the next optimum parameter. Though these are the most optimum parameters, the method used was only optimum for computation time and not achieving the true optimum parameters. Assuming infinite computation power, a more optimal method would be to vary each parameter for each other parameter resulting in 50,000,000 different combinations when varying probability of mutation, probability of random selection, and percentage of retain from zero to one in steps of 0.01 and varying the population size logarithmically from 10 to 1,000,000.

In addition, the method to calculate the fitness of an individual proved to have negligible affect to the number of generations the algorithm takes to produce an individual with a fitness of zero as long as the number of x-values used to the polynomial method is larger than 30.

### <a name="_toc151734488"></a>**Exercise 5: Proving Hollands Schema Theorem**

Schemas are a bit/gene pattern thar can exist within an individual. The schemas can contain either 0, 1, or \* where starts represents bits/genes that could be either 0 or 1. If an individual has the specified bits, then it matches the schema (written as x ϵ H where x is the individual and H is the schema).

Hollands Schema Theorem states the expected number of individuals that matches the schema, H, for the next generation is equal to or larger than the current number of individuals that matches the schema if the average fitness of all individuals that meet the schema is equal to or larger than the average fitness of the population [13,14,15]. In essence, the frequency of schemata with above average fitness will increase exponentially.

<a name="_toc151734500"></a>**Equation 5** Hollands Schema Theorem.

To prove Hollands Schema Theorem, a list of schemata was created all based off the 8-bit binary representation of the coefficient 25 from [Equation 2](#_ref151568793) but each with varying order[^4] and varying defining length[^5].

#### <a name="_toc151734489"></a>**Varying Defining Length**

A list of schemata was created based off the 8-bit binary representation of the coefficient 25 from [Equation 2](#_ref151568793). [Table 1](#_ref151568919) shows the schemata all with an order of two.

|Defining Length|Schema|
| :- | :- |
|1|00\*\*\*\*\*\*|
|2|0\*0\*\*\*\*\*|
|3|0\*\*1\*\*\*\*|
|4|0\*\*\*1\*\*\*|
|5|0\*\*\*\*0\*\*|
|6|0\*\*\*\*\*0\*|
|7|0\*\*\*\*\*\*1|

<a name="_ref151568919"></a><a name="_toc151734519"></a>**Table 1** List of schemata that match the binary representation of 25 with constant order of 2 and varying defining length.

Each schema was tested for the optimisation of the 5<sup>th</sup> – order polynomial from exercise 3. The algorithm stopped running once the max number of generations was reached, when an individual with a fitness of 0 was produced, or when the difference between the current and previous generations individuals that matched the schema was less than 1% of the total population.

![Number of individuals that match the schemata with varying defining length over the number of generations to stabilise within 1% of the population size.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.014.png)

<a name="_ref151568936"></a><a name="_toc151734514"></a>**Figure 14** Number of individuals that match the schemata with varying defining length over the number of generations to stabilise within 1% of the population size.

[Figure 14](#_ref151568936) shows how the number of instances belonging to each schema varies as the number of generations increases. This is to be expected as all the schemas tested can be found within the target individual.

![Fitness of each schema with varying defining length over the number of generations to stabilise within 1% of the population size.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.015.png)

<a name="_ref151568942"></a><a name="_toc151734515"></a>**Figure 15** Fitness of each schema with varying defining length over the number of generations to stabilise within 1% of the population size.

[Figure 15](#_ref151568942) shows how the average fitness of each schema decreased as the number of generations increased. It also shows the average fitness of the total population. All schema fitnesses are smaller than the population fitness at each generation which satisfies the condition for Holland’s Schema Theorem[^6].

#### <a name="_toc151734490"></a>**Varying Order**

A list of schemata was created based off the 8-bit binary representation of the coefficient 25 from [Equation 2](#_ref151568793). [Table 2](#_ref151568967) shows the schemata all with a defining length of seven.

|Order|Schema|
| :- | :- |
|2|0\*\*\*\*\*\*1|
|3|0\*\*\*\*\*01|
|4|0\*\*\*\*001|
|5|0\*\*\*1001|
|6|0\*\*11001|
|7|0\*011001|
|8|00011001|

<a name="_ref151568967"></a><a name="_toc151734520"></a>**Table 2** List of schemata that match the binary representation of 25 with varying order and constant defining length of 7.

Each schema was tested for the optimisation of the 5th – order polynomial from exercise 3. The algorithm stopped running once the max number of generations was reached, when an individual with a fitness of 0 was produced, or when the difference between the current and previous generations individuals that matched the schema was less than 1% of the total population.

![Number of individuals that match the schemata with varying order over the number of generations to stabilise within 1% of the population size.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.016.png)

<a name="_ref151568989"></a><a name="_toc151734516"></a>**Figure 16** Number of individuals that match the schemata with varying order over the number of generations to stabilise within 1% of the population size.

[Figure 16](#_ref151568989) shows that number of individuals that match the schema increases as the number of generations increases. This is to be expected as all schemata can be found within the target individual.

Schema 0\*\*\*\*\*\*1 initially increases then decreases before it continues to increase. It also has many individuals that match the schema at generation 0. Both observations can be explained because of the large number of non-fixed bits/genes within the schema causing the probability that a larger proportion of the initial population would match the schema to be larger than other schemas with  larger orders. This observation can also be seen in [Figure 14](#_ref151568936).

![Fitness of each schema with varying order over the number of generations to stabilise within 1% of the population size.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.017.png)

<a name="_ref151569014"></a><a name="_toc151734517"></a>**Figure 17** Fitness of each schema with varying order over the number of generations to stabilise within 1% of the population size

[Figure 17](#_ref151569014) shows how the fitnesses of the schemata decreases as the number of evolutions increases. It can also be seen that all fitnesses of the schemata are less than the average fitness of the population at each generation therefore satisfying Hollands Schema Theorem condition.

It should also be notes that fitness of schema 00011001 appears to be relatively small for each generation. This is because the schema matches exactly to the 8-bit binary representation of the coefficient 25 from [Equation 2](#_ref151568793). In addition, the genetic algorithm will return zero when calculating the average schema fitness if there are no individuals that match the schema.

#### <a name="_toc151734491"></a>**Proving Hollands Schema Theorem**

All the schemata observed meet the requirement that average fitness of all the individuals that match the schema should be equal to or greater than the average fitness of the population.

To prove Hollands Schema Theorem, the average fitness of each schema at each generation was compared to the expected average fitness of that schema.

![Number of individuals that match the schema 0\*\*\*1001 over the expected number of schema matches.](./img/readme/Aspose.Words.636f98bd-dd88-43f3-b0b4-161b88929dd8.018.png)

<a name="_ref151569044"></a><a name="_toc151734518"></a>**Figure 18** Number of individuals that match the schema 0\*\*\*1001 over the expected number of schema matches.

[Figure 18](#_ref151569044) shows the comparison for the schema 0\*\*\*1001. [Figure 18](#_ref151569044) shows that as the number of generations increases, the actual number of individuals that match the schema is equal to or larger than the expected number of individuals that match the schema. [Figure 18](#_ref151569044) also shows that the expected number of individuals that match the schema grows exponentially as the schema’s fitness is better than the average fitness of the population, proven by [Figure 17](#_ref151569014).

However, once the genetic algorithm has evolved for more than ten generations, at which point the actual value has stabilized around the population size of 10,000, the expected value experiences a wide range of fluctuations. This because the theorem holds under the assumption that the genetic algorithm has an infinitely large population size. This assumption does not always carry over when dealing with populations of finite sizes due to sampling error therefore resulting the expected number of individuals exceeding the population size.

## <a name="_toc151734492"></a>**CONCLUSION**

This report has explored how a genetic algorithm works and how the population size, probability of mutation, probability of random selection, percentage of retain, fitness functions, and selection algorithms impact the performance of the algorithm. The report has also examined Holland’s Schema Theorem which provides insight into the behaviour of genetic algorithms over future generations.

The analysis of the genetic algorithms has shown that it can be an effective tool for solving complex optimisation problems such as optimising neural network. Genetic algorithms can produce individuals with high quality solutions for various problems including search and optimisation by simulating the process of natural selection. However, the performance of the genetic algorithm is dependent on the parameters, fitness function, and selection algorithm used. Therefore, it is important to accurately tune the parameters and wisely pick the fitness function and selection algorithm to use to achieve optimal results efficiently.

## <a name="_toc151734493"></a>**REFERENCES**

|[1] |Wikipedia, The Free Encyclopedia, “Genetic Algorithm,” 20 November 2023. [Online]. Available: https://en.wikipedia.org/w/index.php?title=Genetic_algorithm&oldid=1186064335. [Accessed 22 November 2023].|
| :- | :- |
|[2] |D. Whitley, “A genetic Algorithm Tutorial,” in *Statistics and Computing*, 2022, pp. 65-85.|
|[3] |A. E. Eiben, “Genetic algorithms with multi-parent recombinations,” in *International Conference on Evolutionary Computation*, 1994. |
|[4] |C.-K. Ting, “On the Mean Convergence Time of Multi-parent Genetic Algorithms Without Selection,” in *Advances in Artificial Life*, 2005, pp. 403-412.|
|[5] |Wikipedia Contributors, “Selection (genetic algorithm),” 6 October 2023. [Online]. Available: https://en.wikipedia.org/w/index.php?title=Special:CiteThisPage&page=Selection_%28genetic_algorithm%29&id=1178932579&wpFormIdentifier=titleform. [Accessed 23 November 2023].|
|[6] |J. E. Baker and J. J. Grefenstette, “Adaptive Selection Methods for Genetic Algorithms,” Hillsdale, 1985, pp. 101-111.|
|[7] |J. E. Baker and J. J. Grefenstette, “Reducing Bas and Inefficiency in the Selection Algorithm,” in *Conf. Proc. of the 2nd Int. Conf. on Genetic Algorithms and Their Applications (ICGA)*, Hillsdale, 1987. |
|[8] |D. Whitley and J. Schaffer, “The GENITOR Algorithm and Selection Pressure: Why Rank-Based Allocation of Reproductive Trials is Best,” in *Proceedings of the Third International Conference on Genetic Algorithms (ICGA)*, San Francisco, 1989. |
|[9] |V. Fulber-Garcia, “Elitism in Evolutionary Algorithms,” 13 June 2023. [Online]. Available: https://www.baeldung.com/cs/elitism-in-evolutionary-algorithms. [Accessed 23 November 2023].|
|[10] |Computer Science Wiki, “Termination Condition,” Computer Science Wiki, 3 December 2021. [Online]. Available: https://computersciencewiki.org/index.php/Termination_condition. [Accessed 22 November 2023].|
|[11] |MathWorks, “Genetic Algorithm Options,” MathWorks, [Online]. Available: https://uk.mathworks.com/help/gads/genetic-algorithm-options.html. [Accessed 22 November 2023].|
|[12] |D. S. K., “A Gentle Intoduction To Genetic Algorithms,” Towards AI, 19 July 2023. [Online]. Available: https://towardsai.net/p/machine-learning/a-gentle-introduction-to-genetic-algorithms. [Accessed 22 November 2023].|
|[13] |C. Bridges and D. E. Goldberg, “An analysis of repoduction and crossover in a binary-coded genetic algorithm,” in *2nd Int'l Conf. on Genetic Algorithms and their applications*, 1987. |
|[14] |HandWiki, “Holland's Schema Theorem,” HandWiki, 27 June 2023. [Online]. Available: https://handwiki.org/wiki/index.php?title=Holland%27s_schema_theorem&action=info. [Accessed 22 November 2023].|
|[15] |Traditional and Non-Traditional Optimization Tools, *Lecture 2 Schema Theorem of BCGA,* YouTube, 2018. |

[^1]: Individuals that are kept for the next generation.

[^2]: The test where the probability of mutation is zero is displayed on [Figure 2](#_ref151568697) to ensure all other tests results are visible.

[^3]: The elementwise method will not be varied for each number of x-values used as the elementwise method does not rely on x-values to calculate the fitness.

[^4]: The number of fixed bits/genes in the schema.

[^5]: The length between the first and last fixed bit/gene in the schema.

[^6]: Hollands Schema Theorem condition is for schemata to have above average fitness. This genetic algorithm is coded so that individuals with smaller fitness values are better. Therefore, though the average fitness of the schemata is smaller than the average fitness of the population, it still satisfies the condition.