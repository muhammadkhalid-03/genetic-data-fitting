"""
hw3.py
Name(s): Muhammad Khalid
Date: 13th March 2024



"""

import math
import random
import Organism as Org
import numpy as np
import matplotlib.pyplot as plt

"""
crossover operation for genetic algorithm

Inputs:
parent1 - organism from old population used to mate
parent2 - organism from old population used to mate

Outputs:
child1 - new organism with bits from both parents using random k
child2 - new organism with bits from both parents using random k
"""
def crossover(parent1, parent2):

    if len(parent1.bits) != len(parent2.bits):  #check if length is the same
        raise ValueError("Parent genomes must be of the same length for crossover.")
    k = random.randint(0, len(parent1.bits)) #take random k
    child1Bits = np.concatenate((parent1.bits[:k], parent2.bits[k:]))
    child2Bits = np.concatenate((parent2.bits[:k], parent1.bits[k:]))
    return (child1Bits, child2Bits)

"""
mutation operation for genetic algorithm

Inputs:
genome -  A single organism
mutRate (int) - probability of each bit being flipped

Outputs:
genome - An organism with flipped bits according to probability
"""
def mutation(genome, mutRate):

    for i in range(len(genome.bits)):
        if mutRate > random.random():   #prob to flip
            genome.bits[i] = 1 - genome.bits[i]
    return genome

"""
selection operation for choosing a parent for mating from the population

Inputs:
pop - a population of organisms

Outputs:
org - a single organism whose fitness > threshold (0, 1)
"""
def selection(pop):
    threshold = random.random()
    for org in pop:
        if org.accFit > threshold:
            return org
    return pop[-1]

"""
calcFit will calculate the fitness of an organism
"""
def calcFit(org, xVals, yVals):
    # Create a variable to store the running sum error.
    error = 0

    # Loop over each x value.
    for ind in range(len(xVals)):
        # Create a variable to store the running sum of the y value.
        y = 0
        
        # Compute the corresponding y value of the fit by looping
        # over the coefficients of the polynomial.
        for n in range(len(org.floats)):
            # Add the term c_n*x^n, where c_n are the coefficients.
            try:
                y += org.floats[n] * (xVals[ind])**n
            except OverflowError:
                y += math.inf

        # Compute the squared error of the y values, and add to the running
        # sum of the error.
        try:
            error += (y - yVals[ind])**2
        except OverflowError:
            error += math.inf

    # Now compute the sqrt(error), average it over the data points,
    # and return the reciprocal as the fitness.
    if error == 0:
        return math.inf
    else:
        fitness = len(xVals)/math.sqrt(error)
        if not math.isnan(fitness):
            return fitness
        else:
            return 0



"""
accPop will calculate the fitness and accFit of the population

Inputs:
pop - population of organisms

Outputs:
pop - population of organisms sorted in descending order based on fitness
"""
def accPop(pop, xVals, yVals):

    totalFitness = 0.0
    #loop over each organism in population
    for org in pop:

        #run calcFit() on each organism w/ xVals & yVals -> fitness of each organism
        org.fitness = calcFit(org, xVals, yVals)

        #running sum of total fitness
        totalFitness += org.fitness
        

    #sort fitness in reverse order
    pop.sort(key=lambda x: x.fitness, reverse=True)
    
    accFitness = 0.0
    
    #loop over each organism in population
    for org in pop:

        #divide individual fitness of each organism by total fitness -> normalized fitness
        org.normFit = org.fitness/float(totalFitness)
        accFitness += org.normFit   #running total of accumulated normalized fitness
        org.accFit = accFitness #accumulated normalized fitness for each organism
    return pop

"""
initPop will initialize a population of a given size and number of coefficients
"""
def initPop(size, numCoeffs):
    # Get size-4 random organisms in a list.
    pop = [Org.Organism(numCoeffs) for x in range(size-4)]

    # Create the all 0s and all 1s organisms and append them to the pop.
    pop.append(Org.Organism(numCoeffs, [0]*(64*numCoeffs)))
    pop.append(Org.Organism(numCoeffs, [1]*(64*numCoeffs)))

    # Create an organism corresponding to having every coefficient as 1.
    bit1 = [0]*2 + [1]*10 + [0]*52
    org = []
    for c in range(numCoeffs):
        org = org + bit1
    pop.append(Org.Organism(numCoeffs, org))

    # Create an organism corresponding to having every coefficient as -1.
    bit1 = [1,0] + [1]*10 + [0]*52
    org = []
    for c in range(numCoeffs):
        org = org + bit1
    pop.append(Org.Organism(numCoeffs, org))

    # Return the population.
    return pop

"""
nextGeneration will create the next generation

Inputs:
pop - list of organisms sorted in desc order based on fitness
numCoeffs (int) - Number of coefficients in the polynomial
mutRate (float) - Mutation rate
eliteNum (int) - Number of elite individuals
"""
def nextGeneration(pop, numCoeffs, mutRate, eliteNum):

    #create empty list newPop
    newPop = []

    #get number of crossovers to be performed
    fillPop = (len(pop)-eliteNum)//2

    #for each crossover
    for i in range(fillPop):

        #select two parents from population using selection function
        parent1 = selection(pop)
        parent2 = selection(pop)

        #use parents to get two new children using crossover function
        (child1Bits, child2Bits) = crossover(parent1, parent2)
        child1 = Org.Organism(numCoeffs, child1Bits)
        child2 = Org.Organism(numCoeffs, child2Bits)

        #mutate both childs' genomes using mutation function and add to newPop
        newPop.append(mutation(child1, mutRate))
        newPop.append(mutation(child2, mutRate))

    #append eliteNum children to newPop
    for i in range(eliteNum):
        newPop.append(pop[i])

    return newPop

"""
GA will perform the genetic algorithm for k+1 generations (counting
the initial generation).

INPUTS
k:         the number of generations
size:      the size of the population
numCoeffs: the number of coefficients in our polynomials
mutRate:   the mutation rate
xVals:     the x values for the fitting
yVals:     the y values for the fitting
eliteNum:  the number of elite individuals to keep per generation
bestN:     the number of best individuals to track over time

OUTPUTS
best: the bestN number of best organisms seen over the course of the GA
fit:  the highest observed fitness value for each iteration
"""
def GA(k, size, numCoeffs, mutRate, xVals, yVals, eliteNum, bestN):
    #list of orgs for highest fit in each gen
    #initialize population using size & numCoeffs (initial population)
    #run accFit -> sorted list of orgs w/ fitness decreasing
    #store the highest fitness in curr generation
    #for every generation in k-1 generations
        #nextGeneration(current population) -> new population
        #run accFit -> sorted list of orgs w/ fitness decreasing
        #do highest fitness stuff
        #add this to initial population
    #sort list of generations
    #best is every element till bestN

    fit = []
    best = []

    oldPop = initPop(size, numCoeffs)
    oldSortedPop = accPop(oldPop, xVals, yVals)   #sorted population
    fit.append(oldSortedPop[0].fitness)
    best += oldSortedPop

    #for each generation
    for i in range(k):
        newPop = nextGeneration(oldPop, numCoeffs, mutRate, eliteNum)  #create new population
        newSortedPop = accPop(newPop, xVals, yVals) #sort new population
        fit.append(newSortedPop[0].fitness) #add fittest org to list
        compareOrg = best[-1]
        for org in newSortedPop:    #for each org in new sorted population

            #if the organism isn't in the list of best organisms AND is better than the lowest organism in best
            if (org not in best and org.__gt__(compareOrg)):
                best.append(org)    #add org to best list
        oldPop = newPop
    #sort fitness in reverse order
    best.sort(key=lambda x: x.fitness, reverse=True)

    best = best[:bestN] #till bestN

    

    return (best,fit)

"""
runScenario will run a given scenario, plot the highest fitness value for each
generation, and return a list of the bestN number of top individuals observed.

INPUTS
scenario: a string to use for naming output files.
--- the remaining inputs are those for the call to GA ---

OUTPUTS
best: the bestN number of best organisms seen over the course of the GA
--- Plots are saved as: 'fit' + scenario + '.png' ---
"""
def runScenario(scenario, k, size, numCoeffs, mutRate, \
                xVals, yVals, eliteNum, bestN):

    # Perform the GA.
    (best,fit) = GA(k, size, numCoeffs, mutRate, xVals, yVals, eliteNum, bestN)

    # Plot the fitness per generation.
    gens = range(k+1)
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(gens, fit)
    plt.title('Best Fitness per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.savefig('fit'+scenario+'.png', bbox_inches='tight')
    plt.close('all')

    # Return the best organisms.
    return best

"""
main function
"""
if __name__ == '__main__':

    # Flags to suppress any given scenario. Simply set to False and that
    # scenario will be skipped. Set to True to enable a scenario.
    scenA = False
    scenB = False
    scenC = False
    scenD = True

    if not (scenA or scenB or scenC or scenD):
        print("All scenarios disabled. Set a flag to True to run a scenario.")
    
################################################################################
    ### Scenario A: Fitting to a constant function, y = 1. ###
################################################################################

    if scenA:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = 1 corresponding to the x values.
        yVals = [1. for n in range(len(xVals))]

        # Set the other parameters for the GA.
        sc = 'A'      # Set the scenario title.
        k = 100       # 100 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()

################################################################################
    ### Scenario B: Fitting to a constant function, y = 5. ###
################################################################################
    
    if scenB:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = 1 corresponding to the x values.
        yVals = [5. for n in range(len(xVals))]

        # Set the other parameters for the GA.
        sc = 'B'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()

################################################################################
    ### Scenario C: Fitting to a quadratic function, y = x^2 - 1. ###
################################################################################
    
    if scenC:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = x^2 - 1 corresponding to the x values.
        yVals = [x**2-1. for x in xVals]

        # Set the other parameters for the GA.
        sc = 'C'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()

################################################################################
    ### Scenario D: Fitting to a quadratic function, y = cos(x). ###
################################################################################
    
    if scenD:
        # Create the x values ranging from -5 to 5 with a step of 0.1.
        xVals = [0.1*n-5 for n in range(101)]

        # Create the y values for y = cos(x) corresponding to the x values.
        yVals = [math.cos(x) for x in xVals]

        # Set the other parameters for the GA.
        sc = 'D'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 5 # Quartic polynomial with 4 zeros!
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()
