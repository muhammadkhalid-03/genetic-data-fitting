# Polynomial Fitting using Genetic Algorithm

## Project Overview

This project implements a genetic algorithm to fit polynomials to given data sets. The algorithm determines the coefficients of an nth-order polynomial that minimizes the average squared error between the polynomial and the provided data points.

## Files

- `hw3.py`: Main file containing the genetic algorithm implementation
- `Organism.py`: Provided class file for the Organism objects used in the genetic algorithm
- `fitA.png`, `fitB.png`, `fitC.png`, `fitD.png`: Generated plots for 4 scenarios
- `hw3.pdf`: PDF file provided by the professor

### Main Functions

1. `calcFit`: Calculates the fitness of an organism
2. `initPop`: Initializes the population
3. `crossover`: Performs single-point crossover
4. `mutation`: Applies mutation to an organism's genome
5. `selection`: Selects an organism for mating
6. `accPop`: Calculates fitness values and sorts the population
7. `nextGeneration`: Creates the next generation of organisms
8. `GA`: Main genetic algorithm function

### Scenarios

The project includes four test scenarios:
A. y = 1 (fitA.png)
B. y = 5 (fitB.png)
C. y = x² - 1 (fitC.png)
D. y = cos(x) (fitD.png)

## Usage

1. Ensure both `hw3.py` and `Organism.py` are in the same directory
2. Run `hw3.py`
3. Activate/deactivate scenarios by setting `scenA`, `scenB`, `scenC`, and `scenD` to True/False in the main block

## Parameters

- Population size: 1000
- Elite individuals: 5% (50)
- Mutation rate: 10%
- Number of generations: Several hundred (varies by scenario)

## Output

- Plots of highest fitness score vs. generation number
- List of best individuals found during the algorithm

## Notes

- The first three scenarios use cubic polynomials for fitting
- Scenario D uses a quartic polynomial to fit cos(x) on the domain [-5, 5]
