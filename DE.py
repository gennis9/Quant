"""
The following program is used to implement differential evolution (DE), which is an evolutionary computation variant designed primarily for multidimensional real-valued spaces.
It introduces two new twists:
1. Children must compete directly against their immediate parents for inclusion in the population;
2. DE determines the size of mutates largely based on the current variance in the population.
"""
# Copyright (C) 2022 Chun Hung, Tsang


### Enviornment
"                               ..... Enviornment .....                             "
import numpy as np
import time


## Set default parameters.
# Mutation rate.
alpha = 0.75
# Desired population size.
popsize = 20
# Probability of parameterized uniform crossover.
# See Ackley (1987) for the assumed probability to be set as 50%.
prob = 0.5

# According to Bozorg-Haddad, Solgi, and Loáiciga (2017), we establish the stopping criteria as:
maxIterNum = 10000
improve = 0.0001
maxIterTime = 20 * 60


## Define objective function.
def f(x):
    return 20 + ( x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) ) + ( x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) )
# Specify the dimension of real space.
dim = 2
# Set the domain of real space.
domain = [-5.12, 5.12]
# Specify the type of optimizer: 'min' or 'max'.
opt = 'min'




### Initialization
"                               ..... Initialization .....                             "
## Generate parents.
Parent = [ np.array( [np.random.uniform(domain[0], domain[-1]) for _i in range(dim)] ) for _j in range(popsize) ]
Best = np.array( [np.random.uniform(domain[0], domain[-1]) for _i in range(dim)] )

## Record the start time.
Start = time.time()




### Optimization
"                               ..... Optimization .....                             "
## Start the DE.
# For each round of iteration.
for rd in range(maxIterNum):
    # For each parent of population.
    for _i in range(popsize):
        ## Mutation.
        Mutate = Parent.copy()
        Mutate.pop( _i )
        np.random.shuffle(Mutate)
        Mutate = Mutate[:3]
        Child = Mutate[0] + alpha * (Mutate[1] - Mutate[2])
        ## In order to make sure the mutated child stays within domain, the boundary of real space is set as joined to itself.
        # The lower bound.
        Child = np.where( Child < domain[0], Child + (domain[1]-domain[0]), Child )
        # The upper bound.
        Child = np.where( Child > domain[1], Child - (domain[1]-domain[0]), Child )
                
        ## According to Lukes (2013), apply the crossover algorithm.
        # i.e., Use uniform crossover with at least 1 gene from parent.
        _gene = np.random.randint(dim)
        # For each gene of parent.
        for _j in range(dim):
            # Inheritance.
            if _j == _gene:
                Child[_j] = Parent[_i][_j]
            else:
                if np.random.random() <= prob:
                    Child[_j] = Parent[_i][_j]
            
        ## Compare parent and child and find out the best solution.
        if opt == 'min':
            ## Replace parent.
            if f(Child) < f(Parent[_i]):
                Parent[_i] = Child
            if f(Parent[_i]) < f(Best):
                Best = Parent[_i]
        # Maximizer.
        else:
            ## Replace parent.
            if f(Child) > f(Parent[_i]):
                Parent[_i] = Child
            if f(Parent[_i]) > f(Best):
                Best = Parent[_i]

    ## Count the time.
    Time = time.time() - Start
    ## The execution time is over.
    if Time >= maxIterTime:
        Stop = 'Time is up.'
        break

    ## The improvement of objective function satisfies the stopping criteria.
    # The first round is excluded.
    if rd != 0:
        # To avoid divide-zero error.
        if f(LastBest) == 0 and np.absolute( f(Best) ) < improve:
            Stop = 'Solution is found.'
            break
        elif np.absolute( (f(Best) - f(LastBest))/f(LastBest) ) < improve:
            Stop = 'Solution is found.'
            break

    ## Record the current best optimizer before running the next iteration.
    LastBest = Best

    ## Show the current progress regularly.
    if rd%50 == 49:
        print('Iteration: ', rd + 1, ' rounds; Optimizer: ', Best,
              '; Fitness value: ', f(Best), '; Time: ', int(Time//60), 'min ', str(Time%60)[:6], 's.')

    ## The iteration rounds achieves maximum.
    if rd == maxIterNum - 1:
        Stop = 'Iteration ran out.'




### Result Visualization
"                               ..... Result Visualization .....                             "
## Report the final result.
print('========================================================')
print('  Final Result Table')
print('--------------------------------------------------------')
print('  Optimizer: x1 = ', Best[0], ',')
print('             x2 = ', Best[1])
print('  The function value: ', f(Best))
print('--------------------------------------------------------')
print('  Description: ', Stop)
print('  Total iteration rounds:', rd + 1)
print('  The time used: ', int(Time//60), 'min ', str(Time%60)[:6], 's.')
print('========================================================')


