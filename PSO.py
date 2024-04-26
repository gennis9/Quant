"""
The following program is used to implement particle swarm optimization (PSO), which is a stochastic optimization technique based on the movement and intelligence of swarms.
There is no selection among PSO practiioners (particles), but rather the directed mutation moves the particles about in the space.
Every particle consists of 2 parts:
1. The particle’s location in space, x = <x1, x2, ...>;
2. The particle’s velocity, v = <v1, v2, ...>; i.e., v = x_{t} - x_{t-1}.
"""
# Copyright (C) 2022 Chun Hung, Tsang


### Enviornment
"                               ..... Enviornment .....                             "
import numpy as np
import time


## Set default parameters.
swarmSize = 100
# Proportion of velocity to be retained.
alpha = 0.4
# Proportion of personal best to be retained.
beta = 0.4
# Proportion of the informants' best to be retained.
gamma = 0.2
# Proportion of global best to be retained.
delta = 0.4
# Jump size of a particle.
epsilon = 1

# According to Bozorg-Haddad, Solgi, and Loáiciga (2017), we establish the stopping criteria as:
maxIterNum = 10000
improve = 0.0001
maxIterTime = 20 * 60


## Define objective function.
def f(x):
    return sum([ (1 + x[i])**2 + 100 * (x[i+1] - x[i]**2)**2 for i in range(dim-1) ])
# Specify the dimension of real space.
dim = 3
# Set the domain of real space.
domain = [-2.048, 2.048]
# Specify the type of optimizer: 'min' or 'max'.
opt = 'min'




### Initialization
"                               ..... Initialization .....                             "
## Generate initial positions and velocities of particles.
Particles = [ np.array( [np.random.uniform(domain[0], domain[-1]) for _i in range(dim)] ) for _j in range(swarmSize) ]
Velocity = [ np.zeros(dim) for _i in range(swarmSize) ]

## Update the best position.
if opt == 'min':
    GBest = Particles[ np.array([ f(_i) for _i in Particles ]).argmin() ]
else:
    GBest = Particles[ np.array([ f(_i) for _i in Particles ]).argmax() ]
PBest = Particles

## Record the start time.
Start = time.time()




### Optimization
"                               ..... Optimization .....                             "
## Start the PSO.
# For each round of iteration.
for rd in range(maxIterNum):
    if opt == 'min':
        ## Keep track of the fittest known location of each particle that it has discovered so far.
        PBest = [ _i if f(_i) < f(_j) else _j for _i, _j in zip(PBest, Particles) ]
        ## Give the fittest known location that any of the informants of particle have discovered so far.
        IBest = Particles[ np.array([ f(_i) for _i in Particles ]).argmin() ]
    else:
        PBest = [ _i if f(_i) > f(_j) else _j for _i, _j in zip(PBest, Particles) ]
        IBest = Particles[ np.array([ f(_i) for _i in Particles ]).argmax() ]

    # For each particle.
    for _i in range(swarmSize):
        ## Determine how to mutate.
        # For each particle, update its velocity by adding vectors pointing toward PBest, IBest, and GBest with random degree.
        Velocity[_i] = alpha * Velocity[_i] + \
                         np.random.uniform(0, beta) * (PBest[_i] - Particles[_i]) + \
                         np.random.uniform(0, gamma) * (IBest - Particles[_i]) + \
                         np.random.uniform(0, delta) * (GBest - Particles[_i])
        
        ## Mutate particle by moving it along its velocity vector.
        Particles[_i] = Particles[_i] + epsilon * Velocity[_i]
        
        ## Make sure the mutated particle stays within domain.
        # The lower bound.
        Particles[_i] = np.where( Particles[_i] < domain[0], domain[0], Particles[_i] )
        # The upper bound.
        Particles[_i] = np.where( Particles[_i] > domain[1], domain[1], Particles[_i] )

    ## Update the fittest know location that has been discovered by any particle so far.
    if opt == 'min':
        CurGBest = Particles[ np.array([ f(_i) for _i in Particles ]).argmin() ]
        if f(CurGBest) < f(GBest):
            GBest = CurGBest
    else:
        CurGBest = Particles[ np.array([ f(_i) for _i in Particles ]).argmax() ]
        if f(CurGBest) > f(GBest):
            GBest = CurGBest

    ## Count the time.
    Time = time.time() - Start
    ## The execution time is over.
    if Time >= maxIterTime:
        Stop = 'Time is up.'
        break

    # The improvement of objective function satisfies the stopping criteria.
    # The first round is excluded.
    if rd != 0:
        # To avoid divide-zero error.
        if f(LastBest) == 0 and np.absolute( f(GBest) ) < improve:
            Stop = 'Solution is found.'
            break
        elif np.absolute( (f(GBest) - f(LastBest))/f(LastBest) ) < improve:
            Stop = 'Solution is found.'
            break

    ## Record the current best optimizer before running the next iteration.
    LastBest = GBest

    ## Show the current progress regularly.
    if rd%50 == 49:
        print('Iteration: ', rd + 1, ' rounds; Optimizer: ', GBest,
              '; Fitness value: ', f(GBest), '; Time: ', int(Time//60), 'min ', str(Time%60)[:6], 's.')

    ## The iteration rounds achieves maximum.
    if rd == maxIterNum - 1:
        Stop = 'Iteration ran out.'




### Result Visualization
"                               ..... Result Visualization .....                             "
## Report the final result.
print('========================================================')
print('  Final Result Table')
print('--------------------------------------------------------')
print('  Optimizer: ', GBest)
print('  The function value: ', f(GBest))
print('--------------------------------------------------------')
print('  Description: ', Stop)
print('  Total iteration rounds:', rd + 1)
print('  The time used: ', int(Time//60), 'min ', str(Time%60)[:6], 's.')
print('========================================================')

