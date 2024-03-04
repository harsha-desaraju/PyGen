# Find the global minima of a non-convex function

import numpy as np
from math import exp,sin
from pygen.optimizers import ContinuousOptimizer
from pygen.crossover import UniformCrossover, KPointCrossover
from pygen.selection import ( 
    RouletteWheelSelection, 
    TournamentSelection,
    RankSelection,
    RandomSelection
    )


def cubic(x):
    # Global Optimum at x= 0.679 & -0.679 
    return 5*exp(-(x[0]**2))*sin(10*x[0]**2)




if __name__ == "__main__":

    # Size of population of GA
    n_pop = 100
    # The number of generations to 
    # run the GA for
    n_gen = 10
    # The lower and upper to search
    # Here lets find an optimum
    # between [-10, 10]
    bnds = (np.array([2]), np.array([3]))
    # The len of chromosome. Since there is 
    # only 1 variable its one here
    chrom_len = 1




    ga_optimizer = ContinuousOptimizer(
        n_pop= n_pop,
        n_gen= n_gen,
        chrom_len= chrom_len,
        bounds= bnds,
        selection_rate= 0.5,
        elite= 0.1,
        # mu=0.5,
        n_jobs=-1
    )
    scores, pop = ga_optimizer.optimize(
        cost_function= cubic,
        crossover_scheme= UniformCrossover,
        pairing_scheme= RandomSelection,
        # crossover_args= {'k':1},
        verbose= True,
        show_progress=False

    )

    print(f"The best optimum found is at {round(pop[0][0],3)}")

