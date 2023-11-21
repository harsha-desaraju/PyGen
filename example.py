
import time
import numpy as np
from optimizers._binary import BinaryOptimizer
from crossover._crossover import UniformCrossover, KPointCrossover
from pairing._pairing import RandomPairing
from functions._functions import TestFunction
import matplotlib.pyplot as plt



if __name__ == "__main__":
    n_pop = 50
    n_gen = 20
    chrom_len = 20

    cost_func = TestFunction()

    # Testing Uniformcrossover
    optimizer = BinaryOptimizer(n_pop, n_gen, chrom_len)
    scores, pop = optimizer.optimize(
        cost_func, UniformCrossover,
        RandomPairing
    )

    print()

    # Testing KPointCrossover
    optimizer = BinaryOptimizer(n_pop, n_gen, chrom_len)
    scores, pop = optimizer.optimize(
        cost_func, KPointCrossover,
        RandomPairing, 
    )
        

        






