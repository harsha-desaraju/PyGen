
import time
import numpy as np
from optimizers._binary import BinaryOptimizer
from crossover._crossover import UniformCrossover, KPointCrossover
from selection._selection import RandomSelection, RankSelection, RouletteWheelSelection, TounamentSelection
from functions._functions import TestFunction
import matplotlib.pyplot as plt



if __name__ == "__main__":
    n_pop = 10
    n_gen = 20
    chrom_len = 20

    # cost_func = TestFunction()

    # # Testing Uniformcrossover
    # optimizer = BinaryOptimizer(n_pop, n_gen, chrom_len, n_child=1, rnd_state=0)
    # scores, pop = optimizer.optimize(
    #     cost_func, UniformCrossover,
    #     RouletteWheelSelection
    # )

    # print()

    # # # Testing KPointCrossover
    # # optimizer = BinaryOptimizer(n_pop, n_gen, chrom_len)
    # # scores, pop = optimizer.optimize(
    # #     cost_func, KPointCrossover,
    # #     RandomSelection, 
    # # )

    lb = np.array([1,2,3,4.1,5])
    up = np.array([3,4,5,6.8,7.3])

    # pop = [np.random.uniform(lb, up, (len(lb),)).reshape(1, -1) for _ in range(n_pop)]
    # pop = np.concatenate(pop, axis=0)
    # print(pop)
    # print()

    # lb = np.tile(lb, [n_pop, 1])
    # up = np.tile(up, [n_pop, 1])

    # pop = np.random.uniform(lb, up)
    # print(pop)

    arr = np.concatenate([lb, up])
    print(arr)
    arr = np.round(arr)
    print(arr)




        







