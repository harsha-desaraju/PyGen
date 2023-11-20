

from optimizers._binary import BinaryOptimizer
from crossover._crossover import UniformCrossover
from pairing._pairing import RandomPairing
from functions._functions import TestFunction

## Add number of children as a parameter  - Done
## optimize binary optimizer by using np.uint8
## Directly pass the classes to the optimize. Instantiate objects inside the method.
## Write unit test also. Might be useful later when new commits are made.
## Try paralleizing other parts as well. Like mutation and selection
## Add gene len as an option




if __name__ == "__main__":
    n_pop = 100
    n_gen = 80
    chrom_len = 50

    cost_func = TestFunction()

    
    optimizer = BinaryOptimizer(n_pop, n_gen, chrom_len)
    scores, pop = optimizer.optimize(
        cost_func, UniformCrossover,
        RandomPairing
    )

    print(scores[0])






