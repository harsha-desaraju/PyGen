# This file contains code for implementation of binary Genetic optimizer.

import numpy as np
from joblib import Parallel, delayed
from ._base import Optimizer



class BinaryOptimizer(Optimizer):
    """
        Binary optimizer is optimizing an array of 0's and 1's.
    """
    def __init__(self, n_pop, n_gen, chrom_len, selection_rate=0.5, 
                elite=0.1, mu=0.05, rnd_state=None, n_jobs = -1
        ):

        super().__init__(
            n_pop = n_pop, 
            n_gen = n_gen, 
            chrom_len = chrom_len, 
            selection_rate=selection_rate, 
            elite = elite, 
            mu = mu
        )

        self.rnd_state = rnd_state
        self.n_jobs = n_jobs

        self.R = np.random.RandomState(seed=rnd_state)

        self.pop = self.R.randint(
            0, 2, (self.n_pop, self.chrom_len)
        )
        

    def optimize(self, cost_function, crossover_scheme, 
            pairing_scheme, pairing_args={}, verbose=True
        ):
        
        for gen in range(1, self.n_gen+1):
            
            # Calculate the costs in parallel
            parallel = Parallel(n_jobs=self.n_jobs)
            delay = [
                delayed(cost_function.cost)(chrom) for chrom in self.pop
            ]
            scores = parallel(delay)
            scores = np.array(scores)
        
            self.pop = self.pop[scores.argsort()]
            scores = np.sort(scores)

            if verbose:
                print(f"Best cost at Generation {gen}: ",scores[0])

            # Select 2 sets of parents using the pairing scheme
            pairing = pairing_scheme(
                self.pop, scores, self.selection_rate, 
                self.elite, rnd_state = self.rnd_state, 
                **pairing_args
            )
            p1, p2 = pairing.select()


            # Generate the children by crossing over the parents
            crossover = crossover_scheme(
                p1, p2, rnd_state = self.rnd_state
            )
            children = crossover.mate()

            # Mutate the generated children
            mut_mask = self.R.choice(
                [0,1], p=(1-self.mu, self.mu), size=(len(p1), self.chrom_len)
            ).astype(bool)

            children = np.where(
                mut_mask, np.logical_not(children), children
            )

            # Add the top performing chromozomes to the population
            children = np.concatenate(
                [children, self.pop[:int(self.elite*self.n_pop)]]
            )

            # Children become the population
            self.pop = children

        # Sort the population
        parallel = Parallel(n_jobs=self.n_jobs)
        delay = [delayed(cost_function.cost)(chrom) for chrom in self.pop]
        scores = parallel(delay)
        scores = np.array(scores)
        
        self.pop = self.pop[scores.argsort()]
        scores = np.sort(scores)

        return scores, self.pop



