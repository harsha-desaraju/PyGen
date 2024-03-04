# This file contains code for implementation of binary Genetic optimizer.

import numpy as np
from joblib import Parallel, delayed
from ._base import Optimizer
from tqdm import tqdm



class BinaryOptimizer(Optimizer):
    """
        Binary optimizer is optimizing an array of 0's and 1's.
    """
    def __init__(self, n_pop, n_gen, chrom_len, selection_rate=0.5, 
                elite=0.1, mu=0.05, n_child= 1, rnd_state=None, 
                n_jobs = None
        ):

        super().__init__(
            n_pop = n_pop, 
            n_gen = n_gen, 
            chrom_len = chrom_len, 
            selection_rate=selection_rate, 
            elite = elite, 
            mu = mu,
            n_child = n_child
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
                delayed(cost_function)(chrom) for chrom in self.pop
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
                self.elite, self.n_child, rnd_state = self.rnd_state, 
                **pairing_args
            )
            p1, p2 = pairing.select()



            # Generate the children by crossing over the parents
            crossover = crossover_scheme(
                p1, p2, rnd_state = self.rnd_state,
                type = self.__class__.__name__,
                n_child = self.n_child
            )
            children = crossover.mate()

            # Mutate the generated children
            mut_mask = self.R.choice(
                [0,1], p=(1-self.mu, self.mu), size=(len(p1)*self.n_child, self.chrom_len)
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
        delay = [delayed(cost_function)(chrom) for chrom in self.pop]
        scores = parallel(delay)
        scores = np.array(scores)
        
        self.pop = self.pop[scores.argsort()]
        scores = np.sort(scores)

        return scores, self.pop






class ContinuousOptimizer(Optimizer):
    """ A optimizer for continuous variables """
    def __init__(self, n_pop, n_gen, chrom_len, bounds, selection_rate=0.5, 
                elite=0.1, mu=0.05, n_child= 1, rnd_state=None, 
                n_jobs = None
            ):
        
        super().__init__(
            n_pop = n_pop, 
            n_gen = n_gen, 
            chrom_len = chrom_len, 
            selection_rate=selection_rate, 
            elite = elite, 
            mu = mu,
            n_child = n_child
        )

        self.rnd_state = rnd_state
        self.n_jobs = n_jobs
        self.bounds = (
            np.tile(bounds[0], [self.n_pop, 1]),
            np.tile(bounds[1], [self.n_pop, 1])
        )

        self.R = np.random.RandomState(seed=rnd_state)

        self.pop = self.R.uniform(self.bounds[0], self.bounds[1])

        self.bounds = (
            self.bounds[0][int(self.n_pop*self.elite):],
            self.bounds[1][int(self.n_pop*self.elite):]
        )


    def optimize(
            self, cost_function, crossover_scheme,
            pairing_scheme, crossover_args={}, 
            pairing_args={}, verbose=False, show_progress=True):
        
        for gen in tqdm(range(1, self.n_gen+1), disable=not show_progress):

            if not self.n_jobs is None:
                parallel = Parallel(n_jobs=self.n_jobs)
                delay = [
                    delayed(cost_function)(chrom) for chrom in self.pop
                ]
                scores = parallel(delay)
                scores = np.array(scores).astype(np.float32)
            else:
                scores = np.zeros((self.n_pop,))
                for i in range(self.n_pop):
                    scores[i] = cost_function(self.pop[i])
            
            self.pop = self.pop[scores.argsort()]
            scores = np.sort(scores)

            if verbose:
                print(f"Best cost at Generation {gen}: ",scores[0])

            # Select 2 sets of parents using the pairing scheme
            pairing = pairing_scheme(
                self.pop, scores, self.selection_rate, 
                self.elite, self.n_child, rnd_state = self.rnd_state, 
                **pairing_args
            )
            p1, p2 = pairing.select()



            # Generate the children by crossing over the parents
            crossover = crossover_scheme(
                p1, p2, rnd_state = self.rnd_state,
                type = self.__class__.__name__,
                n_child = self.n_child,
                **crossover_args
            )
            children = crossover.mate()

            # Mutate the generated children to maintain randomness and to explore
            # Add or subtract a random from from the gene within its bounds
            add = (self.bounds[1] - children)*self.R.uniform(
                0,1,(len(children), self.chrom_len))*self.R.choice([0, 1],
                p=(1-self.mu/2, self.mu/2), size=(len(children), self.chrom_len))     # ------ why self.mu/2 ????
            
            sub = (self.bounds[0] - children)*self.R.uniform(
                0,1, (len(children), self.chrom_len))*self.R.choice([0,1],
                p=(1-self.mu/2, self.mu/2), size=(len(children), self.chrom_len))

            children = children + add + sub

            # Add the top performing chromozomes to the population
            children = np.concatenate(
                [children, self.pop[:int(self.elite*self.n_pop)]]
            )

            # Children become the population
            self.pop = children

        # Sort the population
        parallel = Parallel(n_jobs=self.n_jobs)
        delay = [delayed(cost_function)(chrom) for chrom in self.pop]
        scores = parallel(delay)
        scores = np.array(scores)
        
        self.pop = self.pop[scores.argsort()]
        scores = np.sort(scores)

        return scores, self.pop



class DiscreteOptimizer(Optimizer):
    """ A optimizer for continuous variables """
    def __init__(self, n_pop, n_gen, chrom_len, bounds, selection_rate=0.5, 
                elite=0.1, mu=0.05, n_child= 1, rnd_state=None, 
                n_jobs = None
            ):
        
        super().__init__(
            n_pop = n_pop, 
            n_gen = n_gen, 
            chrom_len = chrom_len, 
            selection_rate=selection_rate, 
            elite = elite, 
            mu = mu,
            n_child = n_child
        )

        self.rnd_state = rnd_state
        self.n_jobs = n_jobs
        self.bounds = (
            np.tile(bounds[0], [self.n_pop, 1]),
            np.tile(bounds[1], [self.n_pop, 1])
        )

        self.R = np.random.RandomState(seed=rnd_state)

        self.pop = self.R.randint(self.bounds[0], self.bounds[1])

        self.bounds = (
            self.bounds[0][int(self.n_pop*self.elite):],
            self.bounds[1][int(self.n_pop*self.elite):]
        )


    def optimize(
            self, cost_function, crossover_scheme, 
            pairing_scheme, crossover_args={},
            pairing_args={}, verbose=False, show_progress=True):
        
        
        for gen in tqdm(range(1, self.n_gen+1),disable=not show_progress):
            # print(self.pop)
            # Calculate the costs in parallel       ------------   ???? see if this can be improved??????
    
            if not self.n_jobs is None:
                parallel = Parallel(n_jobs=self.n_jobs)
                delay = [
                    delayed(cost_function)(chrom) for chrom in self.pop
                ]
                scores = parallel(delay)
                scores = np.array(scores).astype(np.float32)
            else:
                scores = np.zeros((self.n_pop,))
                for i in range(self.n_pop):
                    scores[i] = cost_function(self.pop[i])
            
            self.pop = self.pop[scores.argsort()]
            scores = np.sort(scores)

            if verbose:
                print(f"Best cost at Generation {gen}: ",scores[0])

            # Select 2 sets of parents using the pairing scheme
            pairing = pairing_scheme(
                self.pop, scores, self.selection_rate, 
                self.elite, self.n_child, rnd_state = self.rnd_state, 
                **pairing_args
            )
            p1, p2 = pairing.select()



            # Generate the children by crossing over the parents
            crossover = crossover_scheme(
                p1, p2, rnd_state = self.rnd_state,
                type = self.__class__.__name__,
                n_child = self.n_child,
                **crossover_args
            )
            children = crossover.mate()

            # Mutate the generated children to maintain randomness and to explore
            # Add or subtract a random from from the gene within its bounds
            add = (self.bounds[1] - children)*self.R.uniform(
                0,1,(len(children), self.chrom_len))*self.R.choice([0, 1],
                p=(1-self.mu/2, self.mu/2), size=(len(children), self.chrom_len))     # ------ why self.mu/2 ????
            
            sub = (self.bounds[0] - children)*self.R.uniform(
                0,1, (len(children), self.chrom_len))*self.R.choice([0,1],
                p=(1-self.mu/2, self.mu/2), size=(len(children), self.chrom_len))

            children = children + add + sub

            # Add the top performing chromozomes to the population
            children = np.concatenate(
                [children, self.pop[:int(self.elite*self.n_pop)]]
            )

            # Children become the population
            self.pop = np.round(children).astype(np.int32)

        # Sort the population
        if not self.n_jobs is None:
            parallel = Parallel(n_jobs=self.n_jobs)
            delay = [
                delayed(cost_function)(chrom) for chrom in self.pop
            ]
            scores = parallel(delay)
            scores = np.array(scores)
        else:
            scores = np.zeros((self.n_pop,))
            for i in range(self.n_pop):
                scores[i] = cost_function(self.pop[i])
        
        self.pop = self.pop[scores.argsort()]
        scores = np.sort(scores)

        return scores, self.pop

        
