# This file contains the implementations of all the pairing functions for genetic algorithm

import numpy as np






class RandomPairing:
    """
        This method selects parents at random from a subset
        of population. The population is ordered in increasing
        order of their costs. Only the top `selection_rate*n_pop`
        population is considered for mating.
    """

    def __init__(self, pop, costs, selection_rate, elite, n_child=1, rnd_state=None):
        self.pop = pop
        self.costs = costs
        self.selection_rate = selection_rate
        self.elite = elite
        self.n_child = n_child

        self.R = np.random.RandomState(seed=rnd_state)
        
        
    def select(self):
        n_parent = int((1-self.elite)*len(self.pop)/self.n_child)

        p1 = self.pop[self.R.randint(0, int(self.selection_rate*len(self.pop)), n_parent)]
        p2 = self.pop[self.R.randint(0, int(self.selection_rate*len(self.pop)), n_parent)]

        return p1, p2
