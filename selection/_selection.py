# This file contains the implementations of all the pairing functions for genetic algorithm

import numpy as np



class Selection:
    """ A base class for selection. """

    def __init__(self, pop, costs, selection_rate, elite, n_child):
        self.pop = pop
        self.costs = costs
        self.selection_rate = selection_rate
        self.elite = elite
        self.n_child = n_child

    def select(self):
        pass




class RandomSelection(Selection):
    """
        This method selects parents at random from a subset
        of population. The population is ordered in increasing
        order of their costs. Only the top `selection_rate*n_pop`
        population is considered for mating.
    """

    def __init__(self, pop, costs, selection_rate, elite, n_child=1, rnd_state=None):
        super().__init__(
            pop= pop,
            costs= costs,
            selection_rate= selection_rate,
            elite= elite,
            n_child= n_child
        )

        self.R = np.random.RandomState(seed=rnd_state)
        
        
    def select(self):
        n_parent = int((1-self.elite)*len(self.pop)/self.n_child)

        p1 = self.pop[self.R.randint(0, int(self.selection_rate*len(self.pop)), n_parent)]
        p2 = self.pop[self.R.randint(0, int(self.selection_rate*len(self.pop)), n_parent)]

        return p1, p2



class RankSelection(Selection):
    """
        The population is ordered in decreasing order of their costs.
        A chromosome  with higher rank is selected more often as a 
        parent. The probabilities are calculated based on the rank.
    """

    def __init__(self, pop, costs, selection_rate, elite, n_child=1, rnd_state=None):
        super().__init__(
            pop= pop,
            costs= costs,
            selection_rate= selection_rate,
            elite= elite,
            n_child= n_child
        )

        self.R = np.random.RandomState(seed=rnd_state)

    def select(self):
        # ------ Assuming the population is sorted !!!!

        n_parent = int((1-self.elite)*len(self.pop)/self.n_child)

        # The fraction from which parents are selected
        s = int(self.selection_rate*len(self.pop))

        prob = (s - np.arange(1, s+1)+1)/(s*(s+1)/2)
        
        inds1 = self.R.choice(a=np.arange(s), p=prob, size=(n_parent,))
        inds2 = self.R.choice(a=np.arange(s), p=prob, size=(n_parent,))

        # Since some of the places same chromosome is selected as both 
        # the parents, randomly change the 2nd parent if they are same.
        inds2 = np.where(inds1-inds2==0, np.random.randint(s), inds2)

        p1 = self.pop[inds1]
        p2 = self.pop[inds2]

        return p1, p2
    

class RouletteWheelSelection(Selection):
    """
        The probability of selection of a chromosome as a 
        parent is inversely proportional to its cost. The 
        probabilities are calculated based on their costs.
        Lower the cost higher the chance of being a parent.
    """

    def __init__(self, pop, costs, selection_rate, elite, n_child=1, rnd_state=None):
        super().__init__(
            pop= pop,
            costs= costs,
            selection_rate= selection_rate,
            elite= elite, 
            n_child= n_child
        )

        self.R = np.random.RandomState(seed=rnd_state)

    def select(self):
        ## ----- Assuming the population is sorted! -----
        n_parent = int((1-self.elite)*len(self.pop)/self.n_child)

        # The fraction from which parents are selected
        s = int(self.selection_rate*len(self.pop))

        prob = self.costs[:s] - self.costs[s]

        # If the costs of all the chromosomes is same,the prob
        # would become a zero vector. So make it uniform vector.
        if np.sum(prob) == 0:
            prob = np.full(shape=(s,), fill_value=1/s)
        else:
            prob = np.abs(prob/np.sum(prob))

        inds1 = self.R.choice(a=np.arange(s), p=prob, size=(n_parent,))
        inds2 = self.R.choice(a=np.arange(s), p=prob, size=(n_parent,))


        # Since some of the places same chromosome is selected as both 
        # the parents, randomly change the 2nd parent if they are same.
        inds2 = np.where(inds1-inds2==0, self.R.randint(s), inds2)


        p1 = self.pop[inds1]
        p2 = self.pop[inds2]

        return p1, p2


class TounamentSelection(Selection):
    """
        In this method, a subset of chromosomes are randomly selected
        from the population and the chromosome with least cost is
        chosen as a parent. This method is repeated for each parent.
    """

    def __init__(self, pop, costs, selection_rate, elite, n_child=1, subset_size=0.3, rnd_state=None):
        super().__init__(
            pop= pop,
            costs= costs,
            selection_rate= selection_rate,
            elite= elite,
            n_child= n_child
        )

        self.subset_size = int(subset_size*len(self.pop))
        self.R = np.random.RandomState(seed=rnd_state)

    
    def select(self):
        ## ----- Assuming the population is sorted!!!
        n_parent = int((1-self.elite)*len(self.pop)/self.n_child)

        # The fraction from which parents are selected
        s = int(self.selection_rate*len(self.pop))

        # Randomly select a subset of indices for each parent
        rand1 = np.random.randint(0, s, (n_parent, self.subset_size))
        rand2 = np.random.randint(0, s, (n_parent, self.subset_size))

        # Find the min cost among the subset
        ids1 = self.costs[rand1].argmin(axis=1)
        ids2 = self.costs[rand2].argmin(axis=1)

        # Find the index of the min cost chromosome
        inds1 = rand1[np.indices(ids1.shape)[0], ids1]
        inds2 = rand2[np.indices(ids2.shape)[0], ids2]

        # Since some of the places same chromosome is selected as both 
        # the parents, randomly change the 2nd parent if they are same.
        inds2 = np.where(inds1-inds2==0, np.random.randint(s), inds2)        

        # Selecr parents with minimum cost in subset
        p1 = self.pop[inds1]
        p2 = self.pop[inds2]

        return p1, p2







        



        
