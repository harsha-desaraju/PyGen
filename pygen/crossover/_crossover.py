# A file containing all the implementations of crossover functions.

import numpy as np
from joblib import Parallel, delayed

class Crossover:
    ''' A base class for implementing different crossover methodologies/functions '''

    def __init__(self, p1, p2, n_child):
        '''
            p1: parent1
            p2: parent2
            n_child: number of children 
        '''
        self.p1 = p1
        self.p2 = p2
        self.n_child = n_child

        assert len(p1) == len(p2)
        assert n_child > 0 and n_child < 3

    def mate(self):
        pass




class UniformCrossover(Crossover):
    """
        In Uniform crossover, a gene is selected at random 
        from either of the parents to generate children.
        Maximum number of children that can be generated 
        using this method is two, with each child 
        complementary to the other.
    """
    def __init__(self, p1, p2, type, n_child=1, rnd_state=None):
        super().__init__(p1, p2, n_child)
        self.n_parent = len(p1)
        self.chrom_len = len(p1[0])
        self.type = type
        self.R = np.random.RandomState(seed=rnd_state)

    def mate(self):
        if self.type == 'BinaryOptimizer':
            gene_mask = self.R.randint(0, 2, (self.n_parent, self.chrom_len))
            gene_mask_comp = np.logical_not(gene_mask).astype(int)
            
            if self.n_child == 1:
                children = self.p1*gene_mask + self.p2*gene_mask_comp

            else:
                child1 = self.p1*gene_mask + self.p2*gene_mask_comp
                child2 = self.p1*gene_mask_comp + self.p2*gene_mask
                children = np.concatenate([child1, child2], axis=0)
            return children
        
        elif self.type == 'ContinuousOptimizer' or self.type == 'DiscreteOptimizer':
            beta = self.R.uniform(0, 1, (self.n_parent, self.chrom_len))

            if self.n_child == 1:
                children = beta*(self.p1 - self.p2) + self.p2

            else:
                child1 = beta*(self.p1 - self.p2) + self.p2
                child2 = beta*(self.p2 - self.p1) + self.p1
                children = np.concatenate([child1, child2], axis=0)
            return children



class KPointCrossover(Crossover):
    """ 
        KPointCrossover is a generalization of single/double point crossover.
        In this method k points are randomly chosen for each pair of parents
        and genes from each part are selected alternatively to generate children.
    """
    def __init__(self, p1, p2, type, n_child, k=1, rnd_state=None):
        super().__init__(
            p1 = p1,
            p2 = p2, 
            n_child= n_child,
        )

        assert isinstance(k, int)==True and k<len(p1[0])
        self.type = type
        self.chrom_len = len(p1[0])
        self.k = k          
        self.R = np.random.RandomState(seed=rnd_state)

    
    def mate(self):
        mask = np.empty(self.p1.shape)
        for ind in range(len(self.p1)):
            arr = np.arange(self.chrom_len)

            points = self.R.randint(0, self.chrom_len, self.k)
            points = np.append(points, [0, self.chrom_len])
            points = np.sort(points)

            a = 0
            gene_mask = np.zeros(self.chrom_len)
            for i in range(1, len(points)):
                gene_mask += np.where(np.logical_and(arr>points[i-1], arr<points[i]), a, 0)
                a = np.logical_not(a)

            mask[ind] = gene_mask

        mask_comp = np.logical_not(mask)

        if self.n_child == 1:
            children = self.p1*mask + self.p2*mask_comp
            return children
        
        else:
            child1 = self.p1*mask + self.p2*mask_comp
            child2 = self.p1*mask_comp + self.p2*mask
            return np.concatenate([
                child1, 
                child2
            ])

        
