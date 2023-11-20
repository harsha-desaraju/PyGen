# A file containing all the implementations of crossover functions.

import numpy as np

class Crossover:
    ''' A base class for implementing different crossover methodologies/functions '''

    def __init__(self, p1, p2, n_child):
        '''
            p1: parent1
            p2: parent2
            n_child: number of childern 
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
    def __init__(self, p1, p2, n_child=1, rnd_state=None):
        super().__init__(p1, p2, n_child)
        self.n_parent = len(p1)
        self.chrom_len = len(p1[0])
        self.R = np.random.RandomState(seed=rnd_state)

    def mate(self):
        gene_mask = self.R.randint(0, 2, (self.n_parent, self.chrom_len))
        gene_mask_comp = np.logical_not(gene_mask).astype(int)
        
        if self.n_child == 1:
            childern = self.p1*gene_mask + self.p2*gene_mask_comp

        else:
            child1 = self.p1*gene_mask + self.p2*gene_mask_comp
            child2 = self.p1*gene_mask_comp + self.p2*gene_mask
            childern = np.concatenate([child1, child2])
        return childern
