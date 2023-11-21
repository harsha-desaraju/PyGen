# This file contains the base class for optimizers



# Add more tests and better testing for the parameters



class Optimizer:
    """ A base/generic class for implementing Genetic Algorithm. """

    def __init__(self, n_pop, n_gen, chrom_len, 
                selection_rate, elite, mu, n_child
        ):

        self.n_pop = n_pop
        self.n_gen = n_gen
        self.chrom_len = chrom_len
        self.selection_rate = selection_rate
        self.elite = elite
        self.mu = mu
        self.n_child = n_child

        assert n_pop > 1 and isinstance(n_pop, int)
        assert n_gen > 1 and isinstance(n_gen, int)
        assert chrom_len > 1 and isinstance(chrom_len, int)
        assert selection_rate >= 0.0 and selection_rate <= 1.0
        assert elite >= 0.0 and elite <= 1.0
        assert mu >= 0.0 and mu <= 1.0
        assert n_child > 0 and n_child < 3

    def optimize(self):
        pass