# This file contains implementations of different cost functions

import numpy as np







class TestFunction:
    def __init__(self):
        pass

    def cost(self, chrom):
        return np.sum(np.where(chrom == 1)[0])