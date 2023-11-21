from joblib import Parallel, delayed
import time
from multiprocessing import Pool
import numpy as np
from sklearn.cross_decomposition import PLSRegression


def func(x):
    time.sleep(1)
    return x**2


def test_func(a,b,c,d,e):
    return a+b+c+d+e



## Add number of children as a parameter  - Done
## optimize binary optimizer by using np.uint8
## Directly pass the classes to the optimize. Instantiate objects inside the method.
## Write unit test also. Might be useful later when new commits are made.
## Try paralleizing other parts as well. Like mutation or selection
## Add gene len as an option
## Add convergence criteria and early stopping conditions
## How to change k in KPointCrossover?





if __name__ == '__main__':

    # chrom_len = 30
    # num_pop = 2
    # k = 3

    # p1 = np.random.randint(0, 2, (num_pop, chrom_len))
    # p2 = np.random.randint(0, 2, (num_pop, chrom_len))


    # mask = np.empty(p1.shape)
    # for ind in range(num_pop):
    #     arr = np.arange(chrom_len)

    #     points = np.random.randint(0, chrom_len, k)
    #     points = np.append(points, [0, chrom_len])
    #     points = np.sort(points)

    #     a = 0
    #     gene_mask = np.zeros(chrom_len)
    #     for i in range(1, len(points)):
    #         gene_mask += np.where(np.logical_and(arr>points[i-1], arr<points[i]), a, 0)
    #         a = np.logical_not(a)

    #     mask[ind] = gene_mask

    # # print(mask)

    # mask_comp = np.logical_not(mask)

    # children = p1*mask + p2*mask_comp

    # print(children)

    a = np.array([1,2,3])
    print(a.shape)

    print(a.reshape(1, 3))
    


