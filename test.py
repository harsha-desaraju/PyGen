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
## May be add Single and double point crossover seperately
## Remove the sorting before sending ro selection. Let the 
#  selection method do it if needed. 





if __name__ == '__main__':

    n_pop = 20

    pop = np.random.randint(0, 2, (n_pop, 5))
    costs = np.random.uniform(0, 1, (n_pop, ))
    print(costs)

    rand = np.random.randint(0, n_pop, (n_pop//2, 3))
    print(rand)

    print(costs[rand])

    ids = costs[rand].argmin(axis=1)

    print(rand[np.indices(ids.shape)[0], ids])

    
    




