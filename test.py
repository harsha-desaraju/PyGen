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






if __name__ == '__main__':

    x = np.random.uniform(0, 10, (10, 10))
    y = np.random.uniform(0, 10, (10,))

    pls = PLSRegression(n_components="harsha")
    pls.fit(x, y)
    pred = pls.predict(x)
    print(pred)


    i = 10.0

    print(isinstance(i, int))
    
    R = np.random.RandomState(seed=10)

    a = R.randint(0, 2, (10, ))

    # b = R.randint(0, 10, (3,10))

    print(np.where(a == 1)[0])
    print(np.sum(np.where(a == 1)[0]))

    # num = 10
    
    # t1 = time.time()
    # parallel = Parallel(n_jobs=2)
    # delay = [delayed(func)(i) for i in range(num)]
    # res = parallel(delay)
    # t2 = time.time()
    # print(t2-t1)

    # print()

    # t1 = time.time()

    # args = [i for i in range(num)]

    # res = []
    # with Pool(processes=2) as pool:
    #     for sq in pool.map(func, args, chunksize=2):
    #         res.append(sq)

    # t2 = time.time()

    # print(t2-t1)


