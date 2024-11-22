'''
Just randomly generate some instances and solve their LP, to check that you can have two disjoint sets of jobs
'''
from algorithms import JS_LP
import numpy as np
from tqdm import tqdm

TOL = 1e-6

n = 3
m = 2

np.random.seed(7)
def get_unfixed(X):
    n, m = X.shape
    unfixed = []
    for i in range(n):
        for j in range(m):
            if abs(round(X[i, j]) - X[i, j]) > TOL:
                unfixed.append(i)
                break
    return unfixed

for _ in tqdm(range(1)):
    #P = np.random.randint(1, 100, (n, m))
    P = [[35, 67], [36, 27], [1, 73]]
    P = np.asarray(P)
    obj_0, X_s = JS_LP(P)

    unfixed = get_unfixed(X_s)


    if len(unfixed) >= 1:
        # Pick the first unfixed job
        for i in unfixed:
            unfixed_all = []
            for j in range(m):
                obj, X = JS_LP(P, fixed=[((i, j), 1)])
                unfixed_all.append(get_unfixed(X))
            if set(unfixed_all[0]).intersection(set(unfixed_all[1])) == set():
                print('Found a disjoint set of jobs!!')
                print(P)
                print(X_s)
                print(unfixed)
                print(unfixed_all)
