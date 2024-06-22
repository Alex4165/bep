import numpy as np
from scipy.optimize import fsolve

dimension = 6
B = np.random.rand(dimension, dimension)-0.5*np.ones((dimension, dimension))

kmax, kmaxn = -1e6, 0
indices_checked = []
A = np.copy(B)
for n in range(dimension):
    print(np.matrix(A))

    degs = np.transpose(np.dot(A, np.ones(A.shape[0])))

    i = np.argmin(degs)
    indices_checked.append(i)

    if degs[i] >= kmax:
        kmax = degs[i]
        kmaxn = n

    print(degs[i], kmax, kmaxn)
    print()

    A = np.delete(A, i, 0)
    A = np.delete(A, i, 1)

C = np.copy(B)
kmax = -1e6
indices_checked = []
for n in range(dimension):
    degs_all = np.transpose(np.dot(C, np.ones(C.shape[0]))).tolist()
    print(degs_all)
    degs = [degs_all[i] for i in range(dimension) if i not in indices_checked]
    print(degs)
    i = np.argmin(degs)
    i = degs_all.index(degs[i])
    print(degs_all[i], kmax)
    print()

    if degs_all[i] > kmax:
        kmax = degs_all[i]

    indices_checked.append(i)

    for j in range(dimension):
        C[i, j] = 0
        C[j, i] = 0

print(C)
print(kmax)

