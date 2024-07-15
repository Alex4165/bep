import json
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import lru_cache
from typing import Callable

from model import rootfinder, real, exp


@lru_cache(maxsize=None, typed=False)
def lambda_star(tau, mu, function: Callable[[float, float], float] = None, l0=1e-3, l1=2e3, etol=1e-2):
    """Given mu and tau finds smallest lambda s.t. a nonzero solution exists"""
    lmbda = (l0+l1)/2

    if l1 - l0 < etol:
        return lmbda

    if function is None:
        f = lambda x: -x + lmbda/(1+np.exp(tau*(mu-x))) - lmbda/(1+np.exp(tau*mu))
        # f = lambda x: -mu*x + lmbda*x**tau/(1+x**tau)
    else:
        f = lambda x: function(x, tau, mu, lmbda)

    zeros = rootfinder(f, [-0.1, l1], etol=etol)
    if len([zero for zero in zeros if zero > etol]) > 0:
        return lambda_star(tau, mu, l0=l0, l1=lmbda)
    else:
        return lambda_star(tau, mu, l0=lmbda, l1=l1)


def rho(A):
    eigs, vecs = np.linalg.eig(A)
    i = np.argmax(np.array([np.real(e) for e in eigs]))
    return np.real(eigs[i])


def kmax(A):
    kmax = 0
    for i in range(A.shape[0]):
        degs = np.transpose(np.dot(A, np.ones(A.shape[0])))
        i = np.argmin(degs)
        if degs[i] >= kmax:
            kmax = degs[i]
        A = np.delete(A, i, 0)
        A = np.delete(A, i, 1)
    return kmax


def one_laurence(res, adj_matrix, decay_func, interact_func):
    """Given a collection of systems, return a list of the one-dimensional
    Laurence reduction's predicted average in equilibrium and the weighting
    vector to retrieve the average"""
    a, alpha, beta = reduce(adj_matrix)
    alpha, beta = real(alpha), real(beta)
    R_norms = []
    for i in range(len(res[1])):
        f = lambda R: decay_func(R, res[1][i], res[2][i]) + alpha * interact_func(beta * R, R, res[1][i], res[2][i])
        # (1 / (1 + np.exp(res[1][i] * (res[2][i] - R))) - 1 / (1 + np.exp(res[1][i] * res[2][i])))
        roots = rootfinder(f, [-1, adj_matrix.shape[0]])
        R_norms.append(np.max(roots))  # Choose the largest root because 0 will always be one
    return R_norms, a


def reduce(A):
    """Return the one dimensional Laurence reduction parameters"""
    eigs, vecs = np.linalg.eig(np.transpose(A))
    k = np.argmax(np.multiply(eigs, np.conjugate(eigs)))
    a = vecs[:, k] / (sum(vecs[:, k]))
    alpha = eigs[k]
    K = np.diag([sum(A[i, :]) for i in range(A.shape[0])])
    beta = a @ K @ np.transpose(a) / (a @ np.transpose(a) * alpha)
    alpha = alpha
    beta = beta
    return a, alpha, beta


if __name__ == "__main__":
    # Load in data
    filename = "data/CowanWilsonstabilityresults1719149668.7244217"
    with open(filename+".txt", "r") as read_file:
        res = json.load(read_file)

    for lst in res:
        print(lst)
    A = np.array(res[0])

    # Find rho(A) & kmax(A)
    rho = rho(A)
    kmax = kmax(A)

    # # Find 1D reduction: R and solve its equation
    R_norms = one_laurence(res, adj_matrix=A, decay_func=lambda x, p1, p2: -x,
                           interact_func=lambda x, y, p1, p2: 1/(1+exp(p1*(p2-y))))

    # Find 2D reduction  (Appendix C of Laurence)
    eigs, vecs = np.linalg.eig(np.transpose(A))
    print(np.multiply(eigs, np.conjugate(eigs)))
    n = 2  # int(input("How many dominant eigenvalues (int)?\n"))

    i1 = np.argmax([np.real(eig) for eig in eigs])

    r = eigs[i1]
    v1 = vecs[:, i1]/(sum(vecs[:, i1]))

    i2 = eigs.index(-r)
    v2 = vecs[:, i2]/(sum(vecs[:, i2]))

    a1 = (0.5 * v1 + 0.5 * v2) / (0.5 * sum(v1) + 0.5 * sum(v2))
    # a2 =




    # # Find lambda stars (for WU analysis)
    # lambda_stars = [lambda_star(tau, mu) for (tau, mu) in zip(res[1], res[2])]
    # lower_bound_alpha = np.where(lambda_stars <= kmax, 1, 0)
    # upper_bound_alpha = np.where(lambda_stars >= rho, 1, 0)
    # print(f"Total time taken: {round((time.time()-t1) // 60)} min and {round((time.time()-t1) % 60)} s")


    # # Find Wu values for GRN
    # col = np.where(np.array(res[4]) < 1e-3, 'r', 'g')
    # hs = [h for h in np.linspace(np.min(res[1]), np.max(res[1]), 100)]
    # bs = [b for b in np.linspace(np.min(res[2]), np.max(res[2]), 100)]
    # upper = [kmax/(h*(h-1)**(1/h - 1)) for h in hs]
    # lower = [rho/(h*(h-1)**(1/h - 1)) for h in hs]
    # t1 = time.time()

    # Normalize norm of equilibrium vector
    # sig = lambda x: 1/(1+np.exp(5*(1-x))) - 1/(1+np.exp(5))
    # normalized_norms = sig(np.array(res[4]))

    # Scatter plot
    plt.scatter(res[1], res[2], c=res[4], cmap='viridis', s=200)
    plt.colorbar(label="Data norm")

    # Laurence norm prediction
    plt.scatter(res[1], res[2], c=R_norms, cmap='viridis', s=25)
    plt.colorbar(label="1D norm")

    # # Plot Wu general predictions
    # plt.scatter(res[1], res[2], c='b', alpha=lower_bound_alpha, s=25)
    # plt.scatter(res[1], res[2], c='r', alpha=upper_bound_alpha, s=25)

    # # Plot Wu GRN predictions
    # plt.plot(hs, upper, 'k')
    # plt.plot(hs, lower, 'k')
    # plt.plot(hs, [rho-3/tau for tau in hs], 'k')
    # plt.plot(hs, [kmax-3/tau for tau in hs], 'k')

    plt.xlabel("tau")
    plt.ylabel("mu")
    # plt.ylim(top=3)
    plt.title(filename)
    plt.show()


