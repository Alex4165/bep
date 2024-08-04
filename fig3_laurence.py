import time

import numpy as np
from matplotlib import pyplot as plt

from auxiliary_functions import exp, format_time_elapsed
from data_eater import reduce
from model import Model
from network_class import Network

dt = 1e-1
stabtol = 1e-4
tau = 1
mu = 3
center_out = 1
center_in = 2
size = 6
alphas = np.arange(5, 9, 0.2)
kouts = np.arange(2, 10, 0.2)

A = np.zeros((size, size))
A[0, 1:] = np.array([center_in for i in range(size - 1)])
A[1:, 0] = np.array([center_out for i in range(size - 1)]).T
A[0, 1:] = 2 * np.random.random_sample(size - 1)
A[1:, 0] = np.random.random_sample(size - 1).T


def f(x): return -x


def g(x, y): return 1 / (1 + exp(tau * (mu - y)))


one_d = True
# One dimensional
if one_d:
    alpha_zero = np.linalg.norm(max(np.linalg.eigvals(A), key=lambda x: np.linalg.norm(x)))
    results = [[], [], []]
    t1 = time.time()
    for alpha in alphas:
        print(alpha, time.time()-t1)
        t1 = time.time()
        w = alpha / alpha_zero * A
        model = Model(F=f, G=g, W=w)

        large_x0 = 100 * size * np.random.random_sample(size)
        small_x0 = np.random.random_sample(size)
        alpha_, sim_large, pred, _ = model.laurence_one_dim_analysis_run(large_x0, dt, stabtol)
        sim_small = model.laurence_one_dim_analysis_run(small_x0, dt, stabtol)[1]
        if abs(alpha - alpha_) > 0.1:
            print("OH FUCK:", alpha, alpha_)
            reduce(w)
        for p in pred:
            for sim in [sim_small, sim_large]:
                results[0].append(alpha)
                results[1].append(p)
                results[2].append(sim)

    plt.plot(results[0], results[1], '.', label='Pred')
    plt.plot(results[0], results[2], '.', label='Sim')
    plt.title('One-dimensional')
    plt.xlabel('alpha')
    plt.ylabel('R')
    plt.legend()
    plt.show()

# Two dimensional
two_d = True
if two_d:
    model = Model(F=f, G=g, W=A)
    alphas_ = model.laurence_multi_dim_analysis_run(x0=np.random.random_sample(size), dim=2, dt=1)[3]
    alpha_zero = (alphas_[0] + (size - 1) * alphas_[1]) / size
    results = [[], [], []]
    for kout in kouts:
        w = kout / alpha_zero * A
        model = Model(F=f, G=g, W=w)
        for x0 in [100*np.random.random_sample(size), 0.1*np.random.random_sample(size)]:
            sim, pred, a, alphas_, betas = model.laurence_multi_dim_analysis_run(x0, dim=2, dt=dt, stabtol=stabtol)
            p = (pred[0] + (size - 1) * pred[1]) / size
            s = (sim[0] + (size - 1) * sim[1]) / size
            results[0].append(kout)
            results[1].append(p)
            results[2].append(s)

    plt.plot(results[0], results[1], '.', label='Pred')
    plt.plot(results[0], results[2], '.', label='Sim')
    plt.title('Two-dimensional')
    plt.xlabel('<kout>')
    plt.ylabel('<x>')
    plt.legend()
    plt.show()
