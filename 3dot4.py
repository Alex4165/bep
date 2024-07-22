import time

import numpy as np
from matplotlib import pyplot as plt

from model import stability_run
from auxiliary_functions import dx_dt, exp, rootfinder, format_time_elapsed, get_cached_performance
from network_class import Network
from data_eater import one_laurence, reduce

# --- Section 3.4: Application of the One-Dimensional Laurence reduction to Cowan-Wilson dynamics --- #

# --- Parameters --- #
N = 100
dt = 5e-1
stabtol = 1e-4
size = 20
p = 0.2
tau = 1
mu = 2
perturbations = 2
perturb = 1


def decay(x): return -x
def interact(x, y, _tau, _mu): return 1 / (1 + exp(_tau * (_mu - y))) - 1 / (1 + exp(_tau * _mu))


if __name__ == "__main__":
    t0 = time.time()
    net = Network()
    net.gen_random(size, p)
    net.randomize_weights(lambda x: 1 + 5*x)
    A = net.adj_matrix
    results = []
    alpha = 4
    while alpha > 1:
        a, alpha, beta = reduce(A)

        def red_interact(x, y): return interact(x, y, tau, mu)
        def integrator(arr): return dx_dt(tuple(arr), F=decay, G=red_interact, W=tuple(tuple(row) for row in A))
        eq_vector = stability_run(integrator, dt, stabtol, 100*net.size*np.random.random_sample(net.size))[0]
        actual_r = a @ eq_vector

        def reduced(R): return decay(R) + alpha * interact(beta * R, R, tau, mu)
        roots = rootfinder(reduced, [-0.1, 2 * actual_r])
        for root in roots:
            if root > 1e-2:
                results.append([alpha, root, actual_r])

        for j in range(perturbations):
            try:
                m = np.random.choice([k for k in range(A.shape[0]) if sum(A[k, :]) > perturb])
                n = np.random.choice([k for k in range(A.shape[1]) if A[m, k] > perturb])
                A[m, n] -= perturb
            except ValueError:  # i.e. If all entries are less than perturb
                perturb *= 0.5

    results = np.array(results).T
    plt.plot(results[0], results[1], '.', label="Pred")
    plt.plot(results[0], results[2], '.', label="Act")
    plt.legend()
    plt.show()

    get_cached_performance()

    print(format_time_elapsed(time.time() - t0))












