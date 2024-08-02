import json
import time

import numpy as np
from matplotlib import pyplot as plt

from model import stability_run
from auxiliary_functions import dx_dt, exp, rootfinder, format_time_elapsed, get_cached_performance
from network_class import Network
from data_eater import one_laurence, reduce

# --- Section 3.4: Application of the One-Dimensional Laurence reduction to Cowan-Wilson dynamics --- #

# --- Parameters --- #
dt = 5e-1
stabtol = 1e-4
size = 20
p = 0.6
tau = 1
mu = 2
perturbations = 1
perturb = 0.1


def decay(x): return -x
def interact(x, y, _tau, _mu): return 1 / (1 + exp(_tau * (_mu - y))) - 1 / (1 + exp(_tau * _mu))


def gen_alpha_dependence():
    global perturb
    t0 = time.time()
    net = Network()
    net.gen_random(size, p)
    net.randomize_weights(lambda x: 2 + 5 * x)
    A = net.adj_matrix
    results = []
    alpha = 4
    i = 0
    while alpha > 0.5:
        i += 1
        if i % 50 == 0:
            print(alpha, format_time_elapsed(time.time()-t0), "total", format_time_elapsed())
        a, alpha, beta = reduce(A)
        alpha = abs(alpha)

        def reduc_interact(x, y):
            return interact(x, y, tau, mu)

        def integrator(arr):
            return dx_dt(tuple(arr), F=decay, G=reduc_interact, W=tuple(tuple(row) for row in A))

        eq_vector = stability_run(integrator, dt, stabtol, 100 * net.size * np.random.random_sample(net.size))[0]
        actual_r = a @ eq_vector

        def reduced(R):
            return decay(R) + alpha * interact(beta * R, R, tau, mu)

        roots = rootfinder(reduced, [0, 10 + 2 * actual_r], etol=1e-3)
        for root in roots:
            results.append([alpha, root, actual_r])

        for j in range(perturbations):
            row_choices = []
            for i, row in enumerate(A):
                is_added = False
                for x in row:
                    if not is_added and x > perturb:
                        is_added = True
                if is_added:
                    row_choices.append(i)
            if len(row_choices) > 0:
                m = np.random.choice(row_choices)
                column_choices = []
                for i, x in enumerate(A[m, :]):
                    if x > perturb:
                        column_choices.append(i)
                n = np.random.choice(column_choices)
                A[m, n] -= perturb
            else:
                perturb *= 0.9
    filename = 'data/3dot4_1d_reduc_connect'
    results = np.array(results).T.tolist()
    with open(filename, "w") as f:
        json.dump(results, f)
    plt.plot(results[0], results[1], '.', label="Pred")
    plt.plot(results[0], results[2], '.', label="Act")
    plt.legend()
    plt.xlabel("Alpha")
    plt.ylabel("R")
    plt.show()
    get_cached_performance()
    print(format_time_elapsed(time.time() - t0))


if __name__ == "__main__":

    # How accurate is the prediction when reducing the network's connection strength #
    # gen_alpha_dependence()

    # How does accuracy depend on the parameters?
    p1_range = np.arange(0.1, 10.1, 0.1)
    p2_range = np.arange(5, 20, 0.1)

    net = Network()
    net.gen_random(size, p)
    a, alpha, beta = reduce(net.adj_matrix)
    A = tuple(tuple(row) for row in net.adj_matrix)
    results = []

    for p1 in p1_range:
        for p2 in p2_range:
            x0 = 100 * net.size * np.random.random_sample(net.size)

            def reduc_interact(x, y): return interact(x, y, p1, p2)
            def integrator(arr): return dx_dt(tuple(arr), F=decay, G=reduc_interact, W=A)
            eq_vector = stability_run(integrator, dt, stabtol, x0)[0]
            actual_r = a @ eq_vector

            def reduced(R): return decay(R) + alpha * interact(beta * R, R, p1, p2)
            pred_r = stability_run(reduced, dt, stabtol, a @ x0)[0]

            results.append([p1, p2, actual_r, pred_r])
    results = np.array(results).T.tolist()
    filename = 'data/3dot4_1d_reduc_params'
    with open(filename, "w") as f:
        json.dump(results, f)
    mi, ma = min(results[2]+results[3]), max(results[2]+results[3])
    plt.scatter(results[0], results[1], c=results[3], s=100, vmin=mi, vmax=ma)
    plt.scatter(results[0], results[1], c=results[2], s=25, vmin=mi, vmax=ma)
    plt.colorbar()
    plt.show()


