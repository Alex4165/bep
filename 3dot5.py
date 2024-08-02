import json
import time

import numpy as np
from matplotlib import pyplot as plt

from model import stability_run, Model
from auxiliary_functions import dx_dt, exp, rootfinder, format_time_elapsed, get_cached_performance, real
from network_class import Network
from data_eater import one_laurence, reduce

# --- Section 3.4: Application of the One-Dimensional Laurence reduction to Cowan-Wilson dynamics --- #

# --- Parameters --- #
dt = 5e-1
stabtol = 1e-4
size = 5
p = 0.6
tau = 1
mu = 3
dim = 2
perturbations = 1
perturb = 1


def decay(x): return -x
def interact(x, y, _tau, _mu): return 1 / (1 + exp(_tau * (_mu - y))) - 1 / (1 + exp(_tau * _mu))


def gen_alpha_dependence():
    global perturb
    t0 = time.time()
    net = Network()
    net.gen_random(size, p)
    net.randomize_weights(lambda x: 2 + 5 * x)
    A = net.adj_matrix
    A = 
    results = []
    eigs = np.linalg.eigvals(A)
    eigs = sorted(eigs, reverse=True)
    alpha = eigs[0]
    i, t1 = 0, time.time()
    while alpha > 0.5:
        i += 1
        if i % 50 == 0:
            print(alpha, format_time_elapsed(time.time()-t0))

        eigs = np.linalg.eigvals(A)
        eigs = sorted(eigs, reverse=True, key=lambda h: np.linalg.norm(h))
        print(eigs)
        alpha = real(eigs[0])

        def reduc_interact(x, y): return interact(x, y, tau, mu)
        model = Model(W=A, F=decay, G=reduc_interact)
        x0 = np.random.random_sample(net.size) * net.size * 100
        simulation_res, prediction, a, alphas, betas = model.laurence_multi_dim_analysis_run(x0=x0, dim=dim, dt=dt,
                                                                                             stabtol=stabtol, debugging=0)
        simulation_res, prediction = real(simulation_res), real(prediction)
        results.append([alpha] + simulation_res.tolist() + prediction.tolist())

        # Perturb the adjacency matrix
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

    filename = 'data/3dot5_Nd_reduc_connect'
    results = np.array(results).T.tolist()

    with open(filename, "w") as f:
        json.dump(results, f)
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
              u'#bcbd22', u'#17becf']
    for i in range(dim):
        plt.plot(results[0], results[i+1], label=f"Sim {i+1}", color=colors[i], linestyle='None', marker='.')
        plt.plot(results[0], results[dim+i+1], label=f"Pred {i+1}", color=colors[i], linestyle='None', marker='.',
                 alpha=0.5)
    plt.legend()
    plt.xlabel("Alpha")
    plt.ylabel("R")
    plt.show()
    get_cached_performance()
    print(format_time_elapsed(time.time() - t0))


if __name__ == "__main__":
    gen_alpha_dependence()

