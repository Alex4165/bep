import json
import time

import numpy as np
from matplotlib import pyplot as plt

from model import stability_run, Model
from auxiliary_functions import dx_dt, exp, rootfinder, format_time_elapsed, get_cached_performance, real
from network_class import Network
from data_eater import one_laurence, reduce

# --- Section 3.5: Application of the Multi-Dimensional Laurence reduction to Cowan-Wilson dynamics --- #

# --- Parameters --- #
DT = 5e-1
STABTOL = 1e-3
SIZE = 12
P = 0.2
TAU = 1
MU = 3
DIM = 2
ALPHAS = np.linspace(1, 25, 50)


def decay(x): return -x
def interact(x, y, _tau, _mu): return 1 / (1 + exp(_tau * (_mu - y))) - 1 / (1 + exp(_tau * _mu))
def red_interact(x, y): return 1 / (1 + exp(TAU * (MU - y)))


def gen_alpha_dependence():
    t0 = time.time()

    net = Network()
    # net.gen_random(SIZE, P)
    pop1, pop2 = int(SIZE/3), int(2*SIZE/3)
    p = 2/(pop1 * pop2)
    net.gen_community([pop1, pop2], [[1, p], [p, 1]])
    net.randomize_weights(lambda x: 2 + 5 * x)
    net.plot_graph()
    A = net.adj_matrix

    model = Model(W=A, F=decay, G=red_interact)
    alphas_ = model.laurence_multi_dim_analysis_run(x0=np.random.random_sample(SIZE), dim=DIM, max_run_time=0)[3]
    alpha_zero = np.mean(alphas_)

    results = []
    for alpha in ALPHAS:
        w = alpha/alpha_zero * A

        model = Model(W=w, F=decay, G=red_interact)
        for x0 in [np.random.random_sample(net.size) * net.size * 100, np.random.random_sample(net.size)]:
            simulation_res, prediction, a, alphas, betas = model.laurence_multi_dim_analysis_run(x0=x0, dim=DIM, dt=DT,
                                                                                                 stabtol=STABTOL,
                                                                                                 debugging=0)

            if np.mean(alphas) - alpha > 0.1:
                print(np.mean(alphas), alpha)
                model.laurence_multi_dim_analysis_run(x0=x0, dim=DIM, dt=DT, stabtol=STABTOL)
            simulation_res, prediction = real(simulation_res), real(prediction)
            results.append([alpha] + simulation_res.tolist() + prediction.tolist())

    filename = 'data/3dot5_Nd_reduc_connect'
    results = np.array(results).T.tolist()

    with open(filename, "w") as f:
        json.dump(results, f)
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
              u'#bcbd22', u'#17becf']
    for i in range(DIM):
        plt.plot(results[0], results[DIM+i+1], label=f"Pred {i+1}", color=colors[i], linestyle='None',
                 marker='s', alpha=0.5)
        plt.plot(results[0], results[i+1], label=f"Sim {i+1}", color=colors[i], linestyle='None',
                 marker='o', alpha=0.5)
    plt.legend()
    plt.xlabel("Alpha")
    plt.ylabel("R")
    plt.show()
    get_cached_performance()
    print(format_time_elapsed(time.time() - t0))


if __name__ == "__main__":
    gen_alpha_dependence()

