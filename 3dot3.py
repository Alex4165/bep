import gc
import json
import time

import networkx
import numpy as np
from matplotlib import pyplot as plt

from auxiliary_functions import (exp, format_time_elapsed, find_lambda_star, get_cached_performance, plot_solution,
                                 dx_dt, rho, kmax)
from model import Model, stability_run
from network_class import Network

import multiprocessing

# --- Section 3.3: Wu Rigorous bounds applied --- #

# Parameters
network_types = [['gen_random', {"size": 50, "p": 0.102}],
                 ['gen_random', {"size": 50, "p": 0.204}],
                 ['gen_random', {"size": 50, "p": 0.306}],
                 ['gen_random', {"size": 100, "p": 0.102}],
                 ['gen_random', {"size": 100, "p": 0.204}],
                 ['gen_random', {"size": 100, "p": 0.306}],
                 ['gen_hub', {"size": 50, "minimum_deg": 2}],
                 ['gen_hub', {"size": 50, "minimum_deg": 5}],
                 ['gen_hub', {"size": 50, "minimum_deg": 8}],
                 ['gen_hub', {"size": 100, "minimum_deg": 5}],
                 ['gen_hub', {"size": 100, "minimum_deg": 10}],
                 ['gen_hub', {"size": 100, "minimum_deg": 15}],]
runs_per_type = 2
dt = 5e-2
stabtol = 1e-4


# Dynamic equations (we assume all model have the same dynamic equations)
def decay(x):
    return -x


def interact(x, y, tau, mu):
    return 1 / (1 + exp(tau * (mu - y))) - 1 / (1 + exp(tau * mu))


def get_equilibrium(p1_range, p2_range, netw: Network):
    results = []
    metric = np.linalg.norm
    tupled_w = tuple(tuple(row) for row in netw.adj_matrix)

    # ti = time.time()
    for p1 in p1_range:
        for p2 in p2_range:
            def reduced_interact(x, y): return interact(x, y, p1, p2)
            def integrator(arr): return dx_dt(tuple(arr), decay, reduced_interact, tupled_w)
            res = stability_run(integrator, dt, stabtol, 100*netw.size * np.random.random_sample(netw.size))
            results.append([p1, p2, metric(res[0])])
        gc.collect()

        # i = p1_range.index(p1)
        # print(f"{round((i+1) / len(p1_range) * 100)}% equilibrium completed, "
        #       f"estimated {format_time_elapsed((time.time()-ti)/(i+1)*(len(p1_range)-i-1))} left")

    return results


def get_lambda_stars(p1_range, p2_range):
    """Calls find_lambda_star a fixed amount of time, a lot of which is wasted"""
    # ti = time.time()
    results = []
    for p1 in p1_range:
        for p2 in p2_range:
            results.append([p1, p2, find_lambda_star((p1, p2))])

        # i = p1_range.index(p1)
        # print(f"{round((i+1) / len(p1_range) * 100)}% lambda star completed, "
        #       f"estimated {format_time_elapsed((time.time()-ti)/(i+1)*(len(p1_range)-i-1))} left")
    return results


def make_network(network_generator: str, network_params: dict):
    netw = Network()
    generator = getattr(netw, network_generator)
    generator(**network_params)
    netw.randomize_weights(lambda x: 5 * x + 1)
    # netw.plot_graph()
    return netw


def run(network_type, network_param, run_number):
    net = make_network(network_type, network_param)
    p1s = np.linspace(0.1, 5.1, 10).tolist()
    p2s = np.linspace(7.5, 12.5, 10).tolist()

    equilibria = get_equilibrium(p1s, p2s, net)
    lstars = get_lambda_stars(p1s, p2s)
    equilibria, lstars = np.array(equilibria).T.tolist(), np.array(lstars).T.tolist()

    filename = "data/3dot3_" + network_type
    for key, value in network_param.items():
        filename += f"_{key}={value}"
    filename += f"_{run_number}.txt"
    with open(filename, "w") as f:
        f.write(str(["[explanation, net.adj_matrix, equilibria, lstars]", net.adj_matrix, equilibria, lstars]))
    print("finished " + filename)


if __name__ == "__main__":
    opt = True
    if opt:
        futs = []
        start = time.time()
        for i in range(len(network_types)):
            for j in range(runs_per_type):
                futs.append(multiprocessing.Process(target=run, args=(network_types[i][0], network_types[i][1], j+1)))
                futs[-1].start()

        for f in futs:
            f.join()

        print(format_time_elapsed(time.time()-start))
    else:
        start = time.time()
        for i in range(len(network_types)):
            for j in range(runs_per_type):
                run(network_types[i][0], network_types[i][1], j+1)
        print(format_time_elapsed(time.time()-start))


    get_cached_performance()



