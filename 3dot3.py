import gc
import json
import time
import os

import networkx
import numpy as np
from matplotlib import pyplot as plt

from auxiliary_functions import (exp, format_time_elapsed, find_lambda_star, get_cached_performance, plot_solution,
                                 dx_dt, rho, kmax)
from model import stability_run
from network_class import Network

import multiprocessing

# --- Section 3.3: Wu Rigorous bounds applied --- #

# Parameters
network_types = [['gen_random', {"size": 51, "p": 0.1}],
                 ['gen_random', {"size": 51, "p": 0.2}],
                 ['gen_random', {"size": 51, "p": 0.3}],
                 ['gen_random', {"size": 101, "p": 0.05}],
                 ['gen_random', {"size": 101, "p": 0.1}],
                 ['gen_random', {"size": 101, "p": 0.15}],
                 ['gen_random', {"size": 151, "p": 1/30}],
                 ['gen_random', {"size": 151, "p": 2/30}],
                 ['gen_random', {"size": 151, "p": 0.1}],]
# network_types = [['gen_random', {'size': 5, 'p': 0.4}]]
p1s = np.arange(0.1, 10.1, 0.25).tolist()
p2s = np.arange(0, 15, 0.25).tolist()
runs_per_type = 1
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

    for p1 in p1_range:
        for p2 in p2_range:
            def reduced_interact(x, y): return interact(x, y, p1, p2)  # p1=tau, p2=mu
            def integrator(arr): return dx_dt(tuple(arr), decay, reduced_interact, tupled_w)
            res = stability_run(integrator, dt, stabtol, 100*netw.size * np.random.random_sample(netw.size))
            results.append([p1, p2, metric(res[0])])
        gc.collect()
    return results


def get_lambda_stars(p1_range, p2_range):
    """Calls find_lambda_star a fixed amount of time, a lot of which is wasted"""
    # ti = time.time()
    results = []
    for p1 in p1_range:
        for p2 in p2_range:
            results.append([p1, p2, find_lambda_star((p1, p2))])
    return results


def make_network(network_generator: str, network_params: dict):
    netw = Network()
    generator = getattr(netw, network_generator)
    generator(**network_params)
    netw.randomize_weights(lambda x: 5 * x + 1)
    return netw


def run(network_type, network_param, run_number):
    filename = "data/3dot3_" + network_type
    for key, value in network_param.items():
        filename += f"_{key}={value}"
    if run_number > 1:
        filename += f"_{run_number}.txt"
    
    print("starting " + filename)
    t0 = time.time()
          
    net = make_network(network_type, network_param)
    equilibria = get_equilibrium(p1s, p2s, net)
    lstars = get_lambda_stars(p1s, p2s)
    equilibria, lstars = np.array(equilibria).T.tolist(), np.array(lstars).T.tolist()

    with open(filename, "w") as f:
        json.dump([net.adj_matrix.tolist(), equilibria, lstars], f)
        
    print(f"finished {filename} in {format_time_elapsed(time.time()-t0)}")
    get_cached_performance()


if __name__ == "__main__":
    print("----- I AM RUNNING -----\n")
    print("CPU Count:", os.cpu_count())
    generating_data = True
    if generating_data:
        futs = []
        start = time.time()
        for i in range(len(network_types)):
            for j in range(runs_per_type):
                futs.append(multiprocessing.Process(target=run, args=(network_types[i][0], network_types[i][1], j+1)))
                futs[-1].start()

        for f in futs:
            f.join()

        print("\n----- FINISHED RUNNING -----")
        print(f"Took: {format_time_elapsed(time.time()-start)}")
        
    else:
        filenames = ["data/3dot3_gen_random_size=50_p=0.102_1.txt",
                     "data/3dot3_gen_random_size=50_p=0.102_2.txt",
                     "data/3dot3_gen_random_size=50_p=0.204_1.txt",
                     "data/3dot3_gen_random_size=50_p=0.204_2.txt",
                     "data/3dot3_gen_random_size=50_p=0.306_1.txt",
                     "data/3dot3_gen_random_size=50_p=0.306_2.txt",
                     "data/3dot3_gen_random_size=100_p=0.102_1.txt",
                     "data/3dot3_gen_random_size=100_p=0.102_2.txt",
                     "data/3dot3_gen_random_size=100_p=0.204_1.txt",
                     "data/3dot3_gen_random_size=100_p=0.204_2.txt",
                     "data/3dot3_gen_random_size=100_p=0.306_1.txt",
                     "data/3dot3_gen_random_size=100_p=0.306_2.txt",]
        for filename in filenames:
            with open(filename, "r") as f:
                data = json.load(f)  # [adjacency matrix, [[p1], [p2], [eq]], [[p1], [p2], [lambda]]
                kmax, rho = kmax(np.array(data[0])), rho(np.array(data[0]))
                lower_bound_alpha = np.where(data[2][2] <= kmax, 1, 0)
                upper_bound_alpha = np.where(data[2][2] >= rho, 1, 0)

                plt.scatter(data[2][0], data[2][1], c='green', alpha=lower_bound_alpha, s=100)
                plt.scatter(data[2][0], data[2][1], c='red', alpha=upper_bound_alpha, s=100)
                plt.scatter(data[1][0], data[1][1], c=data[1][2], cmap='viridis', s=25)
                plt.colorbar(label='Eq. norm')
                plt.xlabel("Tau")
                plt.ylabel("Mu")
                plt.show()






