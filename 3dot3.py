import gc
import json
import time
import os

import numpy as np
from matplotlib import pyplot as plt

from auxiliary_functions import (exp, format_time_elapsed, find_lambda_star, get_cached_performance, plot_solution,
                                 dx_dt, rho, kmax)
from model import stability_run
from network_class import Network

import multiprocessing

# --- Section 3.3: Wu Rigorous bounds applied --- #

# Parameters
p1_fixed = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 2, 5, 10]
NETWORK_TYPES = [
                 #['gen_random', {"size": 51, "p": 0.1}, p1_fixed, np.linspace(11, 12, 10).tolist()],
                 #['gen_random', {"size": 51, "p": 0.2}, p1_fixed, np.linspace(21, 23, 10).tolist()],
                 #['gen_random', {"size": 51, "p": 0.3}, p1_fixed, np.linspace(30, 35, 10).tolist()],
                 #['gen_random', {"size": 101, "p": 0.05}, p1_fixed, np.linspace(12, 13, 10).tolist()],
                 ['gen_random', {"size": 101, "p": 0.1}, p1_fixed, np.linspace(22, 23, 10).tolist()],
                 #['gen_random', {"size": 101, "p": 0.15}, p1_fixed, np.linspace(30, 35, 10).tolist()],
                 ['gen_random', {"size": 151, "p": 1/30}, p1_fixed, np.linspace(11, 12, 10).tolist()],
                 ['gen_random', {"size": 151, "p": 2/30}, p1_fixed, np.linspace(21, 22, 10).tolist()],
                 ['gen_random', {"size": 151, "p": 0.1}, p1_fixed, np.linspace(30, 35, 10).tolist()]]
# NETWORK_TYPES = [['gen_random', {'size': 5, 'p': 0.4}, np.arange(1, 3, 1).tolist(), np.arange(1, 3, 1).tolist()]]
RUNS_PER_TYPE = 1
DT = 1e-1
STABTOL = 1e-4
def WEIGHT_FUNCTION(x): return 5 * x + 1


# Dynamic equations (we assume all model have the same dynamic equations)
def decay(x):
    return -x


def interact(x, y, tau, mu):
    return 1 / (1 + exp(tau * (mu - y))) - 1 / (1 + exp(tau * mu))


def get_equilibrium(p1_range, p2_range, netw: Network):
    results = []
    metric = np.linalg.norm
    tupled_w = tuple(tuple(row) for row in netw.adj_matrix)

    ti, count = time.time(), 0
    for p1 in p1_range:
        for p2 in p2_range:
            def reduced_interact(x, y): return interact(x, y, p1, p2)  # p1=tau, p2=mu
            def integrator(arr): return dx_dt(tuple(arr), decay, reduced_interact, tupled_w)
            x0 = 100 * netw.size * np.random.random_sample(netw.size)
            res = stability_run(integrator, DT, STABTOL, x0)
            results.append([p1, p2, x0.tolist(), res[0].tolist()])

        count += 1
        print(f"Process {os.getpid()} | get_equilibrium {round(100*count/len(p1_range))}% completed | "
              f"average duration: {format_time_elapsed((time.time()-ti)/count)} | "
              f"expected time left: {format_time_elapsed((time.time()-ti)/count*(len(p1_range)-count))}")
    return results


def get_lambda_stars(p1_range, p2_range):
    """Calls find_lambda_star a fixed amount of time, a lot of which is wasted"""
    ti, count = time.time(), 0
    results = []
    for p1 in p1_range:
        for p2 in p2_range:
            results.append([p1, p2, find_lambda_star((p1, p2))])
        count += 1
        print(f"Process {os.getpid()} | get_lambda_stars {round(100*count/len(p1_range))}% completed | "
              f"average duration: {format_time_elapsed((time.time()-ti)/count)} | "
              f"expected time left: {format_time_elapsed((time.time()-ti)/count*(len(p1_range)-count))}")
    return results


def make_network(network_generator: str, network_params: dict):
    netw = Network()
    generator = getattr(netw, network_generator)
    generator(**network_params)
    netw.randomize_weights(WEIGHT_FUNCTION)
    return netw


def run(i, run_number):
    network_type = NETWORK_TYPES[i][0]
    network_param = NETWORK_TYPES[i][1]
    p1_range = NETWORK_TYPES[i][2]
    p2_range = NETWORK_TYPES[i][3]
    filename = "data/3dot3_" + network_type
    for key, value in network_param.items():
        filename += f"_{key}={value}"
    if run_number > 1:
        filename += f"_{run_number}"
    filename += f"_[{min(p1_range)},{max(p1_range)}]x[{min(p2_range)},{max(p2_range)}]"

    filename += f"_DT={DT}_STABTOL={STABTOL}.txt"
    
    print("starting " + filename + " on Process " + str(os.getpid()))
    t0 = time.time()
          
    net = make_network(network_type, network_param)
    equilibria = get_equilibrium(p1_range, p2_range, net)
    # lstars = get_lambda_stars(P1_RANGE, P2_RANGE)  # This is a waste of time (each process does it)
    equilibria = list(map(list, zip(*equilibria)))
    # equilibria, lstars = np.array(equilibria).T.tolist(), np.array(lstars).T.tolist()

    with open(filename, "w") as f:
        json.dump([net.adj_matrix.tolist(), equilibria], f)
        
    print(f"finished {filename} in {format_time_elapsed(time.time()-t0)}")
    get_cached_performance()


if __name__ == "__main__":
    print("----- PROGRAM STARTED -----\n")
    generating_data = int(input("Would you like to generate data/plots? [1/0]: "))
    start = time.time()
    if generating_data:
        print("CPU Count:", os.cpu_count())
        futs = []
        for i in range(len(NETWORK_TYPES)):
            for j in range(RUNS_PER_TYPE):
                futs.append(multiprocessing.Process(target=run, args=(i, j + 1)))
                futs[-1].start()

        for f in futs:
            f.join()
        
    else:
        filenames = []
        for i in range(len(NETWORK_TYPES)):
            for j in range(RUNS_PER_TYPE):
                network_type, network_param, p1_range, p2_range, run_number = NETWORK_TYPES[i][0], NETWORK_TYPES[i][1], NETWORK_TYPES[i][2], NETWORK_TYPES[i][3], j + 1
                filename = "data/3dot3_" + network_type
                for key, value in network_param.items():
                    filename += f"_{key}={value}"
                if run_number > 1:
                    filename += f"_{run_number}"
                filename += f"_[{min(p1_range)},{max(p1_range)}]x[{min(p2_range)},{max(p2_range)}]"

                filename += f"_DT={DT}_STABTOL={STABTOL}.txt"
                filenames.append(filename)
                
        for filename in filenames:
            with open(filename, "r") as f:
                data = json.load(f)  
                #kmax, rho = kmax(np.array(data[0])), rho(np.array(data[0]))
                #lower_bound_alpha = np.where(data[1][2] <= kmax, 1, 0)
                #upper_bound_alpha = np.where(data[1][2] >= rho, 1, 0)

                #plt.scatter(data[2][0], data[2][1], c='green', alpha=lower_bound_alpha, s=100)
                #plt.scatter(data[2][0], data[2][1], c='red', alpha=upper_bound_alpha, s=100)
                plt.scatter(data[1][0], data[1][1], c=[np.linalg.norm(p) for p in data[1][3]], cmap='viridis', s=25)
                plt.colorbar(label='Eq. norm')
                plt.xlabel("Tau")
                plt.ylabel("Mu")
                title = filename.replace("_", " ").replace("data/3dot3 ", "").replace(".txt", "")
                plt.title(title)
                plt.show()
    
    print("\n----- PROGRAM FINISHED -----")
    print(f"Took: {format_time_elapsed(time.time()-start)}")






