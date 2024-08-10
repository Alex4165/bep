import gc
import json
import time
from datetime import datetime
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
p1_fixed = [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10]
# NETWORK_TYPES = [
#                  ['gen_random', {"size": 51, "p": 0.1}, p1_fixed, np.linspace(9, 20, 10).tolist()],
#                  ['gen_random', {"size": 51, "p": 0.2}, p1_fixed, np.linspace(15, 30, 10).tolist()],
#                  ['gen_random', {"size": 51, "p": 0.3}, p1_fixed, np.linspace(25, 50, 10).tolist()],
#                  ['gen_random', {"size": 101, "p": 0.05}, p1_fixed, np.linspace(9, 20, 10).tolist()],
#                  ['gen_random', {"size": 101, "p": 0.1}, p1_fixed, np.linspace(25, 50, 10).tolist()],
#                  ['gen_random', {"size": 101, "p": 0.15}, p1_fixed, np.linspace(30, 35, 10).tolist()],
#                  ['gen_random', {"size": 151, "p": 1/30}, p1_fixed, np.linspace(25, 50, 10).tolist()],
#                  ['gen_random', {"size": 151, "p": 2/30}, p1_fixed, np.linspace(15, 30, 10).tolist()],
#                  ['gen_random', {"size": 151, "p": 0.1}, p1_fixed, np.linspace(25, 50, 10).tolist()],
# ]
NETWORK_TYPES = []
SIZES = [10,
         20, 30,
         40, 50, 60, 70, 80
         ]
for size in SIZES:
    for av_deg in [1, 5, 10, 15, 25]:
        if av_deg <= size-1:
            NETWORK_TYPES.append(['gen_random',
                                  {"size": size, "p": av_deg/(size-1)},
                                  p1_fixed,
                                  [0, 100]])
# NETWORK_TYPES = [['gen_random', {'size': 5, 'p': 0.4}, np.arange(1, 5, 1).tolist(), np.arange(0.1, 20)]]
RUNS_PER_TYPE = 1
DT = 1e-1
STABTOL = 1e-4
def WEIGHT_FUNCTION(x): return 5*x + 1


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
        print(f"Process {os.getpid()}"
              f"\tget_equilibrium {round(100*count/len(p1_range))}% completed"
              f"\taverage duration: {format_time_elapsed((time.time()-ti)/count)}"
              f"\texpected time left: {format_time_elapsed((time.time()-ti)/count*(len(p1_range)-count))}")
    return results


def get_critical_points(p1_range, p2_min, p2_max, netw: Network, tolerance=1e-1):
    """Currently assumes eq(p1, p2) is DECREASING in p2"""
    results = []
    metric = np.linalg.norm
    max_count = np.ceil(np.log2((p2_max-p2_min)/tolerance))+1  # +1 for good measure
    tupled_w = tuple(tuple(row) for row in netw.adj_matrix)
    ti = time.time()
    for p1 in p1_range:
        p2_low, p2_high = p2_min, p2_max
        count = 0

        def reduced_interact(x, y): return interact(x, y, p1, p2_min)
        def integrator(arr): return dx_dt(tuple(arr), decay, reduced_interact, tupled_w)
        x0 = 100 * netw.size * np.random.random_sample(netw.size)
        res = stability_run(integrator, DT, STABTOL, x0)
        if metric(res[0]) < 1e-2:  # We assume equilibrium at p2_min is 'large' else, we stop
            continue
        
        while (p2_high - p2_low) > tolerance and count <= max_count:
            count += 1
            p2_mid = (p2_low + p2_high) / 2

            def reduced_interact(x, y): return interact(x, y, p1, p2_mid)  # p1=tau, p2=mu
            def integrator(arr): return dx_dt(tuple(arr), decay, reduced_interact, tupled_w)
            x0 = 100 * netw.size * np.random.random_sample(netw.size)
            res = stability_run(integrator, DT, STABTOL, x0)
            results.append([p1, p2_mid, x0.tolist(), res[0].tolist()])

            if metric(res[0]) > 1e-2:
                p2_low = p2_mid
            else:
                p2_high = p2_mid
        if p2_high - p2_low > tolerance:
            print("DIDN'T ACHIEVE UNCERTAINTY WITHIN TOLERANCE")

        progress = p1_range.index(p1) + 1
        print(f"Process {os.getpid()}"
              f"\tget_equilibrium: {round(100*progress/len(p1_range))}%  "
              f"\tavg dur: {format_time_elapsed((time.time()-ti)/progress)}  "
              f"\texp time left: {format_time_elapsed((time.time()-ti)/progress*(len(p1_range)-progress))}  "
              f"\tnow: {datetime.now()}  ")

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


def get_critical_ls(p1_range, p2_min, p2_max, crit_val, tolerance=1e-2):
    p2_low, p2_high = p2_min, p2_max
    crit_p1s, crit_p2s = [], []
    max_count = np.ceil(np.log2((p2_high - p2_low) / tolerance)) + 1
    for p1 in p1_range:
        min_val = find_lambda_star((p1, p2_low))
        max_val = find_lambda_star((p1, p2_high))
        if crit_val < min_val or max_val < crit_val:
            continue
        count = 0
        while p2_high - p2_low > tolerance and count <= max_count:
            p2 = (p2_low + p2_high) / 2
            ls = find_lambda_star((p1, p2))
            if ls > crit_val:
                p2_high = p2
            else:
                p2_low = p2
            count += 1
        if p2_high - p2_low > tolerance:
            print("DIDN'T ACHIEVE UNCERTAINTY WITHIN TOLERANCE")
        crit_p1s.append(p1)
        crit_p2s.append(p2)
        p2_low, p2_high = p2_min, p2_max
    return crit_p1s, crit_p2s


def get_data_critical_eq(run_data):
    metric = np.linalg.norm
    data_dict = {}
    for i, p1 in enumerate(run_data[1][0]):
        if p1 not in data_dict:
            data_dict[p1] = [[], []]
        data_dict[p1][0].append(run_data[1][1][i])
        data_dict[p1][1].append(run_data[1][3][i])

    p1s = []
    p2s = []
    for p1, coll in data_dict.items():
        sorted_indices = sorted([k for k in range(len(coll[0]))], key=lambda k: coll[0][k])
        for m in range(len(sorted_indices)):
            i_here, i_prev = sorted_indices[m], sorted_indices[m-1]
            if metric(coll[1][i_here]) <= 1e-2:
                p1s.append(p1)
                best_guess = (coll[0][i_here]+coll[0][i_prev])/2
                p2s.append(best_guess)
                break
    return p1s, p2s


def make_network(network_generator: str, network_params: dict):
    netw = Network()
    generator = getattr(netw, network_generator)
    generator(**network_params)
    netw.randomize_weights(WEIGHT_FUNCTION)
    return netw


def run(index, run_number):
    network_type = NETWORK_TYPES[index][0]
    network_param = NETWORK_TYPES[index][1]
    p1_range = NETWORK_TYPES[index][2]
    p2_range = NETWORK_TYPES[index][3]
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
    # equilibria = get_equilibrium(p1_range, p2_range, net)  # Non optimized
    equilibria = get_critical_points(p1_range, min(p2_range), max(p2_range), net)
    # lstars = get_lambda_stars(P1_RANGE, P2_RANGE)  # This is a waste of time (each process does it)
    equilibria = list(map(list, zip(*equilibria)))
    # equilibria, lstars = np.array(equilibria).T.tolist(), np.array(lstars).T.tolist()

    with open(filename, "w") as grape_juice:
        json.dump([net.adj_matrix.tolist(), equilibria], grape_juice)
        
    print(f"finished {filename} in {format_time_elapsed(time.time()-t0)}")
    get_cached_performance()


if __name__ == "__main__":
    print("----- PROGRAM STARTED -----\n")
    generating_data = int(input("Would you like to generate data/plots? [1/0]: "))
    start = time.time()
    if generating_data == 1:
        print("CPU Count:", os.cpu_count())
        futs = []
        for i in range(len(NETWORK_TYPES)):
            for j in range(RUNS_PER_TYPE):
                futs.append(multiprocessing.Process(target=run, args=(i, j + 1)))
                futs[-1].start()

        for f in futs:
            f.join()
        
    elif generating_data == 0:
        filenames = []
        for i in range(len(NETWORK_TYPES)):
            for j in range(RUNS_PER_TYPE):
                network_t, network_p = NETWORK_TYPES[i][0], NETWORK_TYPES[i][1]
                p1_r, p2_r, run_num = NETWORK_TYPES[i][2], NETWORK_TYPES[i][3], j + 1
                filename = "data/3dot3_" + network_t
                for key, value in network_p.items():
                    filename += f"_{key}={value}"
                if run_num > 1:
                    filename += f"_{run_num}"
                filename += f"_[{min(p1_r)},{max(p1_r)}]x[{min(p2_r)},{max(p2_r)}]"

                filename += f"_DT={DT}_STABTOL={STABTOL}.txt"
                filenames.append(filename)

        print("Generating plots for:")
        for name in filenames:
            print(name)

        for filename in filenames:
            with open(filename, "r") as f:
                data = json.load(f)

            A = np.array(data[0])
            p1_r = np.unique(data[1][0])
            km, rh = kmax(A), rho(A)

            actual_p1s, actual_p2s = get_data_critical_eq(data)
            x_range = [0.1*(i+1) for i in range(10)] + [1+i*0.5 for i in range(10)] + [i for i in range(5, 11)]
            kmax_p1s, kmax_p2s = get_critical_ls(x_range, 0, 200, km)
            rho_p1s, rho_p2s = get_critical_ls(x_range, 0, 200, rh)

            plt.plot(actual_p1s, actual_p2s, 'k.', label="actual")
            plt.plot(kmax_p1s, kmax_p2s, color='#77DD77', alpha=0.5, label="kmax")
            plt.plot(rho_p1s, rho_p2s, color='red', alpha=0.5, label="rho")

            ymin, ymax = plt.gca().get_ylim()
            plt.fill_between(kmax_p1s, kmax_p2s, ymin, alpha=0.5, color='#77DD77')
            plt.fill_between(rho_p1s, rho_p2s, ymax, alpha=0.5, color="red")

            plt.xlabel("Tau")
            plt.ylabel("Mu")
            # plt.legend()
            plt.savefig(filename.replace(".txt", ".png"), dpi=300)
            plt.show()

            get_cached_performance()

            # lambda_stars = [find_lambda_star((p1, p2)) for (p1, p2) in zip(data[1][0], data[1][1])]
            # lower_bound_alpha = np.where(lambda_stars <= km, 1, 0)
            # upper_bound_alpha = np.where(lambda_stars >= rh, 1, 0)
            # plt.scatter(data[1][0], data[1][1], c='green', s=100, alpha=lower_bound_alpha)
            # plt.scatter(data[1][0], data[1][1], c='red', s=100, alpha=upper_bound_alpha)
            # plt.scatter(data[1][0], data[1][1], c=[np.linalg.norm(x) for x in data[1][3]], cmap="viridis")
            # plt.ylim(top=max(max(actual_p2s), max(rho_p2s), max(kmax_p2s))+0.1,
            #          bottom=min(min(actual_p2s), min(rho_p2s), min(kmax_p2s))-0.1)
            # plt.colorbar()
            # plt.show()
    
    print("\n----- PROGRAM FINISHED -----")
    print(f"Took: {format_time_elapsed(time.time()-start)}")





