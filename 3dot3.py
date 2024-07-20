import json
import time

import networkx
import numpy as np
from matplotlib import pyplot as plt

from auxiliary_functions import exp, format_time_elapsed, find_lambda_star, get_cached_performance, plot_solution, dx_dt
from model import Model, stability_run
from network_class import Network

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

# --- Section 3.3: Wu Rigorous bounds applied --- #

# Parameters
sizes = [5, 10, 15]
network_types = [Network.gen_random]
runs_per_size = 5
runs_per_network = 5
dt = 1e-1
stabtol = 1e-4


# Dynamic equations (we assume all model have the same dynamic equations)
def decay(x):
    return -x


def interact(x, y, tau, mu):
    return 1 / (1 + exp(tau*(mu - y))) - 1 / (1 + exp(tau*mu))


def helper_func(net, p1, p2):
    interactor = partial(interact, tau=p1, mu=p2)
    func = lambda arr: dx_dt(tuple(arr), decay, interactor,
                             tuple(tuple(row) for row in net.adj_matrix))
    res = stability_run(func, dt, stabtol, 10*net.size*np.random.rand(net.size))
    return res[0]


def partial_equilibrium(net, p1_range, p2_range, metric):
    results = []
    for p1 in p1_range:
        for p2 in p2_range:
            results.append([p1, p2, metric(helper_func(net, p1, p2))])
    return results


def get_equilibrium(p1_range, p2_range, net):
    results = []
    metric = np.linalg.norm

    if optimized:
        cpu_count = os.cpu_count()
        param_batch_size = round(np.ceil(np.sqrt(cpu_count)))

        # Note assumes that p1_range and p2_range are around the same length
        p1_parts, p2_parts = np.array_split(p1_range, param_batch_size), np.array_split(p2_range, param_batch_size)
        futures = []
        results = []
        with ThreadPoolExecutor() as exc:
            for p1_r in p1_parts:
                for p2_r in p2_parts:
                    task = partial(partial_equilibrium, net, p1_r, p2_r, metric)
                    futures.append(exc.submit(task))

        print(f"spawned {len(futures)} threads")
        for future in futures:
            for res in future.result():
                results.append(res)
    else:
        for p1 in p1_range:
            for p2 in p2_range:
                model = Model(net.adj_matrix, F=lambda x: decay(x), G=lambda x, y: interact(x, y, p1, p2))
                res = model.stability_run(dt, stabtol, 10*net.size*np.random.rand(net.size))
                results.append([p1, p2, metric(res[0])])

    return results


def partial_lambda_star(p1_range, p2_range):
    results = []
    for p1 in p1_range:
        for p2 in p2_range:
            results.append([p1, p2, find_lambda_star((p1, p2))])
    return results


def get_lambda_stars(p1_range, p2_range):
    """Calls find_lambda_star a fixed amount of time, a lot of which is wasted"""
    results = []
    futures = []

    if optimized:
        cpu_count = os.cpu_count()
        param_batch_size = round(np.ceil(np.sqrt(cpu_count)))

        # Note assumes that p1_range and p2_range are around the same length
        p1_parts, p2_parts = np.array_split(p1_range, param_batch_size), np.array_split(p2_range, param_batch_size)

        with ThreadPoolExecutor() as exc:
            for p1_r in p1_parts:
                for p2_r in p2_parts:
                    task = partial(partial_lambda_star, p1_r, p2_r)
                    futures.append(exc.submit(task))

        print(f"spawned {len(futures)} threads")
        for future in futures:
            for res in future.result():
                results.append(res)
    else:
        for p1 in p1_range:
            for p2 in p2_range:
                results.append([p1, p2, find_lambda_star((p1, p2))])
    return results


def make_network(network_generator: str, network_params: dict):
    netw = Network()
    generator = getattr(netw, network_generator)
    generator(**network_params)
    netw.randomize_weights(lambda x: 5*x + 1)
    netw.plot_graph()
    return netw


if __name__ == "__main__":
    for i in range(1):
        i = 1
        optimized = [True, False][i]
        print(optimized)
        p1s = np.linspace(0.1, 5.1, 5).tolist()
        p2s = np.linspace(7.5,  12, 5).tolist()

        n, t0 = len(p1s)*len(p2s), time.time()

        net = make_network("gen_random", {"size": 21, "p": 0.2})
        equilibria = get_equilibrium(p1s, p2s, net)
        lstars = get_lambda_stars(p1s, p2s)
        equilibria, lstars = np.array(equilibria).T.tolist(), np.array(lstars).T.tolist()
        with open(f"data/3dot3{['_opt', '_nonopt'][i]}.txt", "w") as f:
            f.write(str([net.adj_matrix, equilibria, lstars]))

        print(f"Done after {format_time_elapsed(time.time()-t0)}")
        print(f"So {format_time_elapsed((time.time()-t0)/n)} per point\n")

        plt.scatter(lstars[0], lstars[1], c=lstars[2], marker='s', s=25, cmap='viridis')
        plt.colorbar(label="Lambda star norm")
        plt.xlabel("Tau")
        plt.ylabel("Mu")
        plt.show()

        plt.scatter(equilibria[0], equilibria[1], c=equilibria[2], marker='s', s=25, cmap='viridis')
        plt.colorbar(label="Eq. norm")
        plt.xlabel("Tau")
        plt.ylabel("Mu")
        plt.show()

        get_cached_performance()






