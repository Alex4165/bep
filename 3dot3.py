import time

import numpy as np
from matplotlib import pyplot as plt

from auxiliary_functions import exp, format_time_elapsed, find_lambda_star, get_cached_performance
from model import Model
from network_class import Network
from concurrent.futures import ThreadPoolExecutor

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


def get_equilibrium(p1_range, p2_range, network_generator: str, network_params: dict):
    # Initialize network
    net = Network()
    generator = getattr(net, network_generator)
    generator(**network_params)
    net.randomize_weights(lambda x: 5*x + 1)
    net.plot_graph()
    metric = np.linalg.norm

    results = []
    futures = []
    with ThreadPoolExecutor() as exc:
        for p1 in p1_range:
            for p2 in p2_range:
                model = Model(net.adj_matrix, F=lambda x: decay(x), G=lambda x, y: interact(x, y, p1, p2))
                futures.append((p1, p2, exc.submit(
                    lambda: model.stability_run(dt, stabtol, 10*net.size*np.random.rand(net.size))[0])))
    for p1, p2, future in futures:
        results.append([p1, p2, metric(future.result())])
    return results


def get_lambda_stars(p1_range, p2_range):
    """Calls find_lambda_star a fixed amount of time, a lot of which is wasted"""
    results = []
    futures = []
    with ThreadPoolExecutor() as executor:
        for p1 in p1_range:
            for p2 in p2_range:
                futures.append((p1, p2, executor.submit(
                    lambda: find_lambda_star((p1, p2)))))
    for p1, p2, future in futures:
        results.append([p1, p2, future.result()])
    return results


if __name__ == "__main__":
    p1s = np.linspace(0.1, 10, 10).tolist()
    p2s = np.linspace(0.1,  30, 10).tolist()

    n, t0 = len(p1s)*len(p2s), time.time()

    equilibria = get_equilibrium(p1s, p2s, "gen_random", {"size": 21, "p": 0.2})
    lstars = get_lambda_stars(p1s, p2s)
    equilibria, lstars = np.array(equilibria).T, np.array(lstars).T

    print(f"Done after {format_time_elapsed(time.time()-t0)}")
    print(f"So {format_time_elapsed((time.time()-t0)/n)} per point\n")

    plt.scatter(lstars[0], lstars[1], c=lstars[2], marker='s', cmap='viridis')
    plt.colorbar(label="Lambda star norm")
    plt.xlabel("Tau")
    plt.ylabel("Mu")
    plt.show()

    plt.scatter(equilibria[0], equilibria[1], c=equilibria[2], marker='s', cmap='viridis')
    plt.colorbar(label="Eq. norm")
    plt.xlabel("Tau")
    plt.ylabel("Mu")
    plt.show()

    get_cached_performance()






