import time
from concurrent.futures import ThreadPoolExecutor
from model import Model
from network_class import Network
from auxiliary_functions import plot_solution, get_cached_performance
import numpy as np
import os

# --- Section 4.1: What kind of solutions do we get from noncooperative networks? --- #

# Parameters
size = 20
average_degree = 8
dt = 0.005
stabtol = 1e-4
x0 = np.random.rand(size)
initial_factor = np.random.random_sample((size, size))
second_factor = np.random.random_sample((size, size)) - 0.5 * np.ones((size, size))


# Functions
def decay(x):
    return -x**3


def interact(x, y):
    return (1-x)*y


def run_comparisons(network):
    mat = np.multiply(network.adj_matrix, initial_factor)
    m = Model(mat, decay, interact)
    xs1 = m.stability_run(dt, stabtol, x0, debugging=0)[2]

    mat = np.multiply(network.adj_matrix, second_factor)
    m = Model(mat, decay, interact)
    xs2 = m.stability_run(dt, stabtol, x0, debugging=0)[2]

    return xs1, xs2


def test_er():
    """For an Erdos-Renyi network, compare the stability of the network with and without negative weights."""
    net = Network()
    net.gen_random(size, average_degree / (size-1))  # expected degree = p*(size-1)

    xs1, xs2 = run_comparisons(net)

    return net, xs1, xs2


def test_sw():
    """For a small-world network, compare the stability of the network with and without negative weights."""
    net = Network()
    net.gen_small_world(size, average_degree, 0)  # expected degree = k

    xs1, xs2 = run_comparisons(net)

    return net, xs1, xs2


def test_sf():
    """For a scale free network, compare the stability of the network with and without negative weights."""
    net = Network()
    theoretical_min = (2*size-1-np.sqrt((2*size-1)**2-4*size*average_degree))/2
    net.gen_hub(size, int(round(theoretical_min)))  # expected degree = 2*min deg - min deg/size - min deg^2/size

    xs1, xs2 = run_comparisons(net)

    return net, xs1, xs2


t0 = time.time()
print(f'Number of CPUs: {os.cpu_count()}')
with ThreadPoolExecutor() as executor:
    er_fut = executor.submit(test_er)
    sw_fut = executor.submit(test_sw)
    sf_fut = executor.submit(test_sf)

er_res = er_fut.result()
sw_res = sw_fut.result()
sf_res = sf_fut.result()

for n, t in zip([er_res[0], sw_res[0], sf_res[0]],
                ["Erdos-Renyi graph", "Small-world graph", "Scale-free graph"]):
    print(t, "had average degree =", sum([elmt[1] for elmt in n.graph.degree()]) / size)
    n.plot_graph(t, saving=True)

for xs, t in zip([er_res[1], er_res[2], sw_res[1], sw_res[2], sf_res[1], sf_res[2]],
                 ["Erdos-Renyi", "Reweighted Erdos-Renyi", "Small-world", "Reweighted Small-world",
                  "Scale-free", "Reweighted Scale-free"]):
    plot_solution(dt, t, xs, saving=True)

thread_pool_time = time.time()-t0
print(f'Time taken: {thread_pool_time:.2f} seconds')
get_cached_performance()

# t0 = time.time()
# er_res = test_er()
# sw_res = test_sw()
# sf_res = test_sf()
#
# for net, title in zip([er_res[0], sw_res[0], sf_res[0]],
#                       ["Erdos-Renyi", "Small-world", "Scale-free"]):
#     print(title, "had average degree =", sum([elmt[1] for elmt in net.graph.degree()])/size)
#     net.plot_graph(title)
#
# for xs, title in zip([er_res[1], er_res[2], sw_res[1], sw_res[2], sf_res[1], sf_res[2]],
#                      ["Erdos-Renyi", "Reweighted Erdos-Renyi", "Small-world", "Reweighted Small-world",
#                       "Scale-free", "Reweighted Scale-free"]):
#     plot_solution(dt, title, xs)
#
# linear_time = time.time()-t0
# print(f'Time taken: {linear_time:.2f} seconds')
# print(f'Efficiency gain: {linear_time/thread_pool_time:.2f} times')









