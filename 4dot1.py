from model import Model
from network_class import Network
from auxiliary_functions import exp, plot_solution
import numpy as np

# --- Section 4.1: What kind of solutions do we get from noncooperative networks? --- #

# Parameters
size = 20
exp_degree = 5
dt = 0.005
stabtol = 1e-4
runs = 2  # how many times to reweight, run, and plot and save the network
x0 = np.random.rand(size)
pos_factors = [2*np.random.random_sample((size, size)) for i in range(runs)]
neg_factors = [4*np.random.random_sample((size, size)) - 2*np.ones((size, size)) for i in range(runs)]


# Dynamic equations
def decay(x):
    return -x**3


def interact(x, y):
    return (1-x)*y


# To save the solution data
before_xs = []
after_xs = []

# Define the Erdos-Renyi graph
net = Network()
net.gen_random(size, exp_degree / (size - 1))

# Positive reweighting and running the network
for i in range(runs):
    mat = np.multiply(net.adj_matrix, pos_factors[i])
    m = Model(mat, decay, interact)
    before_xs.append(m.stability_run(dt, stabtol, x0, debugging=0)[2])

# Not-necessarily-positive reweighting and running the network
for i in range(runs):
    mat = np.multiply(net.adj_matrix, neg_factors[i])
    m = Model(mat, decay, interact)
    after_xs.append(m.stability_run(dt, stabtol, x0, debugging=0)[2])

# Plot and save the results
net.plot_graph("Erdos-Renyi Graph", saving=True)
for i in range(runs):
    plot_solution(dt, f"Erdos-Renyi {i}", before_xs[i], saving=True)
    plot_solution(dt, f"Erdos-Renyi Reweighted {i}", after_xs[i], saving=True)


