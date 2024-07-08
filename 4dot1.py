from model import Model, Network, plot_solution
import numpy as np

# --- Section 4.1: What kind of solutions do we get from noncooperative networks? --- #

# Parameters
size = 20
dt = 0.005
stabtol = 1e-4
x0 = np.random.rand(size)
pos_factors = [np.random.random_sample((size, size)) for i in range(5)]
neg_factors = [np.random.random_sample((size, size)) - 0.5 * np.ones((size, size)) for i in range(5)]


# Dynamic equations
def decay(x):
    return -x**3


def interact(x, y):
    return (1-x)*y


before_xs = []
after_xs = []

net = Network()
net.gen_random(size, 0.5)

# Positive reweighting
mat = np.multiply(net.adj_matrix, pos_factors[0])
m = Model(mat, decay, interact)
before_xs.append(m.stability_run(dt, stabtol, x0, debugging=0)[2])

mat = np.multiply(net.adj_matrix, pos_factors[1])
m = Model(mat, decay, interact)
before_xs.append(m.stability_run(dt, stabtol, x0, debugging=0)[2])

# Positive and negative reweighting
mat = np.multiply(net.adj_matrix, neg_factors[0])
m = Model(mat, decay, interact)
after_xs.append(m.stability_run(dt, stabtol, x0, debugging=0)[2])

mat = np.multiply(net.adj_matrix, neg_factors[1])
m = Model(mat, decay, interact)
after_xs.append(m.stability_run(dt, stabtol, x0, debugging=0)[2])

net.plot_graph("Erdos-Renyi Graph", saving=True)
plot_solution(dt, "Erdos-Renyi 1", before_xs[0], saving=True)
plot_solution(dt, "Erdos-Renyi 2", before_xs[1], saving=True)
plot_solution(dt, "Erdos-Renyi Reweighted 1", after_xs[0], saving=True)
plot_solution(dt, "Erdos-Renyi Reweighted 2", after_xs[1], saving=True)


