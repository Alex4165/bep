from model import Model, stability_run
from auxiliary_functions import RK4step, dx_dt, exp, plot_solution
from data_eater import reduce

import numpy as np
import matplotlib.pyplot as plt

from network_class import Network

SIZE = 2
TAU = 1.3
MU = 4
DT = 1e-1
STABTOL = 1e-4
ALPHAS = np.linspace(0.1, 5, 1)
P = [1.2, 0]
# P = np.zeros(SIZE)
# P[0] = 10*np.random.random()


def Z(x):
    num = 1 / (1 + exp(TAU * (MU - x))) - 1 / (1 + exp(TAU * MU))
    den = 1 - 1 / (1 + exp(TAU * MU))
    return num / den


net = Network()
error = 1e-2+1e-18
while error > 1e-2:
    net.gen_random(size=SIZE, p=0.3)
    # net.gen_hub(size=SIZE, minimum_deg=int(10*P))
    # net.gen_small_world(size=SIZE, k=int(10*P)+1, p=0.4)
    net.randomize_weights(lambda x: 50 * x - 25)
    net.plot_graph()
    A_zero = net.adj_matrix
    a, alpha_zero, beta = reduce(A_zero)
    b_zero = np.linalg.lstsq(A_zero, np.ones(SIZE), rcond=None)[0]
    error = np.linalg.norm(A_zero @ b_zero - np.ones(SIZE))
    error = 0
print(error)

sims, preds, nonlinearity_errors = [], [], []
for alpha_ in ALPHAS:
    # A = alpha * A_zero / alpha_zero
    A = np.array([[16, -12],
                 [15, -0.1]])
    a, alpha, _ = reduce(A)
    gamma = a @ P
    # b = np.linalg.lstsq(A, np.ones(SIZE), rcond=None)[0]
    # b = alpha_zero * b_zero / alpha
    # error = np.linalg.norm(A @ b - np.ones(SIZE))


    def model_f(arr):
        k = A @ arr
        return -arr + [Z(k[i] + P[i]) for i in range(len(arr))]


    def red_f(arr):
        return -arr[0] + Z(alpha * arr[0] + gamma)


    x0 = 0.1*np.ones(SIZE)
    sim = stability_run(model_f, DT, None, x0, debugging=2, title='Simulation', max_run_time=41)
    pred = stability_run(red_f, DT, None, [a @ x0], debugging=2, title='Prediction', max_run_time=41)

    # for x in sim[2]:
    #     total = 0
    #     for i, c in enumerate(x):
    #         total += a[i] * Z()

    sims.append(a @ sim[0])
    preds.append(pred[0][0])
    print("Simulation:", a @ sim[0], "Prediction:", pred[0][0])
    nonlinearity_errors.append(error)

plt.plot(ALPHAS, sims, '.', label='Simulation')
plt.plot(ALPHAS, preds, '.', label='Prediction')
plt.legend()
plt.show()

plt.plot(ALPHAS, [abs(s-p) for s, p in zip(sims, preds)], '.', label='Absolute error')
plt.plot(ALPHAS, np.array(nonlinearity_errors), '.', label='Nonlinearity error')
plt.legend()
plt.show()
