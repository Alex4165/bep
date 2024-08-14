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
P = np.array([1, 0])
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
    A = np.array([[105, -100],
                 [8, -1]])
    print("gamma - P =", A @ [0.01, -0.015])
    # a, alpha, _ = reduce(A)
    eigs, vecs = np.linalg.eig(A)
    vecs = vecs.T
    which_one = 0
    alpha = eigs[which_one]
    a = vecs[which_one] / sum(vecs[which_one])
    gamma = a @ P
    b = np.linalg.lstsq(A, alpha*np.ones(SIZE), rcond=None)[0]
    c = np.linalg.lstsq(A, gamma*np.ones(SIZE)-P, rcond=None)[0]
    print(b, c)
    print("Inv. error b", np.linalg.norm(A @ b - alpha*np.ones(SIZE)))
    print("Inv. error c", np.linalg.norm(A @ c - (gamma*np.ones(SIZE)-P)))


    def model_f(arr):
        k = A @ arr
        return -arr + [Z(k[i] + P[i]) for i in range(len(arr))]


    def red_f(arr):
        return -arr[0] + Z(alpha * arr[0] + gamma)


    x0 = 0.1*np.ones(SIZE)
    sim = stability_run(model_f, DT, None, x0, debugging=2, title='Simulation', max_run_time=100)
    pred = stability_run(red_f, DT, None, [a @ x0], debugging=2, title='Prediction', max_run_time=100)

    simrs, predrs, dist1s, dist2s = [], [], [], []
    taylor1s, taylor2s = [], []
    for i in range(len(sim[2])):
        x = sim[2][i]
        simr = np.abs(a @ x)
        predr = np.abs(pred[2][i])
        taylor = simr * b + c
        dist1 = np.abs(x[0] - simr * b[0] - c[0])
        dist2 = np.abs(x[1] - simr * b[1] - c[1])

        simrs.append(simr)
        predrs.append(predr)
        dist1s.append(dist1)
        dist2s.append(dist2)
        taylor1s.append(np.abs(taylor[0]))
        taylor2s.append(np.abs(taylor[1]))

    plt.plot(simrs, label='Simulated R')
    plt.plot(predrs, label='Predicted R')
    # plt.plot(np.array(dist1s), label='Dist 1')
    # plt.plot(np.array(dist2s), label='Dist 2')
    # plt.plot(np.array(dist1s)+np.array(dist2s), label='Total dist')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot([np.abs(x[0]) for x in sim[2]], label='Simulated 1')
    plt.plot([np.abs(x[1]) for x in sim[2]], label='Simulated 2')
    plt.plot(taylor1s, label='Taylor 1')
    plt.plot(taylor2s, label='Taylor 2')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot([np.abs(sim[2][i][0])-taylor1s[i] for i in range(len(sim[2]))], label='delta 1')
    plt.plot([np.abs(sim[2][i][1])-taylor2s[i] for i in range(len(sim[2]))], label='delta 2')
    plt.legend(loc='upper right')
    plt.show()

    # for x in sim[2]:
    #     total = 0
    #     for i, c in enumerate(x):
    #         total += a[i] * Z()

    sims.append(a @ sim[0])
    preds.append(pred[0][0])
    print("Simulation:", a @ sim[0], "Prediction:", pred[0][0])
    nonlinearity_errors.append(error)

# plt.plot(ALPHAS, sims, '.', label='Simulation')
# plt.plot(ALPHAS, preds, '.', label='Prediction')
# plt.legend()
# plt.show()
#
# plt.plot(ALPHAS, [abs(s-p) for s, p in zip(sims, preds)], '.', label='Absolute error')
# plt.plot(ALPHAS, np.array(nonlinearity_errors), '.', label='Nonlinearity error')
# plt.legend()
# plt.show()

