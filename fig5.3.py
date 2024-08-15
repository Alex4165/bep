import json

import numpy as np
import time
from auxiliary_functions import exp, format_time_elapsed, plot_solution
from data_eater import reduce
from model import stability_run
import matplotlib.pyplot as plt

# Simulation parameters
DT = 1e-1
STABTOL = 1e-4
MAX_RUN_TIME = 100

# System parameters
PROB_INHIB = 0.5
P_0 = 0.2
CONNECTION_SCALE = 3
P_VECTOR_SCALE = 0

# Experiment parameters
SIZES = [_ for _ in range(5, 70, 5)]
NETWORKS_PER_SIZE = 1000
print(f"Running {len(SIZES)*NETWORKS_PER_SIZE}")


def Z(x, tau_, mu_):
    num = 1 / (1 + exp(tau_ * (mu_ - x))) - 1 / (1 + exp(tau_ * mu_))
    den = 1 - 1 / (1 + exp(tau_ * mu_))
    return num / den


def proposed_eig(A, flag_double_alpha=False):
    """The proposed solution to the eigenvalue selection problem"""
    eigs, vecs = np.linalg.eig(A.T)
    norms = []
    for ix in range(A.shape[0]):
        a_ix = vecs[:, ix] / sum(vecs[:, ix])
        norms.append(np.linalg.norm(a_ix - 1))
    eig_i = np.argmin(norms)
    a, alpha = vecs[:, eig_i] / sum(vecs[:, eig_i]), eigs[eig_i]
    if flag_double_alpha:
        return a, alpha, (sorted(norms)[0] == sorted(norms)[1])
    return a, alpha


ERRORS_LIST = [[] for _ in range(len(SIZES))]
start = time.time()
for i, size in enumerate(SIZES):
    for q in range(NETWORKS_PER_SIZE):
        # Generate network
        # double_alpha = True
        # while double_alpha:
        columns = []
        for k in range(size):
            col = np.random.choice([1., 0.], size=size, p=[1 - P_0, P_0])
            col *= np.random.exponential(scale=CONNECTION_SCALE, size=size)
            if np.random.random() < PROB_INHIB:
                col *= -1
            columns.append(col)
        A = np.column_stack(columns)
        # double_alpha = proposed_eig(A, flag_double_alpha=True)[2]

        # Generate p vector
        P_VECTOR = np.random.exponential(scale=P_VECTOR_SCALE, size=size)
        NONZERO_P_VECTOR: bool = (sum(p > 0 for p in P_VECTOR) > 0)

        # Compute invertibility error
        b_zero = np.linalg.lstsq(A, np.ones(size), rcond=None)[0]
        inv_error = np.linalg.norm(A @ b_zero - np.ones(size))
        if inv_error > 1e-2:
            print("Invertibility error on b too large:", inv_error)

        tau, mu = (np.random.random() * 4.5 + 0.5), (np.random.random() * 5)  # Random tau and mu
        a, alpha = proposed_eig(A)

        # Compute invertibility error for c
        if NONZERO_P_VECTOR:
            c = np.linalg.lstsq(A, (a @ P_VECTOR) * np.ones(size) - P_VECTOR, rcond=None)[0]
            inv_error = np.linalg.norm(A @ c - (a @ P_VECTOR) * np.ones(size) + P_VECTOR)
            if inv_error > 1e-2:
                print("Invertibility error on c too large:", inv_error)


        def model_f(arr):
            incoming_sum = A @ arr
            return -arr + [Z(incoming_sum[m], tau_=tau, mu_=mu) for m in range(len(arr))]


        def reduced_f(arr):
            return -arr[0] + Z(alpha * arr[0], tau_=tau, mu_=mu)


        x0s = [0.9 + 0.1 * np.random.random_sample(size), 0.1 * np.random.random_sample(size)]
        for x0 in x0s:
            sim = stability_run(model_f, DT, STABTOL, x0, max_run_time=MAX_RUN_TIME, debugging=0)
            pred = stability_run(reduced_f, DT, STABTOL, [a @ x0], max_run_time=MAX_RUN_TIME, debugging=0)

            ERRORS_LIST[i].append(np.abs(a @ sim[0] - pred[0][0]))

    print(f"At {round(100 * (i + 1) / len(SIZES))}%. "
          f"Estimated {format_time_elapsed((time.time() - start) / (i + 1) * (len(SIZES) - i - 1))} remaining")

with open("data/fig5.3error_list.txt", "w") as f:
    json.dump(ERRORS_LIST, f)

plt.figure(figsize=(5, 4), dpi=600)
plt.errorbar(SIZES, [np.mean(error) for error in ERRORS_LIST],
             [np.std(ERRORS_LIST[i]) / np.sqrt(len(ERRORS_LIST)) for i in range(len(SIZES))], capsize=5, marker='o',
             label='Mean')
plt.ylabel("Mean absolute error")
plt.xlabel("Network size")
bottom, top = plt.gca().get_ylim()
if top > 1:
    plt.ylim(bottom, 1)
plt.savefig(f"data/fig5.3_pscale={P_VECTOR_SCALE}_numpoints={len(SIZES) * NETWORKS_PER_SIZE}.png")
plt.show()

print("Phew, that took:", format_time_elapsed(time.time() - start))
