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
SIZE = 10
PROB_INHIB = 0.5
P_0 = 0.2
CONNECTION_SCALE = 3
P_VECTOR_SCALE = 0
print(P_VECTOR_SCALE)

# Experiment parameters
NUM_POINTS = 5000


def Z(x, tau_, mu_):
    num = 1 / (1 + exp(tau_ * (mu_ - x))) - 1 / (1 + exp(tau_ * mu_))
    den = 1 - 1 / (1 + exp(tau_ * mu_))
    return num / den


def largest_eig(A):
    """Take the largest one you can find"""
    eigs, vecs = np.linalg.eig(A.T)
    eig_i = np.argmax(np.abs(eigs))
    a, alpha = vecs[:, eig_i] / sum(vecs[:, eig_i]), eigs[eig_i]
    return a, alpha


def smallest_eig(A):
    eigs, vecs = np.linalg.eig(A.T)
    eig_i = np.argmin(np.abs(eigs))
    a, alpha = vecs[:, eig_i] / sum(vecs[:, eig_i]), eigs[eig_i]
    return a, alpha


def proposed_eig(A, flag_double_alpha=False):
    """The proposed solution to the eigenvalue selection problem"""
    eigs, vecs = np.linalg.eig(A.T)
    norms = []
    for ix in range(SIZE):
        a_ix = vecs[:, ix] / sum(vecs[:, ix])
        norms.append(np.linalg.norm(a_ix - 1))
    eig_i = np.argmin(norms)
    a, alpha = vecs[:, eig_i] / sum(vecs[:, eig_i]), eigs[eig_i]
    if flag_double_alpha:
        return a, alpha, (sorted(norms)[0]==sorted(norms)[1])
    return a, alpha


def minimize_b_eig(A):
    """Minimize the error in b=1"""
    eigs, vecs = np.linalg.eig(A.T)
    norms = []
    for ix in range(SIZE):
        b_ix = np.linalg.lstsq(A, eigs[ix] * np.ones(SIZE), rcond=None)[0]
        norms.append(np.linalg.norm(b_ix - 1))
    eig_i = np.argmin(norms)
    a, alpha = vecs[:, eig_i] / sum(vecs[:, eig_i]), eigs[eig_i]
    return a, alpha


def random_eig(A):
    """Randomly select an eigenvalue"""
    eigs, vecs = np.linalg.eig(A.T)
    eig_i = np.random.randint(SIZE)
    a, alpha = vecs[:, eig_i] / sum(vecs[:, eig_i]), eigs[eig_i]
    return a, alpha


METHODS = [smallest_eig, largest_eig, proposed_eig, minimize_b_eig, random_eig]


ERRORS = [[] for _ in range(len(METHODS))]
start, t1 = time.time(), time.time()
for q in range(NUM_POINTS):
    # Generate network
    double_alpha = True
    while double_alpha:
        columns = []
        for k in range(SIZE):
            col = np.random.choice([1., 0.], size=SIZE, p=[1 - P_0, P_0])
            col *= np.random.exponential(scale=CONNECTION_SCALE, size=SIZE)
            if np.random.random() < PROB_INHIB:
                col *= -1
            columns.append(col)
        A = np.column_stack(columns)
        double_alpha = proposed_eig(A, flag_double_alpha=True)[2]

    # Generate p vector
    P_VECTOR = np.random.exponential(scale=P_VECTOR_SCALE, size=SIZE)
    NONZERO_P_VECTOR: bool = (sum(p > 0 for p in P_VECTOR) > 0)

    # Compute invertibility error
    b_zero = np.linalg.lstsq(A, np.ones(SIZE), rcond=None)[0]
    inv_error = np.linalg.norm(A @ b_zero - np.ones(SIZE))
    if inv_error > 1e-2:
        print("Invertibility error on b too large:", inv_error)

    # W = (8 / 3 * np.random.random() + 1 / 3) * A  # Multiply A by a random number between 1/3 and 3
    W = A  # We remove this element of randomness to reduce deviation
    tau, mu = (np.random.random() * 4.5 + 0.5), (np.random.random() * 5)  # Random tau and mu
    for i, method in enumerate(METHODS):
        a, alpha = method(W)

        # Compute invertibility error for c
        if NONZERO_P_VECTOR:
            c = np.linalg.lstsq(W, (a @ P_VECTOR) * np.ones(SIZE) - P_VECTOR, rcond=None)[0]
            inv_error = np.linalg.norm(W @ c - (a @ P_VECTOR) * np.ones(SIZE) + P_VECTOR)
            if inv_error > 1e-2:
                print("Invertibility error on c too large:", inv_error)


        def model_f(arr):
            incoming_sum = W @ arr
            return -arr + [Z(incoming_sum[m], tau_=tau, mu_=mu) for m in range(len(arr))]


        def reduced_f(arr):
            return -arr[0] + Z(alpha * arr[0], tau_=tau, mu_=mu)


        x0s = [0.9 + 0.1 * np.random.random_sample(SIZE), 0.1 * np.random.random_sample(SIZE)]
        for x0 in x0s:
            sim = stability_run(model_f, DT, STABTOL, x0, max_run_time=MAX_RUN_TIME, debugging=0)
            pred = stability_run(reduced_f, DT, STABTOL, [a @ x0], max_run_time=MAX_RUN_TIME, debugging=0)

            ERRORS[i].append(np.abs(a @ sim[0] - pred[0][0]))

    if (q + 1) % int(NUM_POINTS / 100) == 0:
        print(
            f"At {(q + 1) // int(NUM_POINTS / 100)}%. Estimated {format_time_elapsed((time.time() - start) / (q + 1) * (NUM_POINTS - q - 1))} remaining")

NAME_DICT = {"largest_eig": "LE",
             "proposed_eig": "MA",
             "smallest_eig": "SE",
             "random_eig": "Random",
             "minimize_b_eig": "MB"}

# plt.boxplot(ERRORS, labels=[NAME_DICT[method.__name__] for method in METHODS])
# plt.ylabel("Error")
# plt.show()

with open("data/fig5.1error_list.txt","w") as f:
    json.dump(ERRORS, f)

plt.figure(figsize=(4, 3), dpi=600)
plt.errorbar([NAME_DICT[method.__name__] for method in METHODS], [np.mean(err) for err in ERRORS],
             [np.std(err) / np.sqrt(NUM_POINTS) for err in ERRORS], capsize=5, fmt='o', label='Mean')
plt.ylabel("Absolute error")
bottom, top = plt.gca().get_ylim()
if top > 1:
    plt.ylim(bottom, 1)
plt.savefig(f"data/fig5.1_pscale={P_VECTOR_SCALE}_numpoints={NUM_POINTS}.png")
plt.show()

# for i, error_list in enumerate(ERRORS):
#     plt.hist(np.array(error_list)/NUM_POINTS, label=NAME_DICT[METHODS[i].__name__], alpha=0.25)
# plt.legend()
# plt.xlabel("Error")
# plt.ylabel("Frequency")
# plt.show()

print("Phew, that took:", format_time_elapsed(time.time() - start))
