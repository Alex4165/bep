import numpy as np
import time
from auxiliary_functions import exp, format_time_elapsed
from data_eater import reduce
from model import stability_run
import matplotlib.pyplot as plt

taus = np.linspace(0.5, 5, 2)
mus = np.linspace(0, 4.5, 2)
MUS, TAUS = np.meshgrid(mus, taus)
ERRORS = np.zeros(MUS.shape)

SIZE = 10
DT = 1e-1
STABTOL = 1e-4
MAX_RUN_TIME = 100
PROB_INHIB = 0.5
P_0 = 0.2
SCALE = 3
RUNS_PER_POINT = 15
NUM_ALPHAS = 20

print(f"{len(mus) * len(taus) * RUNS_PER_POINT * NUM_ALPHAS} simulations will be run at dt={DT}")

P_VECTOR = np.zeros(SIZE)  # Written here for clarity


def Z(x, tau_, mu_):
    num = 1 / (1 + exp(tau_ * (mu_ - x))) - 1 / (1 + exp(tau_ * mu_))
    den = 1 - 1 / (1 + exp(tau_ * mu_))
    return num / den


def proposed_eig(A, speak=False):
    """The proposed solution to the eigenvalue selection problem"""
    eigs, vecs = np.linalg.eig(A.T)
    norms = []
    for ix in range(SIZE):
        a_ix = vecs[:, ix] / sum(vecs[:, ix])
        norms.append(np.linalg.norm(a_ix - 1))
    eig_i = np.argmin(norms)
    a, alpha = vecs[:, eig_i] / sum(vecs[:, eig_i]), eigs[eig_i]
    if speak:
        return a, alpha, (sorted(norms)[0] == sorted(norms)[1])
    return a, alpha


start = time.time()
for q in range(RUNS_PER_POINT):
    # Generate network
    columns = []
    for k in range(SIZE):
        col = np.random.choice([1., 0.], size=SIZE, p=[1 - P_0, P_0])
        col *= np.random.exponential(scale=SCALE, size=SIZE)
        if np.random.random() < PROB_INHIB:
            col *= -1
        columns.append(col)
    A = np.column_stack(columns)
    print(f"Eigenvalues run={q}: {np.linalg.eigvals(A)}")

    a, alpha_zero, is_multi_dim = proposed_eig(A, speak=True)
    print(f"Chosen eig: {alpha_zero}")

    # Compute invertibility error (we assume P = 0)
    b_zero = np.linalg.lstsq(A, alpha_zero * np.ones(SIZE), rcond=None)[0]
    inv_error = np.linalg.norm(A @ b_zero - alpha_zero * np.ones(SIZE))
    if inv_error > 1e-2:
        print("Invertibility error too large:", inv_error)

    saved = 0
    # Run simulations
    for i in range(MUS.shape[0]):
        for j in range(MUS.shape[1]):
            mu = MUS[i, j]
            tau = TAUS[i, j]

            # Get predicted smallest alpha where system survives
            pred_alpha = mu + 2 / tau
            alphas = np.linspace(pred_alpha - 2, pred_alpha + 8, NUM_ALPHAS)
            dalpha = alphas[1] - alphas[0]

            # Run simulations and get total area of the difference between predicted and simulated R
            error_area = 0

            # temp
            preds, sims = [], []
            for ideal_alpha in alphas:
                # Shift W to have the almost alpha
                W = ideal_alpha * A / np.real(alpha_zero)
                a, alpha = proposed_eig(W)


                def model_f(arr):
                    incoming_sum = W @ arr
                    return -arr + [Z(incoming_sum[m], tau_=tau, mu_=mu) for m in range(len(arr))]


                def reduced_f(arr):
                    return -arr[0] + Z(alpha * arr[0], tau_=tau, mu_=mu)


                x0 = 0.9 + 0.1 * np.random.random_sample(SIZE)
                sim = stability_run(model_f, DT, STABTOL, x0, max_run_time=MAX_RUN_TIME, debugging=0,
                                    title=f"simulation_tau={tau}_mu={mu}_run={q}_{round(time.time()) % 1000}")
                pred = stability_run(reduced_f, DT, STABTOL, [a @ x0], max_run_time=MAX_RUN_TIME, debugging=0,
                                     title=f"reduction_tau={tau}_mu={mu}_run={q}_{round(time.time()) % 1000}")

                error_area += np.abs(a @ sim[0] - pred[0][0]) * dalpha

                # temp
                preds.append(np.abs(pred[0][0]))
                sims.append(np.abs(a @ sim[0]))

            # temp
            if saved <= 2:
                saved += 1
                plt.figure(figsize=(5, 4), dpi=400)
                plt.plot(alphas, preds, '.', label='Pred')
                plt.plot(alphas, sims, '.', label='Sim')
                plt.fill_between(alphas, preds, sims, alpha=0.5, color='grey')
                # plt.axvline(pred_alpha, color='r', label='Pred alpha')
                # plt.title(f"tau={tau}, mu={mu}")
                plt.xlabel("Alpha (real)")
                plt.ylabel("R")
                plt.legend()
                plt.savefig(f'data/fig5.2_tau={tau}_mu={mu}_error={round(error_area,1)}.png')
                plt.title(is_multi_dim)
                plt.show()

            ERRORS[i, j] += error_area

    print(f"Estimated {format_time_elapsed((time.time() - start) / (q + 1) * (RUNS_PER_POINT - q - 1))} remaining\n")

ERRORS /= RUNS_PER_POINT

# A heatmap of the error area for each tau and mu
plt.figure(figsize=(5, 4), dpi=600)
plt.pcolormesh(TAUS, MUS, ERRORS, shading='auto', cmap='viridis')
plt.colorbar(label='Error area')
plt.xlabel('Tau')
plt.ylabel('Mu')
plt.savefig(f'data/fig5.2_runs={RUNS_PER_POINT}_alphas={NUM_ALPHAS}.png')
plt.show()

print("Phew, that took:", format_time_elapsed(time.time() - start))
