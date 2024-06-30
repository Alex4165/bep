from model import Model
from data_eater import lambda_star, one_laurence, kmax, rho

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

N = 4
STEPS = 4


kmax = kmax(w)
rho = rho(w)

F = lambda x, tau, mu: -x
G = lambda x, y, tau, mu: 1/(1+np.exp(tau*(mu-y)))

model = Model(w, F=lambda x: F(x, 0, 0), G=lambda x, y: G(x, y, 1, 3))

av_deg = []
mses = []
accuracy = []
t1 = time.time()
for index in range(2*N):
    av_deg.append(np.transpose(np.ones(N)) @ w @ np.ones(N) / N)
    res = model.random_stability_analysis(p1p=[1, 10, STEPS], p2p=[1, 10, STEPS], dt=1e-2)
    R_norms, a = one_laurence(res, adj_matrix=w, decay_func=F, interact_func=G)

    count = 0
    for i in range(STEPS**2):
        if lambda_star(res[1][i], res[2][i]) <= kmax and res[4][i] > 1e-2:
            # Predicted survival
            count += 1
        elif lambda_star(res[1][i], res[2][i]) >= rho and res[4][i] < 1e-2:
            # Predicted collapse
            count += 1

    lambda_stars = [lambda_star(tau, mu) for (tau, mu) in zip(res[1], res[2])]
    lower_bound_alpha = np.where(lambda_stars <= kmax, 1, 0)
    upper_bound_alpha = np.where(lambda_stars >= rho, 1, 0)
    plt.scatter(res[1], res[2], c=res[4], cmap='viridis', s=200)
    plt.colorbar(label="Data norm")
    plt.scatter(res[1], res[2], c='b', alpha=lower_bound_alpha, s=25)
    plt.scatter(res[1], res[2], c='r', alpha=upper_bound_alpha, s=25)
    plt.show()

    mses.append(sum([(res[4][i]-R_norms[i])**2 for i in range(len(R_norms))]))
    accuracy.append(count/STEPS**2)
    # Perturb
    i = np.random.choice([k for k in range(w.shape[0])])
    j = np.random.choice([k for k in range(w.shape[1])])
    w[i, j] -= 2

    duration = time.time() - t1
    remaining = (2*N - index - 1) * duration / (index + 1)
    print(f"Done {index+1}. Estimated time remaining: "
          f"{round(remaining // 3600)} h {round(remaining // 60)} m {round(remaining % 60)} s")

plt.plot(av_deg, mses)
plt.show()

plt.plot(av_deg, accuracy)
plt.show()







