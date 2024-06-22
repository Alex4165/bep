from model import Model, RK4step, dxdt
from data_eater import reduce

import numpy as np
import matplotlib.pyplot as plt

size = 5

W0 = np.array([[0.52829123, -0.12620005, 0.07649462, -0.51603925, 0.3324112],
               [-0.65911129, -0.2884593, 0.35157307, -0.61664797, -0.47923406],
               [-0.95658735, -0.99555071, 0.61247847, 0.57499545, -0.185355],
               [-0.63192054, -0.36443042, 0.76658192, -0.94645962, 0.85193479],
               [-0.41915602, -0.02816688, -0.39475975, -0.48100525, 0.8211069]])

F = lambda x: -x**3
G = lambda x, y: (1-x)*y

model = Model(W0, F, G)
for step in range(0, int(size*size/1.5), max(int(size*size/15), 1)):
    if step > 0:
        break
    W = np.copy(W0)
    # coords = [(i, j) for i in range(W.shape[0]) for j in range(W.shape[1])]
    # for n in range(int(size**2/1.5)):
    #     i, j = coords[np.random.randint(len(coords))]
    #     W[i, j] = -np.random.rand()
    #     coords.remove((i, j))
    # model.W = W

    dt, stabtol, x0 = 1e-2, 1e-3, 0.5*np.random.rand(size)+0.25
    av_k_in = round(np.transpose(np.ones(size)) @ W @ np.ones(size)/size, 2)

    model.LaurenceMultiDimAnalysisRun(x0=x0, dt=dt, stabtol=stabtol)

    func = lambda arr: dxdt(tuple(arr), F, G, tuple(tuple(row) for row in W))
    xf, N = model.stabilityrun(dt=dt, stabtol=stabtol, x0=x0, debugging=2, title=f"step {step}, kin {av_k_in}")
    print("normed eigs", np.sort(np.absolute(np.linalg.eig(W)[0])))
    a, alpha, beta = reduce(W)
    f = lambda R: F(R) + alpha * G(beta*R, R)

    Rs = [a @ x0]
    for i in range(N-1):
        R = Rs[-1]
        dR = RK4step(dt, f, Rs[-1])
        Rs.append(Rs[-1] + dR)

    print("error:", np.absolute(a @ xf - Rs[-1]), "step", step)
    # print(W)

    plt.plot([i*dt for i in range(N)], np.absolute(Rs))
    plt.title(f"R amplitude, step {step}")
    plt.show()

    plt.plot([i*dt for i in range(N)], np.angle(Rs))
    plt.title(f"R argument, step {step}")
    plt.show()

    print()




