import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import json
import time
from dataclasses import dataclass, field

from auxiliary_functions import real, RK4step, rootfinder, exp, plot_solution, dx_dt
from data_eater import reduce
from network_class import Network

# ECB_link = scipy.io.loadmat('ECB_link.mat')["outputs"]
# ECB_node = scipy.io.loadmat('ECB_node.mat')["outputs"]
# ECB_weight = scipy.io.loadmat('ECB_weight.mat')["outputs"]
# print(ECB_link.shape)
# print(ECB_weight.shape)
# print(ECB_node.shape)

TALKING = True


def calculate_a1a2(c_arr, w_t, vs):
    """Calculates the inner product between a1 and a2"""
    n = len(vs)  # dimension of reduction
    m = len(vs[0])  # dimension of vector
    a1 = np.sum([c_arr[i] * vs[i] for i in range(n)], axis=0) / np.sum([c_arr[i]*(np.ones(m) @ vs[i]) for i in range(n)])
    alpha1 = np.ones(a1.shape) @ w_t @ a1
    a2 = (w_t @ a1) / alpha1
    a1_inner_a2 = np.transpose(a1) @ a2
    return np.abs(a1_inner_a2)


def stability_run(func: Callable[[List[float]], List[float]], dt, stabtol, x0, debugging=0, title="Stability",
                  max_run_time=50) -> (list[float], int, list[list[float]]):
    """Calculates a network until stability has been reached. \n
    If stabtol is set to None, then the first max_run_time seconds are calculated regardless.\n
    Debugging=0 no plots, =1 full debug, =2 only the final plot.\n
    Returns the stable solution vector and the index at stability and a list of xs over time."""
    normF = 100  # Some large number
    x = np.copy(x0)

    xs = [np.copy(x)]

    i = 1
    # If stabtol is not given, we run for a fixed time, else we run till the norm of the derivative is less than stabtol
    while ((stabtol is not None and normF > stabtol and i < max_run_time/dt)
           or (stabtol is None and i < max_run_time/dt)):
        x += RK4step(dt, func, x)
        normF = np.linalg.norm(func(x))
        xs.append(np.copy(x))
        if debugging == 1 and i % (10/dt) == 0:  # plot every number of computed seconds
            plt.plot([dt * n for n in range(len(xs))], xs)
            plt.show()
            time.sleep(1)
        i += 1

    if debugging > 0:
        plot_solution(dt, title, xs)
        time.sleep(1)

    return x, i, xs


@dataclass
class Result:
    """A dataclass for various types of results, from a random stability analysis to a laurence analysis"""
    w_matrix: np.ndarray
    # TODO: remove hardcode that assumes 2 parameters
    parameter1_list: List[float] = field(default_factory=list)
    parameter2_list: List[float] = field(default_factory=list)
    stable_x_list: List[np.ndarray] = field(default_factory=list)

    alphas: List[np.ndarray] = field(default_factory=list)
    betas: List[np.ndarray] = field(default_factory=list)
    a_vectors: List[np.ndarray] = field(default_factory=list)
    predictions: List[np.ndarray] = field(default_factory=list)
    simulations: List[np.ndarray] = field(default_factory=list)


def multi_dim_reduce(n, w_t):
    # Sort eigenvalues (and vectors) by norm, descending, and with positives first
    eigs, vecs = np.linalg.eig(w_t)
    size = w_t.shape[0]
    arranged = [[vecs[:, i], eigs[i]] for i in range(size)]
    arranged.sort(key=lambda item: np.conj(item[1]) * item[1], reverse=True)
    if w_t.all() >= 0:
        for index in range(size - 1):
            if np.linalg.norm(arranged[index][1]) == np.linalg.norm(arranged[index + 1][1]):
                # We want to sort by positive eigenvalues first
                if np.real(arranged[index][1]) < 0:
                    arranged[index], arranged[index + 1] = arranged[index + 1], arranged[index]

    # Get eigenvectors and K matrix
    vs = [item[0] for item in arranged[:n]]
    cs = np.array([0.5 for _ in range(n)])  # guess for the 'c' weight vector that defines the 'a' vectors
    K = np.diag([sum(w_t[:, i]) for i in range(size)])

    # Find 'c' vector by minimizing the inner product between a1 and a2
    res = minimize(lambda c_arr: calculate_a1a2(c_arr=c_arr, w_t=w_t, vs=vs), cs)
    c = res.x
    # print(f'Chosen c array = {c}: {res.message} Eigenvector overlap = {calculate_a1a2(c, w_t, alphas[0], vs)}')

    # Use the minimizing 'c' vector to calculate 'a' vectors and betas
    a = [np.sum([c[i] * vs[i] for i in range(n)], axis=0) /
         np.sum([c[i] * np.dot(np.ones(size), vs[i]) for i in range(n)])]
    alphas = [np.ones(size) @ w_t @ a[0]]
    betas = [a[0] @ K @ np.transpose(a[0]) / (a[0] @ np.transpose(a[0]) * alphas[0])]
    for i in range(1, n):
        a.append(np.dot(w_t, a[-1]) / alphas[i - 1])
        alphas.append(np.ones(size) @ w_t @ a[-1])
        betas.append(a[-1] @ K @ np.transpose(a[-1]) / (a[-1] @ np.transpose(a[-1]) * alphas[i]))
    return a, alphas, betas


class Model:
    def __init__(self, W: np.ndarray, F: Callable[[float], float], G: Callable[[float, float], float]):
        self.w = W
        self.decay_func = F
        self.interact_func = G
        self.dim = self.w.shape[0]
        self.solution = 0

    def run(self, x0: np.ndarray, tf: float = 10):
        """Compute N iterations with initial condition x0, from t=0 to tf, with precision dt"""
        self.solution = solve_ivp(lambda t, x: dx_dt(x, self.decay_func, self.interact_func, self.w), t_span=(0, tf),
                                  y0=x0)

    def stability_run(self, dt, stabtol, x0, debugging=0, title="Stability Run"):
        """Run the stability analysis for the model"""
        def func(arr): return dx_dt(tuple(arr), self.decay_func, self.interact_func,
                                    tuple(tuple(row) for row in self.w))
        return stability_run(func, dt, stabtol, x0, debugging, title=title)

    def random_stability_analysis(self, p1p, p2p, dt, stabtol=1e-5, max_run_time=50):
        """Run an analysis of stable solutions with random initial vector for different input parameters"""
        p1s = np.linspace(p1p[0], p1p[1], p1p[2])  # tau
        p2s = np.linspace(p2p[0], p2p[1], p2p[2])  # mu

        result = Result(w_matrix=np.copy(self.w))
        t1 = time.time()
        for p1 in p1s:
            for p2 in p2s:
                self.decay_func = lambda x: -x
                const = 1 / (1 + np.exp(p1 * p2))
                self.interact_func = lambda x, y: 1 / (1 + np.exp(p1 * (p2 - y))) - const

                x = np.random.rand(self.dim)*self.dim+self.dim

                func = lambda arr: dx_dt(tuple(arr), self.decay_func, self.interact_func, tuple(tuple(row) for row in self.w))
                x = stability_run(func=func, dt=dt, stabtol=stabtol, x0=x, debugging=False, max_run_time=max_run_time)[0]

                result.parameter1_list.append(p1)
                result.parameter2_list.append(p2)
                result.stable_x_list.append(x)

            if TALKING and __name__ == "__main__":
                print(f"Completed p1={round(p1,3)}. Total time taken: {round((time.time()-t1) // 60)} min and {round((time.time()-t1) % 60)} s")

        if __name__ == "__main__":
            results = [result.w_matrix.tolist(), result.parameter1_list, result.parameter2_list,
                       np.mean(result.stable_x_list, axis=1).tolist(),
                       np.linalg.norm(result.stable_x_list, axis=1).tolist()]
            with open('data/CowanWilsonstabilityresults'+str(time.time())+'.txt', 'w') as filehandle:
                json.dump(results, filehandle)

        return result

    def laurence_one_dim_analysis_run(self, x0, dt=1e-1, stabtol=1e-3, debugging=0):
        # Definitions of One-Dimensional Model
        a, alpha, beta = reduce(self.w)

        # Stability run
        x = stability_run(lambda arr:
                          dx_dt(tuple(arr), self.decay_func, self.interact_func, tuple(tuple(row) for row in self.w)),
                          dt, stabtol, x0, debugging=debugging)[0]
        simulation = real(a @ x)

        def reduced_eq(R): return self.decay_func(R) + alpha * self.interact_func(beta * R, R)
        prediction = rootfinder(reduced_eq,[-1, 50], etol=1e-3)
        for p in prediction:
            if reduced_eq(p - 1e-2) < 0 < reduced_eq(p + 1e-2):
                # Unstable equilibrium, let's remove it
                prediction.remove(p)

        return alpha, simulation, prediction, x  # pass x to use as x0 next run (speeds up running time)

    def laurence_multi_dim_analysis_run(self, x0, dim: int = 0, dt=1e-2, stabtol=1e-3, max_run_time=50, debugging=0):
        """Returns simulation result, prediction with multidimensional reduction, list of 'a' vectors, alphas and
        beta values used (in order of eigenvalue norm descending)."""
        w_t = np.transpose(self.w)
        eigs, vecs = np.linalg.eig(w_t)

        # I/O and validation
        # print("First ten eigenvalues:", eigs[:10], "\n")
        if dim == 0:
            print("Sorted, normed eigenvalues:", np.sort(np.real(np.abs(eigs))), "\n")
            n = input("How many eigenvalues are significant? (integer): ")
            try:
                n = int(n)
            except ValueError:
                print("That's not an integer!")
                return self.laurence_multi_dim_analysis_run(x0=x0, dt=dt, stabtol=stabtol, max_run_time=max_run_time)
        else:
            n = dim

        a, alphas, betas = multi_dim_reduce(n, w_t)

        # Let's integrate the reduced ODE!
        func = lambda Rs: np.array([self.decay_func(Rs[i]) + alphas[i] *
                                    self.interact_func(betas[i] * Rs[i], Rs[i + 1]) for i in range(n - 1)] +
                                   [self.decay_func(Rs[-1]) + alphas[-1] * self.interact_func(betas[-1] * Rs[-1], Rs[0])])
        prediction = stability_run(func, dt, stabtol, np.array([a[i] @ x0 for i in range(n)]),
                                   title=f"{n}D reduction", debugging=debugging, max_run_time=max_run_time)[0]

        # Integrate the original ODE
        func = lambda arr: dx_dt(tuple(arr), self.decay_func, self.interact_func, tuple(tuple(row) for row in self.w))
        x = stability_run(func, dt, stabtol, x0, title="Full simulation", debugging=debugging, max_run_time=max_run_time)[0]
        simulation_res = np.array([a[i] @ x for i in range(n)])

        # dec = 3
        # print(f"Error (act: {np.round(simulation_res, decimals=dec)} pred: {np.round(prediction, decimals=dec)}) = "
        #       f"{np.round(np.linalg.norm(simulation_res-prediction), decimals=dec)}")

        return simulation_res, prediction, a, alphas, betas


if __name__ == "__main__":
    net = Network()
    # net.gen_community([10, 10, 10])
    # net.randomize_weights()
    #
    # const = 1/(1 - exp(3))
    #
    #
    # def decay(x):
    #     return -x
    #
    #
    # def interact(x, y):
    #     return 1/(1 + exp(3 - y)) - const
    #
    #
    # w = net.adj_matrix
    # model = Model(w, decay, interact)
    #
    # x0 = 5 * np.random.rand(w.shape[0]) + 5
    # dt = 1e-2
    # stabtol = 1e-5
    #
    # res = model.laurence_multi_dim_analysis_run(dt=dt, stabtol=stabtol, x0=x0)
    #
    # res = model.laurence_multi_dim_analysis_run(dt=dt, stabtol=stabtol, x0=x0)

    ### Non-cooperative Networks ###
    # N = 15
    # w = 2*np.random.rand(N, N)-np.ones((N, N))
    # w = np.random.exponential(scale=1/10, size=(N,N))
    # w = np.random.normal(loc=0, scale=1, size=(N, N))
    # for i in range(N):
    #     w[i, i] = 0
    #
    # tau, mu = 1e1, 1e-1
    # symm = Model(w, lambda x: -x, lambda x, y: 1/(1+np.exp(tau*(mu-y))))
    # x0 = symm.stabilityrun(dt=1e-3, stabtol=1e-2, x0=np.random.rand(N)+mu, debugging=True)

    # # Perturb and check stability!
    # N_pert = N
    # for n in range(N_pert):
    #         i = np.random.choice([k for k in range(N) if sum(w[k, :]) != 0])
    #         j = np.random.choice([k for k in range(N) if w[i, k] != 0])
    #         w[i, j] = 0
    #
    # symm.W = w
    # symm.stabilityrun(dt=1e-1, stabtol=1e-2, x0=x0, debugging=True)

    ### Symmetric Systems ###
    # dim = 3
    # w = np.array([[0, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 0]])
    # if dim != w.shape[0]:
    #     input("Are you sure that's right?")
    # # for i in range(dim):
    # #     if i < dim-1:
    # #         w[i, i+1] = 1
    # #     else:
    # #         w[dim-1, 0] = 1
    #
    # symm = Model(w, lambda x: -x, lambda x, y: x*(1/(1+np.exp(-10*y))-1/2))
    #
    # stable_vectors = []
    # for i in range(50):
    #     x0 = dim*np.random.rand(dim)
    #     sol = symm.stabilityrun(dt=1e-3, stabtol=1e-3, x0=x0, debugging=True)
    #     stable_vectors.append(sol)
    #     print(sol)
    #
    # plt.plot([np.mean(stable_vectors[i]) for i in range(dim)])
    # plt.xlabel("mean(x0)-1")
    # plt.ylabel("mean(solution)")
    # plt.show()



    ### OLD LAURENCE ANALYSIS ###
    # dim = 10

    # W = np.random.rand(dim, dim)
    # eig = np.linalg.eig(W)
    # print("Eigenvalues of W:", eig[0], "\n")
    # print("Norm squared eigenvalues of W:", np.multiply(np.conj(eig[0]), eig[0]), "\n")
    # F = lambda x: -x
    # G = lambda x, y: 1/(1+np.exp(-y+3))
    # model = Model(W, F, G)
    # model.run(np.random.rand(dim), 100)

    # plt.plot(model.solution.t, np.transpose(model.solution.y))
    # plt.show()

    ### WU ANALYSIS ###
    # N = 10
    # s = nx.utils.powerlaw_sequence(N, 2)  # 100 nodes, power-law exponent 2.5
    # G = nx.expected_degree_graph(s, selfloops=False)
    #
    # w = nx.adjacency_matrix(G).toarray()
    #
    # # draw and show graph
    # pos = nx.spring_layout(G, k=1/N**0.1)
    # nx.draw_networkx(G, pos)
    # plt.show()

    # w = np.array([[0,0,0,0,0,0,0,0,0,1],
    #               [0,0,0,1,0,0,0,0,1,0],
    #               [0,0,0,0,0,1,0,0,1,1],
    #               [0,1,0,0,0,1,1,1,1,1],
    #               [0,0,0,0,0,0,0,0,1,0],
    #               [0,0,1,1,0,0,0,0,1,0],
    #               [0,0,0,1,0,0,0,0,0,0],
    #               [0,0,0,1,0,0,0,0,0,0],
    #               [0,1,1,1,1,1,0,0,0,1],
    #               [1,0,1,1,0,0,0,0,1,0]], dtype=float)
    #
    # w = np.array([[0, 1, 1, 1],
    #               [1, 0, 0, 0],
    #               [0, 1, 0, 1],
    #               [0, 0, 1, 0]])
    #
    # model = Model(W=w, F=lambda x: -x, G=lambda x, y: 1 / (1 + np.exp(5*(0.1-y))) - 1/(1+np.exp(0.5)))
    #
    # print(model.stabilityrun(1e-2, 1e-8, np.random.rand(4), True))
    #
    # w = np.array([[0, 1, 1, 1, 0],
    #               [1, 0, 0, 0, 0],
    #               [0, 1, 0, 1, 1],
    #               [0, 0, 1, 0, 0],
    #               [1, 0, 0, 0, 0]])
    #
    # model = Model(W=w, F=lambda x: -x, G=lambda x, y: 1 / (1 + np.exp(5*(0.1-y))) - 1/(1+np.exp(0.5)))
    #
    # print(model.stabilityrun(1e-2, 1e-8, np.random.rand(5), True))
    #
    # w = np.array([[0, 1, 1, 1, 0],
    #               [1, 0, 0, 0, 0],
    #               [0, 3, 0, 1, 2],
    #               [0, 0, 1, 0, 0],
    #               [1, 0, 0, 0, 0]])
    #
    # model = Model(W=w, F=lambda x: -x, G=lambda x, y: 1 / (1 + np.exp(5*(0.1-y))) - 1/(1+np.exp(0.5)))
    #
    # print(model.stabilityrun(1e-2, 1e-8, np.random.rand(5), True))
    #
    # input("BROOOOO")
    # for i in range(w.shape[0]):
    #     for j in range(w.shape[1]):
    #         if w[i, j] != 0:
    #             w[i, j] += 2*np.random.rand()-1

    # eigs, vecs = np.linalg.eig(w)
    # print(eigs)
    # print(np.multiply(eigs, np.conjugate(eigs)))
    # input("Continue?")
    #
    # model = Model(W=w, F=lambda x: -x, G=lambda x, y: 1 / (1 + np.exp(5*(1-y))) - 1/(1+np.exp(5)))

    # model.randomStabilityAnalysis(20, 20, 1e-2)

    # cache_info = dxdt.cache_info()
    # print("Cache hits:", cache_info.hits)
    # print("Cache misses:", cache_info.misses)

    ### LAURENCE 1D ANALYSIS ###
    # t1 = time.time()
    # t2 = t1
    # cwf, cwg = lambda x: -x, lambda x, y: 1/(1+np.exp(3-y))
    #
    # alphas, sims = [], []
    # alphap, preds = [], []
    # w = np.random.rand(30, 30)
    # x0 = np.random.rand(w.shape[0])*5
    #
    # N = 100
    # for m in range(N):
    #     model = Model(W=w, F=cwf, G=cwg)
    #
    #     # perturb!
    #     for n in range((900 // N)):
    #         i = np.random.choice([k for k in range(w.shape[0]) if sum(w[k, :]) > 0])
    #         j = np.random.choice([k for k in range(w.shape[1]) if w[i, k] > 0])
    #         w[i, j] = 0
    #
    #     alpha, sim, pred, x0 = model.LaurenceOneDimAnalysisRun(x0=x0)
    #
    #     alphas.append(alpha)
    #     sims.append(sim)
    #
    #     for p in pred:
    #         alphap.append(alpha)
    #         preds.append(p)
    #
    #     print(f"Completed i={m}. Time taken: {round(1000*(time.time()-t2))} ms\n")
    #     t2 = time.time()
    #
    # print(f"Total time taken: {round((time.time()-t1) // 60)} min and {round((time.time()-t1) % 60)} s")
    #
    # plt.plot(alphap, preds, '.', label="pred", markersize=10)
    # plt.plot(alphas, sims, '.', label="sim", markersize=5)
    # plt.legend()
    # plt.show()








