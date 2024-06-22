import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, fsolve
import scipy.io
import json
import time
import networkx as nx
import cProfile
from functools import lru_cache

# ECB_link = scipy.io.loadmat('ECB_link.mat')["outputs"]
# ECB_node = scipy.io.loadmat('ECB_node.mat')["outputs"]
# ECB_weight = scipy.io.loadmat('ECB_weight.mat')["outputs"]
# print(ECB_link.shape)
# print(ECB_weight.shape)
# print(ECB_node.shape)

TALKING = True


def real(x):
    if np.isreal(x):
        return np.real(x)
    else:
        print(f"Uh oh! {x} is not real")
        return np.real(x)


def RK4step(dt, f, x):
    """Returns one Runge-Kutta 4 integration step
     Note:  assumes f is independent of time!
            returns dx, not x+dx"""
    k1 = f(x)
    k2 = f(x + dt * k1 / 2)
    k3 = f(x + dt * k2 / 2)
    k4 = f(x + dt * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def rootfinder(f, interval, etol=1e-4, N=1000):
    """Finds roots in the interval=[a,b] for function f by the bisection method in
    N=1000 segments of the interval"""
    xs = np.linspace(interval[0], interval[1], N)
    left_bounds = []
    roots = []

    for i, x in enumerate(xs[:-1]):
        if f(x) * f(xs[i+1]) < 0:
            left_bounds.append(i)

    for i in left_bounds:
        x1, x2 = xs[i], xs[i+1]
        zero = f(xs[i])

        while abs(zero) > etol:
            m = (x1+x2)/2
            if f(x1) * f(m) < 0:
                x2 = m
                zero = f(x1)
            elif f(m) * f(x2) < 0:
                x1 = m
                zero = f(x1)

        roots.append(x1)

    return roots


@lru_cache(maxsize=None)
def dxdt(x: tuple, F: Callable[[float], float], G: Callable[[float, float], float], W: tuple):
    """The N-dimensional function determining the derivative of x."""
    f = np.zeros(len(x))
    for i, xi in enumerate(x):
        f[i] = F(xi)
        for j, xj in enumerate(x):
            f[i] += W[i][j] * G(xi, xj)
    return f


def is_eig_vec(vec, matrix):
    prod = matrix @ vec

    # Check for zeroes in vec
    for i in range(len(vec)):
        if vec[i] == 0 and prod[i] != 0:
            return False

    res = []
    for i in range(len(vec)):
        if vec[i] != 0:
            res.append(prod[i] / vec[i])
            if len(res) > 1 and np.absolute(res[-1] - res[-2]) > 1e-4:
                return False

    return True


def calculate_a1a2(c_arr, w_t, alpha1, vs):
    """Calculates the inner product between a1 and a2"""
    n = len(vs)  # dimension of reduction
    m = len(vs[0])  # dimension of vector
    a1 = np.sum([c_arr[i] * vs[i] for i in range(n)], axis=0) / np.sum([c_arr[i]*np.dot(np.ones(m), vs[i]) for i in range(n)])
    a2 = np.dot(w_t, a1) / alpha1
    a1_inner_a2 = np.dot(np.transpose(a1), a2)
    return np.abs(a1_inner_a2)


class Model:
    def __init__(self, W: np.ndarray, F: Callable[[float], float], G: Callable[[float, float], float]):
        self.W = W
        self.F = F
        self.G = G
        self.dim = self.W.shape[0]
        self.solution = 0

    def run(self, x0: np.ndarray, tf: float = 10):
        """Compute N iterations with initial condition x0, from t=0 to tf, with precision dt"""
        self.solution = solve_ivp(lambda t, x: dxdt(x, self.F, self.G, self.W), t_span=(0, tf), y0=x0)

    def randomStabilityAnalysis(self, p1p, p2p, dt, stabtol=1e-5):
        """Run an analysis of stable solutions with random initial vector for different input parameters"""
        p1s = np.linspace(p1p[0], p1p[1], p1p[2])  # tau
        p2s = np.linspace(p2p[0], p2p[1], p2p[2])  # mu

        results = []
        t1 = time.time()
        for p1 in p1s:
            for p2 in p2s:
                self.F = lambda x: -x
                self.G = lambda x, y: 1 / (1 + np.exp(p1*(p2-y))) - 1/(1+np.exp(p1*p2))

                x = np.random.rand(self.dim)*self.dim+self.dim

                func = lambda arr: dxdt(tuple(arr), self.F, self.G, tuple(tuple(row) for row in self.W))
                x = self.stabilityrun(func=func, dt=dt, stabtol=stabtol, x0=x, debugging=False)

                results.append([p1, p2, np.mean(x), np.linalg.norm(x)])

            if TALKING and __name__ == "__main__":
                print(f"Completed p1={round(p1,3)}. Total time taken: {round((time.time()-t1) // 60)} min and {round((time.time()-t1) % 60)} s")
                # t1 = time.time()

        results = np.transpose(results).tolist()
        results.insert(0, self.W.tolist())  # results = [w, p1, p2, mean(x), norm(x)]
        if __name__ == "__main__":
            with open('CowanWilsonstabilityresults'+str(time.time())+'.txt', 'w') as filehandle:
                json.dump(results, filehandle)

        return results

    def LaurenceOneDimAnalysisRun(self, x0=None, dt=0.1, stabtol=1e-3):
        # Definitions of One Dimensional Model
        eigs, vecs = np.linalg.eig(np.transpose(self.W))
        k = np.argmax(np.multiply(eigs, np.conjugate(eigs)))
        a = vecs[:, k]/(sum(vecs[:, k]))
        alpha = eigs[k]
        K = np.diag([sum(self.W[i, :]) for i in range(self.dim)])
        beta = a @ K @ np.transpose(a) / (a @ np.transpose(a) * alpha)

        alpha = real(alpha)
        beta = real(beta)

        # Stability run
        func = lambda arr: dxdt(tuple(arr), self.F, self.G, tuple(tuple(row) for row in self.W))
        x = self.stabilityrun(func, dt, stabtol, x0)

        simulation = real(a @ x)

        rooter = lambda R: self.F(R) + alpha * self.G(beta * R, R)
        # prediction = scipy.optimize.fsolve(rooter, [0, 20])
        prediction2 = rootfinder(rooter, [-1, 50])
        # print(f"{prediction} vs {prediction2}")

        return [alpha, simulation, prediction2, x]  # pass x to use as x0 next run (speeds up running time)

    def stabilityrun(self, func, dt, stabtol, x0, debugging=0, title="Stability") -> (list, float):
        """Calculates a network until stability has been reached. Debugging=1 full debug, =2 one plot.\n
        Returns the stable solution vector and the index at stability"""
        normF = 1
        x = x0

        if debugging > 0:
            xs = [np.copy(x)]

        i = 1
        while normF > stabtol and i < 100/dt:
            x += RK4step(dt, func, x)
            normF = np.linalg.norm(func(x))
            if debugging > 0:
                xs.append(np.copy(x))
                if debugging == 1 and i % (10/dt) == 0:  # plot every computed second
                    plt.plot([dt * n for n in range(len(xs))], xs)
                    plt.show()
                    time.sleep(1)
            i += 1

        if debugging > 0:
            plt.plot([dt * n for n in range(len(xs))], xs)
            plt.title(title)
            plt.show()

        return x, i

    def LaurenceMultiDimAnalysisRun(self, x0, dt=1e-2, stabtol=1e-3):
        """Returns simulation result, prediction with multidimensional reduction, list of a vectors and
        list of beta values used."""
        w_t = np.transpose(self.W)
        eigs, vecs = np.linalg.eig(w_t)

        # I/O and validation
        # print("First ten eigenvalues:", eigs[:10], "\n")
        print("Sorted normed eigenvalues:", np.sort(np.real(np.multiply(eigs, np.conj(eigs)))), "\n")
        n = 2  # input("How many eigenvalues are significant? (integer): ")
        try:
            n = int(n)
        except ValueError:
            print("That's not an integer!")
            self.LaurenceMultiDimAnalysisRun(x0=x0, dt=dt, stabtol=stabtol)

        # Get eigenvectors and K matrix
        vs = [[vecs[:, i], eigs[i]] for i in range(self.dim)]
        vs.sort(key=lambda item: np.conj(item[1])*item[1], reverse=True)
        vs = [item[0] for item in vs[:n]]

        cs = np.array([(-1)**i for i in range(n)])  # guess
        K = np.diag([sum(self.W[i, :]) for i in range(self.dim)])

        # Find c by minimizing the inner product between a1 and a2
        res = minimize(lambda c_arr: calculate_a1a2(c_arr, w_t, eigs, vs), cs)
        c = res.x
        print(c)

        # Use minimizing c to calculate a vectors
        a = [np.sum([c[i] * vs[i] for i in range(n)], axis=0) /
             np.sum([c[i]*np.dot(np.ones(self.dim), vs[i]) for i in range(n)])]
        beta = [a[0] @ K @ np.transpose(a[0]) / (a[0] @ np.transpose(a[0]) * eigs[0])]
        for i in range(1, n):
            a.append(np.dot(w_t, a[-1])/eigs[i-1])
            beta.append(a[-1] @ K @ np.transpose(a[-1]) / (a[-1] @ np.transpose(a[-1]) * eigs[i]))

        # Let's integrate the reduced ODE!
        func = lambda Rs: np.array([self.F(Rs[i]) + eigs[i]*self.G(beta[i]*Rs[i], Rs[i+1]) for i in range(n-1)] +
                                   [self.F(Rs[-1]) + eigs[-1]*self.G(beta[-1]*Rs[-1], Rs[0])])
        prediction = self.stabilityrun(func, dt, stabtol, np.array([a[i] @ x0 for i in range(n)]), debugging=2)[0]

        # Integrate the original ODE
        func = lambda arr: dxdt(tuple(arr), self.F, self.G, tuple(tuple(row) for row in self.W))
        x = self.stabilityrun(func, dt, stabtol, x0, debugging=2)[0]
        simulation_res = [a[i] @ x for i in range(n)]

        return [simulation_res, prediction, a, beta]


if __name__ == "__main__":
    #-- LAURENCE ND ANALYSIS --#
    w = np.array([[0.52829123, -0.12620005, 0.07649462, -0.51603925, 0.3324112],
                   [-0.65911129, -0.2884593, 0.35157307, -0.61664797, -0.47923406],
                   [-0.95658735, -0.99555071, 0.61247847, 0.57499545, -0.185355],
                   [-0.63192054, -0.36443042, 0.76658192, -0.94645962, 0.85193479],
                   [-0.41915602, -0.02816688, -0.39475975, -0.48100525, 0.8211069]])

    f = lambda x: -x**3
    g = lambda x, y: (1-x)*y

    model = Model(w, f, g)

    res = model.LaurenceMultiDimAnalysisRun(x0=2*np.random.rand(w.shape[0]))
    for r in res[:2]:
        print(r)



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








