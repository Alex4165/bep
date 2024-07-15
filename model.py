import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root
import scipy.io
import json
import time
import networkx as nx
import cProfile
from functools import lru_cache
from dataclasses import dataclass, field

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


def RK4step(dt, func, x):
    """Returns one Runge-Kutta 4 integration step
     Note:  assumes f is independent of time!
            returns dx, not x+dx"""
    k1 = func(x)
    k2 = func(x + dt * k1 / 2)
    k3 = func(x + dt * k2 / 2)
    k4 = func(x + dt * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def rootfinder(f, interval, etol=1e-4, N=1000):
    """Finds roots in the interval=[a,b] for function f by the bisection method
    after splitting the interval into N segments of the interval"""
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
def dx_dt(x: tuple, F: Callable[[float], float], G: Callable[[float, float], float], W: tuple):
    """The N-dimensional function determining the derivative of x."""
    f = np.zeros(len(x))
    for i, xi in enumerate(x):
        f[i] = F(xi)
        f[i] += np.sum([W[i][j] * G(x[i], x[j]) for j in range(len(x))])
    return f


@lru_cache(maxsize=None)
def exp(x):
    return np.exp(x)


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
    a1 = np.sum([c_arr[i] * vs[i] for i in range(n)], axis=0) / np.sum([c_arr[i]*(np.ones(m) @ vs[i]) for i in range(n)])
    a2 = (w_t @ a1) / alpha1
    a1_inner_a2 = np.transpose(a1) @ a2
    return np.abs(a1_inner_a2)


def stability_run(func: Callable[[List[float]], List[float]], dt, stabtol, x0, debugging=0, title="Stability", max_run_time=50) -> (list, float):
    """Calculates a network until stability has been reached. \n
    If stabtol is set to None, then the first 50 seconds are calculated regardless.\n
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
        if debugging > 0:
            if debugging == 1 and i % (10/dt) == 0:  # plot every number of computed seconds
                plt.plot([dt * n for n in range(len(xs))], xs)
                plt.show()
                time.sleep(1)
        i += 1

    if debugging > 0:
        plot_solution(dt, title, xs)

    return x, i, xs


def plot_solution(dt, title, xs, saving=False):
    if True in np.iscomplex(xs):
        plt.plot([dt * n for n in range(len(xs))], np.abs(xs))
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Magnitude")
        plt.show()

        plt.plot([dt * n for n in range(len(xs))], np.angle(xs))
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (radians)")
        plt.show()
    else:
        plt.plot([dt * n for n in range(len(xs))], xs)
        plt.xlabel("Time (s)")
        plt.ylabel("Activity")
        if not saving:
            plt.title(title)
        else:
            plt.savefig("data/"+title+".pdf")

        plt.show()


def get_cached_performance():
    """Prints the cache hit ratio for each cached function"""
    cached_functions = [dx_dt, exp]
    for func in cached_functions:
        ci = func.cache_info()
        if ci.hits+ci.misses > 0:
            print(f"{func.__name__} cache hits: {round(100 * ci.hits / (ci.hits + ci.misses))}%")
        else:
            print(f"{func.__name__} was never called")


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


class Network:
    def __init__(self):
        self.adj_matrix = None
        self.graph = None
        self.size = 0

    def gen_random(self, size, p):
        """Generate an Erdos-Renyi random graph with connection probability p."""
        self.graph = nx.erdos_renyi_graph(size, p)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.size = self.graph.number_of_nodes()

    def gen_small_world(self, size, k, p):
        """Generate a Watts-Strogatz random graph.
        k is the max distance of nearest neighbors first connected to.
        p is the rewiring probability."""
        self.graph = nx.watts_strogatz_graph(size, k, p)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.size = self.graph.number_of_nodes()

    def gen_community(self, sizes: List[int], probabilities: List[List[float]] = None):
        """Generate a community graph.
        sizes is a list of the size of each community.
        probabilities is a list of lists of probabilities such that (r,s) gives the
        probability of community r attaching to s. If None, random values are taken."""
        if probabilities is None:
            probabilities = [[np.random.rand() for s in range(len(sizes))] for r in range(len(sizes))]
        self.graph = nx.stochastic_block_model(sizes=sizes, p=probabilities, directed=True)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.size = self.graph.number_of_nodes()

    def gen_hub(self, size, minimum_deg):
        """Generate Barabasi-Albert graph which attaches new nodes preferentially to high degree nodes.
        Creates a hub structure with predefined minimum node degree."""
        self.graph = nx.barabasi_albert_graph(size, minimum_deg)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.size = self.graph.number_of_nodes()

    def plot_graph(self, title=" ", weighted=False, saving=False):
        """Plot graph"""
        fig, ax = plt.subplots()
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        pos = nx.spring_layout(self.graph, k=1 / self.size ** 0.1)
        if weighted:
            for u, v, d in self.graph.edges(data=True):
                d['weight'] = self.adj_matrix[u, v]

            edges, weights = zip(*nx.get_edge_attributes(self.graph, 'weight').items())

            nx.draw(self.graph, pos, node_color='b', edgelist=edges, edge_color=weights,
                    width=3, edge_cmap=plt.cm.Blues)
        else:
            nx.draw_networkx(self.graph, pos, with_labels=False, ax=ax)
        if not saving:
            plt.title(title)
        else:
            plt.savefig("data/"+title+".pdf")
        plt.show()

    def randomize_weights(self, factor=lambda x: 2*x-1):
        """Reweight the network with factor a function of a uniform random variable between 0 and 1."""
        if self.adj_matrix is not None:
            random_adjustments = factor(np.random.random_sample((self.size, self.size)))
            self.adj_matrix = np.multiply(self.adj_matrix, random_adjustments)
        else:
            raise ValueError("No network to speak of!")


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
        func = lambda arr: dx_dt(tuple(arr), self.decay_func, self.interact_func, tuple(tuple(row) for row in self.w))
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

    def LaurenceOneDimAnalysisRun(self, x0, dt=0.1, stabtol=1e-3):
        # Definitions of One Dimensional Model
        eigs, vecs = np.linalg.eig(np.transpose(self.w))
        k = np.argmax(np.multiply(eigs, np.conjugate(eigs)))
        a = vecs[:, k]/(sum(vecs[:, k]))
        alpha = eigs[k]
        K = np.diag([sum(self.w[i, :]) for i in range(self.dim)])
        beta = a @ K @ np.transpose(a) / (a @ np.transpose(a) * alpha)

        alpha = real(alpha)
        beta = real(beta)

        # Stability run
        func = lambda arr: dx_dt(tuple(arr), self.decay_func, self.interact_func, tuple(tuple(row) for row in self.w))
        x = stability_run(func, dt, stabtol, x0)[0]

        simulation = real(a @ x)

        rooter = lambda R: self.decay_func(R) + alpha * self.interact_func(beta * R, R)
        numeric_R = stability_run(rooter, dt, stabtol, a @ x0, debugging=2)[0]
        print("numeric 1d", numeric_R)
        # prediction = scipy.optimize.fsolve(rooter, [0, 20])
        prediction2 = rootfinder(rooter, [-1, 50])
        # print(f"{prediction} vs {prediction2}")

        return [alpha, simulation, prediction2, x]  # pass x to use as x0 next run (speeds up running time)

    def laurence_multi_dim_analysis_run(self, x0, dt=1e-2, stabtol=1e-3, max_run_time=50):
        """Returns simulation result, prediction with multidimensional reduction, list of 'a' vectors, alphas and
        beta values used (in order of eigenvalue norm descending)."""
        w_t = np.transpose(self.w)
        eigs, vecs = np.linalg.eig(w_t)

        # I/O and validation
        # print("First ten eigenvalues:", eigs[:10], "\n")
        print("Sorted, normed eigenvalues:", np.sort(np.real(np.abs(eigs))), "\n")
        n = input("How many eigenvalues are significant? (integer): ")
        try:
            n = int(n)
        except ValueError:
            print("That's not an integer!")
            return self.laurence_multi_dim_analysis_run(x0=x0, dt=dt, stabtol=stabtol, max_run_time=max_run_time)

        # Get sorted eigenvalues, -vectors and K matrix
        arranged = [[vecs[:, i], eigs[i]] for i in range(self.dim)]
        arranged.sort(key=lambda item: np.conj(item[1])*item[1], reverse=True)
        vs = [item[0] for item in arranged[:n]]
        alphas = [item[1] for item in arranged[:n]]
        cs = np.array([(0.5)**i for i in range(n)])  # guess
        K = np.diag([sum(self.w[i, :]) for i in range(self.dim)])

        # Find 'c' vector by minimizing the inner product between a1 and a2
        res = minimize(lambda c_arr: calculate_a1a2(c_arr=c_arr, w_t=w_t, alpha1=alphas[0], vs=vs), cs)
        c = res.x
        print(f'Chosen c array = {c}: {res.message} Eigenvector overlap = {calculate_a1a2(c, w_t, alphas[0], vs)}')

        # Use the minimizing 'c' vector to calculate 'a' vectors and betas
        a = [np.sum([c[i] * vs[i] for i in range(n)], axis=0) /
             np.sum([c[i]*np.dot(np.ones(self.dim), vs[i]) for i in range(n)])]
        betas = [a[0] @ K @ np.transpose(a[0]) / (a[0] @ np.transpose(a[0]) * alphas[0])]
        for i in range(1, n):
            a.append(np.dot(w_t, a[-1])/alphas[i-1])
            betas.append(a[-1] @ K @ np.transpose(a[-1]) / (a[-1] @ np.transpose(a[-1]) * alphas[i]))
        print('alphas', alphas, '\nbetas', betas)

        # Let's integrate the reduced ODE!
        func = lambda Rs: np.array([self.decay_func(Rs[i]) + alphas[i] *
                                    self.interact_func(betas[i] * Rs[i], Rs[i + 1]) for i in range(n - 1)] +
                                   [self.decay_func(Rs[-1]) + alphas[-1] * self.interact_func(betas[-1] * Rs[-1], Rs[0])])
        prediction = stability_run(func, dt, stabtol, np.array([a[i] @ x0 for i in range(n)]),
                                   title=f"{n}D reduction", debugging=2, max_run_time=max_run_time)[0]

        # Integrate the original ODE
        func = lambda arr: dx_dt(tuple(arr), self.decay_func, self.interact_func, tuple(tuple(row) for row in self.w))
        x = stability_run(func, dt, stabtol, x0, title="Full simulation", debugging=2, max_run_time=max_run_time)[0]
        simulation_res = [a[i] @ x for i in range(n)]

        dec = 3
        print(f"Error (act: {np.round(simulation_res, decimals=dec)} pred: {np.round(prediction, decimals=dec)}) = "
              f"{np.round(np.linalg.norm(simulation_res-prediction), decimals=dec)}")

        return [simulation_res, prediction, a, alphas, betas]


if __name__ == "__main__":
    net = Network()
    net.gen_community([10, 10, 10])
    net.randomize_weights()

    const = 1/(1-exp(3))


    def decay(x):
        return -x


    def interact(x, y):
        return 1/(1+exp(3-y)) - const


    w = net.adj_matrix
    model = Model(w, decay, interact)

    x0 = 5 * np.random.rand(w.shape[0]) + 5
    dt = 1e-2
    stabtol = 1e-5

    res = model.laurence_multi_dim_analysis_run(dt=dt, stabtol=stabtol, x0=x0)

    res = model.laurence_multi_dim_analysis_run(dt=dt, stabtol=stabtol, x0=x0)

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








