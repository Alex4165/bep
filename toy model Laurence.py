import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.integrate import solve_ivp


class Model:
    def __init__(self, W: np.ndarray, F: Callable[[float], float], G: Callable[[float, float], float]):
        self.W = W
        self.F = F
        self.G = G
        self.dim = self.W.shape[0]
        self.solution = 0

    def f(self, t: float, x: np.ndarray):
        """The N-dimensional function determining the derivative of x"""
        f = np.zeros(self.dim)
        for i, xi in enumerate(x):
            f[i] = self.F(xi)
            for j, xj in enumerate(x):
                f[i] += self.W[i, j] * self.G(xi, xj)
        return f

    def run(self, x0: np.ndarray, tf: float = 10, dt: float = 0.1):
        """Compute N iterations with initial condition"""
        self.solution = solve_ivp(self.f, t_span=(0, tf), y0=x0, t_eval=np.arange(0, tf, dt))



dim = 10

W = (np.random.rand(dim, dim)-0.5*np.ones((dim, dim)))*10
eig = np.linalg.eig(W)
print("Eigenvalues of W:", eig[0], "\n")
print("Norm squared eigenvalues of W:", np.multiply(np.conj(eig[0]), eig[0]), "\n")
F = lambda x: -x
G = lambda x, y: 1/(1+np.exp(-y+3))
model = Model(W, F, G)
model.run(np.random.rand(dim), 100)

plt.plot(model.solution.t, np.transpose(model.solution.y))
plt.show()








