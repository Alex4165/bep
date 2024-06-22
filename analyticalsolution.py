import numpy as np
from scipy.optimize import fsolve
from sympy import Symbol, I, nsolve, E

dimension = 1000
A = np.random.rand(dimension, dimension)


def self(x, B=1):
    # return -x
    return -B*x


def interaction(x, y, h=2, tau=1, mu=3):
    # return 1/(1+np.exp(tau*(mu-y)))
    return y**h/(1+y**h)


def function(x):
    y = []
    for i in range(dimension):
        me = x[i]
        out = self(me)
        for j in range(dimension):
            out += A[i, j] * interaction(me, x[j])
        y.append(out)
    return y


u = Symbol('u')
v = Symbol('v')
alpha = 1 + 1j
f1 = alpha / (1 + E**(3-u-I*v))  # alpha / (1 + E**(3-u-I*v))
f2 = 0
print(nsolve((f1, f2), (u, v), (100*I, 100*I)))

