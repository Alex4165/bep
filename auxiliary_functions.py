import time
from functools import lru_cache
from typing import Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt


def real(x):
    if np.isreal(x.all()):
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


def rootfinder(f, interval, etol=1e-2, N=1000, do_initial_search=True, speak=False):
    """Finds roots in the interval=[a,b] for function f by the bisection method
    after splitting the interval into N segments of the interval"""
    roots = []

    if do_initial_search:
        xs = np.linspace(interval[0], interval[1], N)
        left_bounds = []

        y1 = f(xs[0])
        for i in range(1, len(xs)):
            y2 = f(xs[i])
            if y1 * y2 < 0:
                left_bounds.append(i-1)
            y1 = y2

    else:
        left_bounds = [0]
        xs = interval

    for i in left_bounds:
        x1, x2 = xs[i], xs[i+1]
        y1, y2 = f(x1), f(x2)

        best_zero = f(x1)
        best_guess = x1

        count = 0
        while abs(best_zero) > etol:
            m = (x1+x2)/2
            ym = f(m)
            if y1 * ym < 0:
                x2 = m
                y2 = ym
            elif ym * y2 < 0:
                x1 = m
                y1 = ym

            # Computationally inexpensive way to find the best zero earlier
            k = np.argmin([np.abs(y1), np.abs(ym), np.abs(y2)])
            best_zero = [y1, ym, y2][k]
            best_guess = [x1, m, x2][k]
            count += 1

        roots.append(best_guess)
    return roots


@lru_cache()
def exp(x):
    return np.exp(x)


def plot_solution(dt, title, xs, saving=False):
    if True in np.iscomplex(xs):
        plt.plot([dt * n for n in range(len(xs))], np.abs(xs))
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Magnitude")
        if not saving:
            plt.title(title)
            plt.show()

            # plt.plot([dt * n for n in range(len(xs))], np.angle(xs))
            # plt.title(title)
            # plt.xlabel("Time (s)")
            # plt.ylabel("Phase (radians)")
            # plt.show()
        else:
            # We only save the magnitude plot
            plt.savefig("data/"+title+".png")
            plt.clf()
    else:
        plt.plot([dt * n for n in range(len(xs))], xs)
        plt.xlabel("Time (s)")
        plt.ylabel("Activity")
        if not saving:
            plt.title(title)
            plt.show()
        else:
            plt.savefig("data/"+title+".png")
            plt.clf()


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


def get_cached_performance():
    """Prints the cache hit ratio for each cached function"""
    cached_functions = [exp, dx_dt, find_lambda_star]
    for func in cached_functions:
        ci = func.cache_info()
        if ci.hits+ci.misses > 0:
            print(f"{func.__name__} cache hits: {round(100 * ci.hits / (ci.hits + ci.misses))}%"
                  f" ({ci.hits+ci.misses} calls)")
        else:
            print(f"{func.__name__} was never called")
    print()


@lru_cache(maxsize=1024)
def dx_dt(x: tuple, F: Callable[[float], float], G: Callable[[float, float], float], W: tuple):
    """The N-dimensional function determining the derivative of x."""
    f = np.zeros(len(x))
    for i, xi in enumerate(x):
        f[i] = F(xi)
        f[i] += np.sum([W[i][j] * G(x[i], x[j]) for j in range(len(x))])
    return f


def format_time_elapsed(elapsed_time):
    if elapsed_time < 60:
        return f"{round(elapsed_time, 3)} s"
    if elapsed_time < 3600:
        return (f"{round(elapsed_time // 60)} min "
                f"{round(elapsed_time % 60)} s")
    if elapsed_time < 86400:
        return (f"{round(elapsed_time // 3600)} h "
                f"{round((elapsed_time % 3600) // 60)} min "
                f"{round(elapsed_time % 60)} s")
    return (f"{round(elapsed_time // 86400)} d "
            f"{round((elapsed_time % 86400) // 3600)} h "
            f"{round((elapsed_time % 3600) // 60)} min "
            f"{round(elapsed_time % 60)} s")


@lru_cache(maxsize=30000, typed=False)
def find_lambda_star(parameters: Tuple[float, float],
                     lower_bound=0.0, upper_bound=1e4, accuracy=1e-3):
    """Decay and interaction function are hardcoded!"""
    def z(x, tau, mu): return (1/(1+exp(tau*(mu-x))) - 1/(1+exp(tau*mu)))/(1 - 1/(1+exp(tau*mu)))

    guess = (lower_bound + upper_bound) / 2

    if upper_bound - lower_bound < accuracy:
        return guess

    def f(x): return -x + z(guess * x, *parameters)

    is_nonzero_root = False
    xs = np.linspace(-1e-3, upper_bound, max(1000, int(10*upper_bound)))
    for i in range(len(xs) - 1):
        if f(xs[i]) * f(xs[i+1]) < 0 and xs[i] > accuracy:
            is_nonzero_root = True
            break

    if is_nonzero_root:
        return find_lambda_star(parameters, lower_bound=lower_bound, upper_bound=guess)
    else:
        return find_lambda_star(parameters, lower_bound=guess, upper_bound=upper_bound)


def rho(A):
    eigs, vecs = np.linalg.eig(A)
    i = np.argmax(np.array([np.real(e) for e in eigs]))
    return eigs[i]


def kmax(A):
    kmax = 0
    for i in range(A.shape[0]):
        degs = np.transpose(np.dot(A, np.ones(A.shape[0])))
        i = np.argmin(degs)
        if degs[i] >= kmax:
            kmax = degs[i]
        A = np.delete(A, i, 0)
        A = np.delete(A, i, 1)
    return kmax





