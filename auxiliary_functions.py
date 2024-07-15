import time
from functools import lru_cache
from typing import Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt


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


def rootfinder(f, interval, etol=1e-2, N=1000, do_initial_search=True, speak=False):
    """Finds roots in the interval=[a,b] for function f by the bisection method
    after splitting the interval into N segments of the interval"""
    if speak:
        print("Hi! I'm gonna do my best for you :)")
    roots = []

    if do_initial_search:
        xs = np.linspace(interval[0], interval[1], N)
        left_bounds = []

        t0 = time.time()
        t1 = t0
        for i, x in enumerate(xs[:-1]):
            if time.time() - t1 > 30 and speak:
                print(f"Still working on it! {round(100*(i+1)/len(xs))}% done. "
                      f"Takes {(time.time()-t0)/(2*(i+1))} s per call so "
                      f"around {(time.time()-t0)/(i+1)*(len(xs)-i-1)} s left.")
                t1 = time.time()
            if f(x) * f(xs[i+1]) < 0:
                left_bounds.append(i)

        if speak:
            print(f"Took me {time.time()-t0} seconds to find the {len(left_bounds)} locations of zeroes.")
    else:
        left_bounds = [0]
        xs = interval

    for i in left_bounds:
        x1, x2 = xs[i], xs[i+1]
        zero = f(xs[i])

        while abs(zero) > etol:
            if speak:
                print(f"my best guess f({(x1+x2)/2})={zero}")
                print(f"my other guesses: {f(x1)} {f(x2)}")
                print(f"my accuracy: {x2-x1}\n")

            m = (x1+x2)/2
            if f(x1) * f(m) < 0:
                x2 = m
                zero = f(x1)
            elif f(m) * f(x2) < 0:
                x1 = m
                zero = f(x1)
            else:
                # Couldn't find the zero crossing
                left_zeroes = rootfinder(f, [x1, m], etol, N, do_initial_search, speak)
                right_zeroes = rootfinder(f, [m, x2], etol, N, do_initial_search, speak)
                return left_zeroes + right_zeroes

        roots.append(x1)

    return roots


@lru_cache(maxsize=None)
def exp(x):
    return np.exp(x)


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
            print(f"{func.__name__} cache hits: {round(100 * ci.hits / (ci.hits + ci.misses))}%")
        else:
            print(f"{func.__name__} was never called")


@lru_cache(maxsize=None)
def dx_dt(x: tuple, F: Callable[[float], float], G: Callable[[float, float], float], W: tuple):
    """The N-dimensional function determining the derivative of x."""
    f = np.zeros(len(x))
    for i, xi in enumerate(x):
        f[i] = F(xi)
        f[i] += np.sum([W[i][j] * G(x[i], x[j]) for j in range(len(x))])
    return f


def format_time_elapsed(start_time):
    return f"{round((time.time()-start_time) // 60)} min and {round((time.time()-start_time) % 60)} s"


@lru_cache(maxsize=None, typed=False)
def find_lambda_star(pos_decay_func, pos_interaction_func, parameters: Tuple[float],
                     lower_bound=0.0, upper_bound=1e4, accuracy=1e-3):
    """pos_decay_func and pos_interaction_func must have signature (x, *parameters) -> float and
    (x, y, *parameters) -> float"""
    guess = (lower_bound + upper_bound) / 2

    if upper_bound - lower_bound < accuracy:
        return guess

    f = lambda x: pos_decay_func(x, *parameters) + guess * pos_interaction_func(x, x, *parameters)

    is_nonzero_root = False
    xs = np.linspace(-1e-3, upper_bound, max(1000, int(10*upper_bound)))
    for i in range(len(xs) - 1):
        if f(xs[i]) * f(xs[i+1]) < 0 and xs[i] > accuracy:
            is_nonzero_root = True
            break

    if is_nonzero_root:
        return find_lambda_star(pos_decay_func, pos_interaction_func, parameters, lower_bound, guess)
    else:
        return find_lambda_star(pos_decay_func, pos_interaction_func, parameters, guess, upper_bound)


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


if __name__ == "__main__":
    print(find_lambda_star(lambda x, tau, mu: -x,
                           lambda x, y, tau, mu: 1/(1+np.exp(tau*(mu-y))) - 1/(1+np.exp(tau*mu)),
                           (1, 1)))



