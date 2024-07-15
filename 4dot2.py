import numpy as np
import matplotlib.pyplot as plt
from model import Model
from network_class import Network
from auxiliary_functions import rootfinder, rho, kmax
from concurrent.futures import ThreadPoolExecutor
import os
from typing import List
from functools import lru_cache

# --- Section 4.2: How badly does Wu predict the collapse of noncooperative networks? --- #

# Parameters
sizes = [5, 10, 15]
num_trials = 1
dt = 1e-1
stabtol = 1e-4
max_run_time = 1e3
print("CPU Count:", os.cpu_count())


@lru_cache(maxsize=None, typed=False)
def find_lambda_star(pos_decay_func, pos_interaction_func, parameters: List[float],
                     lower_bound=0.0, upper_bound=1e4, accuracy=1e-3):
    """pos_decay_func and pos_interaction_func must have signature (x, *parameters) -> float and
    (x, y, *parameters) -> float"""
    guess = (lower_bound + upper_bound) / 2

    if upper_bound - lower_bound < accuracy:
        return guess

    f = lambda x: pos_decay_func(x, *parameters) + guess * pos_interaction_func(x, x, *parameters)
    zeros = rootfinder(f, [-0.1, upper_bound], etol=accuracy)
    if len([zero for zero in zeros if zero > accuracy]) > 0:
        return find_lambda_star(pos_decay_func, pos_interaction_func, parameters, lower_bound, guess)
    else:
        return find_lambda_star(pos_decay_func, pos_interaction_func, parameters, guess, upper_bound)


def run_test(network, decay_func, interaction_func, parameter_1_range, parameter_2_range):
    # Initialize the system
    model = Model(network.adj_matrix, decay_func, interaction_func)
    res = model.random_stability_analysis(p1p=parameter_1_range, p2p=parameter_2_range, dt=dt, stabtol=stabtol,
                                          max_run_time=max_run_time)
    km = kmax(network.adj_matrix)
    ro = rho(network.adj_matrix)

    # Find the lambda stars
    lambda_stars = [find_lambda_star(decay_func, interaction_func,
                                     [res.parameter1_list[i], res.parameter2_list[i]])
                    for i in range(len(res.parameter1_list))]





