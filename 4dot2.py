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
# This is the exact same code as threedotthree.py, but using a noncooperative network instead.


def decay(x): return -x**3 + 3*x**2-2.5*x
def interact(x, y): return 0.5*y


net = Network()
net.gen_random(size=10, p=0.3)
net.randomize_weights(lambda x: x+0.5)
A = net.adj_matrix
model = Model(W=A, F=decay, G=interact)
print(model.stability_run(0.01, 1e-4, np.random.random_sample(10), debugging=2))







