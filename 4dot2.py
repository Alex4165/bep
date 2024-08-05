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
# This is the exact same code as 3dot3.py, but using a noncooperative network instead.







