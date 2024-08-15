import time

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from auxiliary_functions import find_lambda_star, get_cached_performance, format_time_elapsed, exp

start = time.time()

taus, lstars = [1/n for n in range(100, 4, -1)]+[1/5 + 0.1*i for i in range(25)], []
for tau in taus:
    lstars.append(find_lambda_star((tau, 1)))
taus = np.array(taus)
lstars = np.array(lstars)


def f(x, a, b, c):
    return a * x ** b + c


popt, pcov = curve_fit(f, taus, lstars, p0=[2, -1, 1])
print(popt)

plt.plot(taus, lstars, label='Data')
plt.plot(taus, f(taus, *popt), label='Fit')
plt.xlabel("Tau")
plt.ylabel("Lambda star")
plt.legend()
plt.show()

plt.plot(taus, lstars-f(taus, *popt), '.')
plt.show()

print("Done that took: ", format_time_elapsed(time.time()-start))
get_cached_performance()



