import time

import numpy as np
import matplotlib.pyplot as plt
from auxiliary_functions import exp, rootfinder, format_time_elapsed
from scipy.optimize import fsolve

tau = 2
mu = 1
theta = exp(-tau * mu)
alpha = 10
gamma = 0

umin, umax = -0.1, 1
us = np.linspace(umin, umax, 5000)
zs = np.linspace(umin, umax, 1000)
du = us[1]-us[0]
dp = 7e-3

real_isocline_points = []
imag_isocline_points = []
real_positive_points = []
imag_positive_points = []
real_zero_pos = []
imag_zero_pos = []


def Z(x):
    return (1+theta)/(1+np.exp(tau*(mu-x))) - theta


def deriv(R):
    return -R + Z(alpha*R + gamma)


start = time.time()
for u in us:
    real_roots = rootfinder(lambda v: np.real(deriv(u+1j*v)), [umin, umax], N=100, etol=0.5*du)
    imag_roots = rootfinder(lambda v: np.imag(deriv(u+1j*v)), [umin, umax], N=100, etol=0.5*du)
    for root in real_roots:
        real_isocline_points.append([u, root])
        if np.real(deriv(u+dp+1j*(root+dp))) > 0:
            real_positive_points.append([u+dp, root+dp])
        if np.real(deriv(u-dp+1j*(root-dp))) > 0:
            real_positive_points.append([u-dp, root-dp])
    for root in imag_roots:
        imag_isocline_points.append([u, root])
        if np.imag(deriv(u+dp+1j*(root+dp))) > 0:
            imag_positive_points.append([u+dp, root+dp])
        if np.imag(deriv(u-dp+1j*(root-dp))) > 0:
            imag_positive_points.append([u-dp, root-dp])

for u in zs:
    if np.real(deriv(0+dp+1j*(u+dp))) > 0:
        real_zero_pos.append([0+dp, u+dp])
    if np.real(deriv(0-dp+1j*(u-dp))) > 0:
        real_zero_pos.append([0-dp, u-dp])
    if np.imag(deriv(u+dp+1j*(0+dp))) > 0:
        imag_zero_pos.append([u+dp, 0+dp])
    if np.imag(deriv(u-dp+1j*(0-dp))) > 0:
        imag_zero_pos.append([u-dp, 0-dp])

real_isocline_points = np.array(real_isocline_points).T
imag_isocline_points = np.array(imag_isocline_points).T
real_positive_points = np.array(real_positive_points).T
imag_positive_points = np.array(imag_positive_points).T
real_zero_pos = np.array(real_zero_pos).T
imag_zero_pos = np.array(imag_zero_pos).T


plt.figure(figsize=(6, 5), dpi=600)
plt.plot(real_isocline_points[0], real_isocline_points[1], 'r.', markersize=3, label='Real Isocline')
plt.plot([0 for _ in range(len(zs))], zs, 'r.', markersize=3)
plt.plot(imag_isocline_points[0], imag_isocline_points[1], 'b.', markersize=3, label='Imag. Isocline')
plt.plot(zs, [0 for _ in range(len(zs))], 'b.', markersize=3)
plt.plot(real_positive_points[0], real_positive_points[1], 'k.', markersize=1, label='Positive Side')
plt.plot(imag_positive_points[0], imag_positive_points[1], 'k.', markersize=1)
plt.plot(real_zero_pos[0], real_zero_pos[1], 'k.', markersize=1)
plt.plot(imag_zero_pos[0], imag_zero_pos[1], 'k.', markersize=1)

plt.xlabel('u')
plt.ylabel('v')
plt.legend()
plt.savefig('data/fig5.4.png')
plt.show()

print("Took", format_time_elapsed(time.time()-start))

