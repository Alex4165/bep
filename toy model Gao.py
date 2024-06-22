import numpy as np
import matplotlib.pyplot as plt

# dxi/dt = F(xi) + Sum Aij G(xi, xj)


def F(x):
    """Self-interaction function"""
    return -1


def G(x, y):
    """Co-interaction function"""
    try:    # (x+y)/2
        return (x+y)/(x+y+100)
    except ZeroDivisionError:
        print("whoopsie")
        return 0


def L(arr):
    """Nearest neighbor average"""
    n = len(arr)
    num = np.dot(np.dot(np.ones(n), A), arr)
    den = np.dot(np.dot(np.ones(n), A), np.ones(n))
    return num/den


            # evil, friendly, neutral
# A = np.array([[0, 1, 0.5],
#               [-1, 0, 0.5],
#               [-1, 1, 0]])

dim = 10

A = np.random.rand(dim, dim)
A -= 0.5*np.ones((dim, dim))
A *= 2

xs = [[np.random.randint(-10,10) for i in range(dim)]]
xeffs = [L(xs[-1])]
xeffsapprox = [L(xs[-1])]
N = 100000
dt = 0.0001


for i in range(N):
    xn = []
    for j in range(dim):
        total = F(xs[-1][j])
        for k in range(dim):
            total += A[j, k] * G(xs[-1][j], xs[-1][k])
        xn.append(xs[-1][j] + total * dt)
    xs.append(xn)
    xeffs.append(L(xs[-1]))
    xe = xeffsapprox[-1]
    dxe = (F(xe) + L(np.dot(A, np.ones(dim))) * G(xe, xe)) * dt
    xeffsapprox.append(xe+dxe)


for j in range(dim):
    plt.plot([arr[j] for arr in xs])
plt.plot(xeffs, label="x_eff actual")
plt.plot(xeffsapprox, label="x_eff approx")
plt.legend()
plt.show()






