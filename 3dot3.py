import time

import numpy as np
from matplotlib import pyplot as plt

from model import stability_run
from auxiliary_functions import exp, dx_dt, format_time_elapsed, find_lambda_star, rootfinder, kmax, rho, \
    get_cached_performance
from network_class import Network

from concurrent.futures import ThreadPoolExecutor

# --- Section 3.3: Wu Rigorous bounds applied --- #

# Parameters
sizes = [5, 10, 15]
network_types = [Network.gen_random]
runs_per_size = 5
runs_per_network = 5
dt = 1e-1
stabtol = 1e-4


# Dynamic equations (we assume all model have the same dynamic equations)
def decay(x):
    return -x


def interact(x, y, tau, mu):
    return 1 / (1 + exp(tau*(mu - y))) - 1 / (1 + exp(tau*mu))


def find_critical_points(network_generator: str, network_params: dict, p1_range, p2_error_tolerance=1e-1, zero_tol=1e-1):
    # Initialize the network
    net = Network()
    generator = getattr(net, network_generator)
    generator(**network_params)
    net.randomize_weights(lambda x: x+1)

    # Find the critical points, the points where the system goes from surviving to collapsing (in terms of p2)
    critical_points = []
    for p1 in p1_range:
        p2_low, p2_high = 0, 10

        # First check whether there are any critical points in this range
        # Also we define x_low and x_high here to avoid repeating the same code
        x0 = net.size * np.random.random_sample(net.size)
        derivative = lambda arr, p2: dx_dt(tuple(arr), decay, lambda x, y: interact(x, y, p1, p2),
                                           tuple(tuple(row) for row in net.adj_matrix))
        x_low = stability_run(lambda a: derivative(a, p2_low), dt, stabtol, x0)[0]
        x_high = stability_run(lambda a: derivative(a, p2_high), dt, stabtol, x0)[0]
        if ((np.mean(x_low) > zero_tol and np.mean(x_high) > zero_tol) or
                (np.mean(x_low) < zero_tol and np.mean(x_high) < zero_tol)):
            ValueError(f"No critical points in this parameter 2 range: {p2_low}-{p2_high}")

        while p2_high - p2_low > p2_error_tolerance:
            p2_middle = (p2_low + p2_high) / 2

            x0 = net.size * np.random.random_sample(net.size)
            x_mid = stability_run(lambda a: derivative(a, p2_middle), dt, stabtol, x0)[0]

            # Run all three on different processors (ideally at the same time)
            # with ThreadPoolExecutor() as exc:
            #     x_low_fut = exc.submit(lambda: stability_run(lambda a: derivative(a, p2_low), dt, stabtol, x0)[0])
            #     x_mid_fut = exc.submit(lambda: stability_run(lambda a: derivative(a, p2_middle), dt, stabtol, x0)[0])
            #     x_high_fut = exc.submit(lambda: stability_run(lambda a: derivative(a, p2_high), dt, stabtol, x0)[0])
            # x_low, x_mid, x_high = x_low_fut.result(), x_mid_fut.result(), x_high_fut.result()

            if (np.mean(x_mid) - zero_tol) * (np.mean(x_low) - zero_tol) > 0:
                # If the middle point and the low point have the same sign w.r.t. zero_tol,
                # the critical point is in the upper half
                p2_low = p2_middle
                x_low = x_mid
            else:
                p2_high = p2_middle
                x_high = x_mid
        critical_points.append((p2_low + p2_high) / 2)
    return critical_points, net


def get_wu_bounds(p1_range, kmax, rho):
    lower_bounds, upper_bounds = [], []
    lb_futures, ub_futures = [], []
    for p1 in p1_range:
        lb_func = lambda p2: find_lambda_star(lambda x, dummy1, dummy2: decay(x), interact, (p1, p2)) - kmax
        ub_func = lambda p2: find_lambda_star(lambda x, dummy1, dummy2: decay(x), interact, (p1, p2)) - rho
        t0 = time.time()
        lower_bounds.append(min(rootfinder(lb_func, [0, 10], do_initial_search=False, speak=True)))
        print(f"Time to find one bound: " + format_time_elapsed(t0))
        upper_bounds.append(min(rootfinder(ub_func, [0, 10])))
        print(f"Time to find two: " + format_time_elapsed(t0))
        print(f"lower bound: {lower_bounds} and upper bound: {upper_bounds}")
    return lower_bounds, upper_bounds


if __name__ == "__main__":
    t0 = time.time()
    p1s = np.linspace(1, 5, 1).tolist()
    print(p1s)
    critical_p2s, network = find_critical_points('gen_random', {'size': 5, 'p': 0.5}, p1s)
    t1 = time.time()
    print("DONE")
    lbs, ubs = get_wu_bounds(p1s, kmax(network.adj_matrix), rho(network.adj_matrix))
    print("DONE 2")
    plt.plot(p1s, critical_p2s)
    plt.fill_between(p1s, lbs, color='green', alpha=0.2)
    plt.fill_between(p1s, ubs, plt.gca().get_ylim(), color='red', alpha=0.2)
    plt.show()
    print("total " + format_time_elapsed(t0))
    print("to get lambda bounds " + format_time_elapsed(t1))
    get_cached_performance()






