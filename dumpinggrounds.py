from concurrent.futures import ThreadPoolExecutor

import numpy as np

from auxiliary_functions import dx_dt, find_lambda_star, rootfinder
from model import stability_run
from network_class import Network


def find_critical_points(network_generator: str, network_params: dict, p1_range, p2_error_tolerance=1e-1, zero_tol=1e-1):
    # Initialize the network
    net = Network()
    generator = getattr(net, network_generator)
    generator(**network_params)
    net.randomize_weights(lambda x: x+1)
    metric = np.linalg.norm

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
        if ((metric(x_low) > zero_tol and metric(x_high) > zero_tol) or
                (metric(x_low) < zero_tol and metric(x_high) < zero_tol)):
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

            if (metric(x_mid) - zero_tol) * (metric(x_low) - zero_tol) > 0:
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
    """Doesn't work yet entirely"""
    lower_bounds, upper_bounds = [], []
    lb_futures, ub_futures = [], []
    for p1 in p1_range:
        lb_func = lambda p2: find_lambda_star(lambda x, dummy1, dummy2: decay(x), interact, (p1, p2)) - kmax
        ub_func = lambda p2: find_lambda_star(lambda x, dummy1, dummy2: decay(x), interact, (p1, p2)) - rho
        with ThreadPoolExecutor() as exc:
            lb_futures.append(exc.submit(lambda: rootfinder(lb_func, [0, 10], N=10)))
            ub_futures.append(exc.submit(lambda: rootfinder(ub_func, [0, 10], N=10)))
    for lb_fut, ub_fut in zip(lb_futures, ub_futures):
        lb_roots, ub_roots = lb_fut.result(), ub_fut.result()
        if len(lb_roots) > 0:
            lower_bounds.append(lb_roots[0])
        else:
            lower_bounds.append(0)
        if len(ub_roots) > 0:
            upper_bounds.append(ub_roots[0])
        else:
            # TODO: This doesn't work
            print("We've got a problem here")
            upper_bounds.append(0)
    return lower_bounds, upper_bounds
