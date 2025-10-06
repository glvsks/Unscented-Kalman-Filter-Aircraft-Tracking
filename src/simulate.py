import numpy as np
from numpy.random import multivariate_normal
from .models import fx, hx, x0, sigma1, sigma2, sigma_r, sigma_theta, sigma_phi

def generate_trajectory(steps, dt, seed=42):
    """Simulate true trajectory and noisy measurements."""
    np.random.seed(seed)
    true_states, measurements = [], []
    x = x0.copy()

    for _ in range(steps):
        process_noise = multivariate_normal(np.zeros(7),
                                            np.diag([0, sigma1**2, 0, sigma1**2, 0, sigma1**2, sigma2**2]))
        meas_noise = multivariate_normal(np.zeros(3),
                                         np.diag([sigma_r**2, sigma_theta**2, sigma_phi**2]))
        x = fx(x, dt) + process_noise
        z = hx(x) + meas_noise
        true_states.append(x)
        measurements.append(z)
    return np.array(true_states), np.array(measurements)
