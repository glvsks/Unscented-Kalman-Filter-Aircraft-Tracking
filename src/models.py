import numpy as np

# Initial parameters
omega = np.deg2rad(3)

x0 = np.array([1000, 0, 2650, 150, 200, 0, omega])

# Noise parameters
sigma1 = np.sqrt(0.2)
sigma2 = 7e-3
sigma_r = 50
sigma_theta = np.deg2rad(0.1)
sigma_phi = np.deg2rad(0.1)

# Models
def fx(x, dt):
    """Nonlinear state transition model."""
    eps, eps_dot, eta, eta_dot, zeta, zeta_dot, omega = x
    new_eps = eps + eps_dot * dt - 0.5 * omega * eta_dot * dt**2
    new_eps_dot = eps_dot - omega * eta_dot * dt - 0.5 * omega**2 * eps_dot * dt**2
    new_eta = eta + eta_dot * dt + 0.5 * omega * eps_dot * dt**2
    new_eta_dot = eta_dot + omega * eps_dot * dt - 0.5 * omega**2 * eta_dot * dt**2
    new_zeta = zeta + zeta_dot * dt
    new_zeta_dot = zeta_dot
    return np.array([new_eps, new_eps_dot, new_eta, new_eta_dot, new_zeta, new_zeta_dot, omega])

def hx(x):
    """Measurement model: converts Cartesian state to polar coordinates."""
    eps, _, eta, _, zeta, _, _ = x
    r = np.sqrt(eps**2 + eta**2 + zeta**2)
    theta = np.arctan2(eta, eps)
    phi = np.arctan2(zeta, np.sqrt(eps**2 + eta**2))
    return np.array([r, theta, phi])
