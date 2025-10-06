import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from .models import fx, hx, x0, sigma1, sigma2, sigma_r, sigma_theta, sigma_phi

def init_filters(dt):
    """Initialize UKF and UKF-SQRT with identical settings."""
    points = MerweScaledSigmaPoints(n=7, alpha=0.1, beta=2.0, kappa=0.0)
    def create_filter():
        f = UKF(dim_x=7, dim_z=3, dt=dt, fx=fx, hx=hx, points=points)
        f.x = x0.copy()
        f.P = np.diag([10, 1, 10, 1, 10, 1, 0.1])
        f.Q = np.diag([0, sigma1**2, 0, sigma1**2, 0, sigma1**2, sigma2**2])
        f.R = np.diag([sigma_r**2, sigma_theta**2, sigma_phi**2])
        return f
    return create_filter(), create_filter()
