import numpy as np

def rmse(true, est):
    """Compute root mean square error across trajectories."""
    return np.sqrt(np.mean((np.array(true) - np.array(est))**2, axis=1))

def has_diverged(true_states, est, threshold=500):
    """Check divergence by comparing last position."""
    last_true = true_states[-1][:3]
    last_est = est[-1][:3]
    return np.linalg.norm(last_est - last_true) > threshold
