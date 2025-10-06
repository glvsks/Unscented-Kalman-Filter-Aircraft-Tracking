import numpy as np
import pandas as pd
from src.simulate import generate_trajectory
from src.filters import init_filters
from src.analysis import rmse, has_diverged
from src.plots import plot_states, plot_errors, plot_std, plot_trajectories

# Simulation configuration
steps = 64
dt = 1.0
n_monte_carlo = 1000
div_threshold = 500

all_true, all_ukf, all_ukf_sqrt, all_ukf_P, all_ukf_sqrt_P = [], [], [], [], []
div_ukf, div_ukf_sqrt = [], []

for mc_run in range(n_monte_carlo):
    true_states, measurements = generate_trajectory(steps, dt, seed=42 + mc_run)
    ukf, ukf_sqrt = init_filters(dt)

    ukf_est, ukf_sqrt_est, ukf_P, ukf_sqrt_P = [], [], [], []

    for z in measurements:
        ukf.predict(); ukf.update(z)
        ukf_sqrt.predict(); ukf_sqrt.update(z)
        ukf_est.append(ukf.x.copy())
        ukf_P.append(ukf.P.copy())
        ukf_sqrt_est.append(ukf_sqrt.x.copy())
        ukf_sqrt_P.append(ukf_sqrt.P.copy())

    if has_diverged(true_states, ukf_est, div_threshold):
        div_ukf.append(np.array(true_states))
    else:
        div_ukf.append(None)
    if has_diverged(true_states, ukf_sqrt_est, div_threshold):
        div_ukf_sqrt.append(np.array(true_states))
    else:
        div_ukf_sqrt.append(None)


    all_true.append(np.array(true_states))
    all_ukf.append(np.array(ukf_est))
    all_ukf_sqrt.append(np.array(ukf_sqrt_est))
    all_ukf_P.append(np.array(ukf_P))
    all_ukf_sqrt_P.append(np.array(ukf_sqrt_P))

# Visualization
time = np.arange(steps) * dt
labels = ['ε [m]', 'ε̇ [m/s]', 'η [m]', 'η̇ [m/s]', 'ζ [m]', 'ζ̇ [m/s]', 'ω [rad/s]']
titles = ['ε', 'ε̇', 'η', 'η̇', 'ζ', 'ζ̇', 'ω']

plot_states(time, all_true[0], all_ukf[0], all_ukf_sqrt[0], labels, titles)
plot_errors(time, all_true[0], all_ukf[0], all_ukf_sqrt[0], all_ukf_P[0], all_ukf_sqrt_P[0], labels, titles)

sample_std_true = np.std(np.array(all_true), axis=0)
sample_std_ukf = np.std(np.array(all_true) - np.array(all_ukf), axis=0)
sample_std_ukf_sqrt = np.std(np.array(all_true) - np.array(all_ukf_sqrt), axis=0)
plot_std(time, sample_std_true, sample_std_ukf, sample_std_ukf_sqrt, labels, titles)

plot_trajectories(all_true[0], all_ukf[0], all_ukf_sqrt[0])

# RMSE and summary
rmse_ukf = rmse(all_true, all_ukf)
rmse_ukf_sqrt = rmse(all_true, all_ukf_sqrt)
print("\nAverage RMSE UKF:", np.mean(rmse_ukf, axis=0))
print("Average RMSE UKF-SQRT:", np.mean(rmse_ukf_sqrt, axis=0))

count_ukf = sum(1 for x in div_ukf if x is not None)
count_ukf_sqrt = sum(1 for x in div_ukf_sqrt if x is not None)
print(f"\nDivergence count UKF: {count_ukf}/{n_monte_carlo}")
print(f"Divergence count UKF-SQRT: {count_ukf_sqrt}/{n_monte_carlo}")

