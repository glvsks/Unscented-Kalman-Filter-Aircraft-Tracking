import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_states(time, true, ukf, ukf_sqrt, labels, titles):
    for i in range(7):
        plt.figure(figsize=(10, 5))
        plt.plot(time, true[:, i], 'k-', label='True')
        plt.plot(time, ukf[:, i], 'r--', label='UKF')
        plt.plot(time, ukf_sqrt[:, i], 'b-.', label='UKF-SQRT')
        plt.ylabel(labels[i]); plt.xlabel('Time [s]')
        plt.legend(); plt.grid(True)
        plt.title(f'{titles[i]} – State evolution')
        plt.tight_layout()
        plt.show()

def plot_errors(time, true, ukf, ukf_sqrt, ukf_P, ukf_sqrt_P, labels, titles):
    for i in range(7):
        plt.figure(figsize=(10, 5))
        err_ukf = true - ukf
        err_ukf_sqrt = true - ukf_sqrt
        plt.plot(time, err_ukf[:, i], 'r-', label='UKF error')
        plt.fill_between(time, -3*np.sqrt(ukf_P[:, i, i]), 3*np.sqrt(ukf_P[:, i, i]),
                         color='red', alpha=0.2, label='±3σ UKF')
        plt.plot(time, err_ukf_sqrt[:, i], 'b-', label='UKF-SQRT error')
        plt.fill_between(time, -3*np.sqrt(ukf_sqrt_P[:, i, i]), 3*np.sqrt(ukf_sqrt_P[:, i, i]),
                         color='blue', alpha=0.2, label='±3σ UKF-SQRT')
        plt.ylabel(labels[i]); plt.xlabel('Time [s]')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True); plt.title(f'{titles[i]} – Estimation error and ±3σ')
        plt.tight_layout(); plt.show()

def plot_std(time, true_std, ukf_std, ukf_sqrt_std, labels, titles):
    for i in range(7):
        plt.figure(figsize=(10, 5))
        plt.plot(time, true_std[:, i], 'k-', label='True')
        plt.plot(time, ukf_std[:, i], 'r-', label='UKF')
        plt.plot(time, ukf_sqrt_std[:, i], 'b-', label='UKF-SQRT')
        plt.ylabel(labels[i]); plt.xlabel('Time [s]')
        plt.legend(); plt.grid(True)
        plt.title(f'{titles[i]} – Sample standard deviation')
        plt.tight_layout(); plt.show()

def plot_trajectories(true, ukf, ukf_sqrt):
    # 2D
    plt.figure(figsize=(8, 8))
    plt.plot(true[:, 0], true[:, 2], 'k-', label='True')
    plt.plot(ukf[:, 0], ukf[:, 2], 'r--', label='UKF')
    plt.plot(ukf_sqrt[:, 0], ukf_sqrt[:, 2], 'b-.', label='UKF-SQRT')
    plt.xlabel('ε [m]'); plt.ylabel('η [m]')
    plt.legend(); plt.grid(True)
    plt.title('Aircraft trajectory in ε–η plane')
    plt.axis('equal'); plt.tight_layout()
    plt.savefig('reports/figures/trajectory_xy.png', dpi=200)
    plt.show()

    # 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(true[:, 0], true[:, 2], true[:, 4], 'k-', label='True')
    ax.plot(ukf[:, 0], ukf[:, 2], ukf[:, 4], 'r--', label='UKF')
    ax.plot(ukf_sqrt[:, 0], ukf_sqrt[:, 2], ukf_sqrt[:, 4], 'b-.', label='UKF-SQRT')
    ax.set_xlabel('X (ε) [m]'); ax.set_ylabel('Y (η) [m]'); ax.set_zlabel('Z (ζ) [m]')
    ax.set_title('3D Aircraft trajectory'); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig('reports/figures/trajectory_3d.png', dpi=200)
    plt.show()
