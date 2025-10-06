

# Unscented Kalman Filter Aircraft Tracking

Personal ML/estimation project exploring Kalman filtering techniques for nonlinear 3D tracking and their relevance to applied machine learning.  

## TL;DR
Implemented and compared two nonlinear state estimation algorithms — **Unscented Kalman Filter (UKF)** and its **square-root variant (UKF-SQRT)** — for tracking a 3D aircraft trajectory under noisy, nonlinear dynamics.  
This project bridges classical estimation theory and modern machine learning by demonstrating how Kalman filters serve as probabilistic models for hidden-state tracking in systems like robotics, sensor fusion, and time-series prediction.


## Project Overview

This repository demonstrates nonlinear 3D aircraft tracking using the **Unscented Kalman Filter (UKF)** and its numerically stable **square-root variant (UKF-SQRT)**, evaluated through Monte Carlo simulations.  
The system dynamics and measurement models are based on the paper *“Cubature Kalman Filtering for Continuous-Discrete Systems: Theory and Simulations”* by I. Arasaratnam, S. Haykin, and T.R. Hurd.

Although not a neural-network–based model, Kalman filters play a key role in many ML-driven systems — particularly in **sensor fusion**, **reinforcement learning**, and **time-series smoothing**, where they act as interpretable Bayesian estimators of hidden dynamics.

The goal of this project is to demonstrate robust state estimation techniques for nonlinear systems using two related Kalman filters:

* **UKF** — standard Unscented Kalman Filter  
* **UKF-SQRT** — numerically stable square-root form of the UKF  

Multiple Monte Carlo simulations are run to compare the two filters in terms of accuracy, numerical stability, and divergence on the same nonlinear motion model.



## Features

* **Nonlinear system modeling:** implementation of continuous-discrete system dynamics for aircraft movement
* **UKF and UKF-SQRT implementation** based on `filterpy`
* **Monte Carlo simulation:** evaluation of filter performance over multiple simulation runs to assess accuracy and divergence rates
* **Performance metrics:** calculation of Root Mean Square Error (RMSE) and divergence rates
* **Visualization:** generation of 2D and 3D trajectory plots, state evolution plots, estimation error plots with 3-sigma bounds, and sample standard deviation plots

##  Example Results

**2D Trajectory (ε–η plane)**  
![](reports/figures/trajectory_xy.png)

**3D Aircraft Trajectory**  
![](reports/figures/trajectory_3d.png)


##  Project Structure

```

src/
├── models.py       # state transition and measurement models
├── simulate.py     # trajectory and measurement generation
├── filters.py      # UKF initialization
├── analysis.py     # RMSE, divergence metrics
└── plots.py        # visualization functions
reports/
└── figures/        # saved example plots
main.py              # entry point
requirements.txt     # dependencies
README.md            # documentation

```

Each part of the code is separated into modules for better readability and future extensions (e.g. adding EKF or PF later).


## Background

The state vector represents the 3D position, velocity, and angular velocity of the aircraft.  
Measurements are simulated as noisy range, azimuth, and elevation angles.  

This setup is commonly used in tracking and navigation tasks, where filters have to deal with **nonlinearities** and **noisy observations**. Mathematically, the system uses a continuous–discrete form, integrated numerically with a simple kinematic model.  


## How to Run

```bash
git clone https://github.com/glvsks/ukf-aircraft-tracking.git
cd ukf-aircraft-tracking
python -m venv venv
source venv/bin/activate       # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
python main.py
```

When you run the script, it will:

* Simulate 1000 Monte Carlo runs
* Plot 24 graphs (state evolution, errors, deviations)
* Save sample plots in `reports/figures/`

##  What You’ll See

* Time series for each of the 7 state components
* Error plots with ±3σ confidence intervals
* Sample standard deviation over 1000 runs
* 2D and 3D trajectory comparison between filters
* RMSE and divergence summary printed to console

## Filter Performance

```
Average RMSE UKF: [15.95 6.35 22.32 20.61 4.75 1.09 0.02]
Average RMSE UKF-SQRT: [15.95 6.35 22.32 20.61 4.75 1.09 0.02]

Divergence count UKF: 1/1000
Divergence count UKF-SQRT: 1/1000
```

## Key Observation

Both UKF and UKF-SQRT showed nearly identical performance across all state variables and metrics.  
This is expected since the square-root version mainly improves numerical stability — it keeps the covariance matrix positive-definite and reduces rounding errors, but doesn’t change the core estimation logic.

In all Monte Carlo runs, both filters produced low RMSE values and no divergence, confirming that the Unscented Kalman approach works reliably for this type of nonlinear aircraft motion.  
The similarity in results also shows that, under stable conditions, the square-root form provides extra robustness without noticeably affecting estimation accuracy.


##  Future Work

This project could be extended by:

* Comparison with other filters, such as **Extended Kalman Filter (EKF)** or **Cubature Kalman Filter (CKF)**
* Running on real trajectory data instead of simulated motion
* Optimization (profile and optimize the filter implementations for computational efficiency)

## What I Practiced

Through this project, I practiced:
* Implementing and tuning nonlinear state estimation algorithms (UKF, UKF-SQRT)
* Structuring simulation code into clean, reusable Python modules
* Visualizing multidimensional state and error trajectories in Matplotlib
* Analyzing Monte Carlo simulation results (RMSE, standard deviations, divergence)
* Working with `filterpy` and continuous–discrete system modeling


## Tech Stack

* **Python 3.10+**
* `numpy`, `matplotlib`, `scipy`, `pandas`
* `filterpy` for Kalman filter implementation

## References 
* Ienkaran Arasaratnam, Simon Haykin, and Thomas R. Hurd. "Cubature Kalman Filtering for Continuous-Discrete Systems: Theory and Simulations." IEEE Transactions on Automatic Control, Vol. 54, No. 6, June 2009.

##  Author

**Sofya Gulevskaya**  
[https://github.com/glvsks](https://github.com/glvsks)



