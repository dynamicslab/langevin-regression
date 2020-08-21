"""
Utility functions for stochastic modeling of the double-well potential

Jared Callaham (2020)
"""

import numpy as np
import sympy
from scipy.optimize import curve_fit

import utils
import fpsolve

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import julia
jl = julia.Julia()
jl.include("../sim_doublewell.jl")

def switched_states(X, thresh=1):
    # Decide whether the bistable states are "up" or "down" based on threshold
    
    N = len(X)
    state = np.zeros((N))
    
    # Step forward until X is either positive or negative (if simulation is initialized at 0)
    idx = 0
    while abs(X[idx]) < thresh:
        idx += 1
    
    # Consider everything up to this point to be in the first well
    cur_state = np.sign(X[idx])
    state[:idx] = cur_state
    
    while idx<N:
        
        # Switch states after crossing threshold
        if -cur_state*X[idx] > thresh:
            cur_state = -cur_state
        
        state[idx] = cur_state
        idx += 1
        
    return state


def dwell_times(states, dt=1):
    # Given a vector of states (i.e. from switched_states() ), return dwell time in each state
    N = len(states)
    switch_times = []  # List of dwell times
    idx = 0
    last_switch = idx
    cur_state = states[idx]
    
    while idx < N:
        if states[idx] != cur_state:
            switch_times.append( dt*(idx-last_switch) )
            last_switch = idx
            cur_state = states[idx]
        
        idx += 1
        
    return switch_times

def dwell_stats(X, thresh, dt):
    
    state = switched_states(X, thresh=thresh)  # Categorize into "up" or "down"
    switch_times = dwell_times(state, dt=dt)   # Compute list of dwell times in each metastable state

    if len(switch_times) > 0:
        return np.mean(switch_times), np.std(switch_times)/np.sqrt(len(switch_times))
    else:
        return np.nan, np.nan


def fit_pdf(X, edges, p_hist, dt, p0=None):
    # Mean-square displacement
    fit_start, fit_stop = 0.1, 3
    n_lags = int(fit_stop/dt)
    tau = dt*np.arange(1, n_lags)

    # Lagged mean square displacement
    msd = np.zeros((len(tau)))
    for i in range(1, n_lags):
        msd[i-1] = np.mean( (X[i:]-X[:-i])**2)
    
    # Linear fit for radial displacement
    to_fit = np.nonzero( (tau > fit_start) * (tau < fit_stop) )[0]
    p_rad = np.polyfit(tau[to_fit], msd[to_fit], deg=1)
    a_pdf = 0.5*p_rad[0]
    
    # Fit PDF
    p_model = lambda x, C, a, b:  C*np.exp(a*x**2 + b*x**4)
    centers = 0.5*(edges[1:]+edges[:-1])
    if p0 is not None:
        popt, pcov = curve_fit(p_model, centers, p_hist, p0=p0)
    else:
        popt, pcov = curve_fit(p_model, centers, p_hist)

    # Separate parameters in model
    sigma_pdf = np.sqrt(2*a_pdf)
    lamb_pdf = popt[1]*sigma_pdf**2
    mu_pdf = popt[2]*2*sigma_pdf**2
    
    return lamb_pdf, mu_pdf, sigma_pdf


def langevin_regression(X, edges, p_hist, dt, stride=200, kl_reg=0):
    """
    Wrapper for full Langevin regression so we can loop over it to explore variations with distance from bifurcation
    """
    centers = 0.5*(edges[1:]+edges[:-1])
    N = len(centers)
    
    # Kramers-Moyal average
    tau = stride*dt
    f_KM, a_KM, f_err, a_err = KM_avg(X, bins, stride=stride, dt=dt)

    # Initialize libraries
    x = sympy.symbols('x')

    f_expr = np.array([x**i for i in [1, 3]])  # Polynomial library for drift
    s_expr = np.array([x**i for i in [0]])  # Polynomial library for diffusion

    lib_f = np.zeros([len(f_expr), N])
    for k in range(len(f_expr)):
        lamb_expr = sympy.lambdify(x, f_expr[k])
        lib_f[k] = lamb_expr(centers)

    lib_s = np.zeros([len(s_expr), N])
    for k in range(len(s_expr)):
        lamb_expr = sympy.lambdify(x, s_expr[k])
        lib_s[k] = lamb_expr(centers)

    # Initialize Xi with plain least-squares (just helpf the optimization a bit)
    Xi0 = np.zeros((len(f_expr) + len(s_expr)))
    mask = np.nonzero(np.isfinite(f_KM))[0]
    Xi0[:len(f_expr)] = np.linalg.lstsq( lib_f[:, mask].T, f_KM[mask], rcond=None)[0]
    Xi0[len(f_expr):] = np.linalg.lstsq( lib_s[:,mask].T, np.sqrt(2*a_KM[mask]), rcond=None)[0]

    # Parameter dictionary for optimization
    W = np.array((f_err.flatten(), a_err.flatten()))
    W[np.less(abs(W), 1e-12, where=np.isfinite(W))] = 1e6  # Set zero entries to large weights
    W[np.logical_not(np.isfinite(W))] = 1e6                 # Set NaN entries to large numbers (small weights)
    W = 1/W  # Invert error for weights
    W = W/np.nansum(W.flatten())

    # Adjoint solver
    afp = fpsolve.AdjFP(centers)

    # Forward solver
    fp = fpsolve.SteadyFP(N, centers[1]-centers[0])

    params = {"W": W, "f_KM": f_KM, "a_KM": a_KM, "Xi0": Xi0,
              "f_expr": f_expr, "s_expr": s_expr,
              "lib_f": lib_f.T, "lib_s": lib_s.T, "N": N,
              "kl_reg": kl_reg,
              "fp": fp, "afp": afp, "p_hist": p_hist, "tau": tau,
              "radial": False}

    # Tune KL regularization automatically
    Xi, _ = utils.AFP_opt(utils.cost, params)
    return Xi

def model_eval(eps, sigma, N, kl_reg):
    """
    Construct and evaluate all models of the double-well
    1. Analytic normal form model
    2. PDF fitting without Kramers-Moyal average
    3. Full Langevin regression
    """
    ### Generate data
    x_eq = np.sqrt(eps)  # Equilibrium value
    
    edges = np.linspace(-2*x_eq, 2*x_eq, N+1)
    centers = 0.5*(edges[:-1]+edges[1:])
    dx = centers[1]-centers[0]

    dt = 1e-2
    tmax = int(1e5)
    t, X = jl.run_sim(eps, sigma, dt, tmax)
    X, V = X[0, :], X[1, :]
    
    # PDF of states
    p_hist = np.histogram(X, edges, density=True)[0]
    
    # Dwell-time slope
    b, b_err = dwell_stats(X, x_eq, dt)
    print("\tData: ", b, b_err)
    
    ### 1. Normal form
    lamb1 = -1 + np.sqrt(1 + eps)
    lamb2 = -1 - np.sqrt(1 + eps)
    h = -lamb1/lamb2
    mu = -(1+h)**2*lamb1/eps

    _, phi1 = jl.run_nf(lamb1, mu, sigma/(2*np.sqrt(1+eps)), dt, tmax)
    X_nf = (1+h)*phi1[0, :]
    
    # Statistics
    p_nf = np.histogram(X_nf, edges, density=True)[0]
    b_nf, b_nf_err = dwell_stats(X_nf, x_eq, dt)
    print("\tNormal form: ", b_nf, b_nf_err)
    
    ### 2. PDF fit
    Xi = fit_pdf(X, edges, p_hist, dt, p0=[1, lamb1/sigma**2, mu/sigma**2])
    #print(Xi)
    
    # Monte Carlo evaluation
    _, X_pdf = jl.run_nf(Xi[0], Xi[1], Xi[2], dt, tmax)
    X_pdf = X_pdf[0, :]
    
    # Statistics
    p_pdf = np.histogram(X_pdf, edges, density=True)[0]
    b_pdf, b_pdf_err = dwell_stats(X_pdf, x_eq, dt)
    print("\tPDF fit: ", b_pdf, b_pdf_err)
    
    ### 3. Langevin regression
    Xi = langevin_regression(X, edges, p_hist, dt, stride=200, kl_reg=kl_reg)
    #print(Xi)
    
    # Monte Carlo evaluation
    _, X_lr = jl.run_nf(Xi[0], Xi[1], Xi[2], dt, tmax)
    X_lr = X_lr[0, :]
    
    # Statistics
    p_lr = np.histogram(X_lr, edges, density=True)[0]
    b_lr, b_lr_err = dwell_stats(X_lr, x_eq, dt)
    print("\tLangevin regression: ", b_lr, b_lr_err)
    
    ### KL-divergence of all models against true data
    KL_nf = utils.kl_divergence(p_hist, p_nf, dx=dx, tol=1e-6)
    KL_pdf = utils.kl_divergence(p_hist, p_pdf, dx=dx, tol=1e-6)
    KL_lr = utils.kl_divergence(p_hist, p_lr, dx=dx, tol=1e-6)
    print("\tKL div: ", KL_nf, KL_pdf, KL_lr)
    
    return [b, b_nf, b_pdf, b_lr], [KL_nf, KL_pdf, KL_lr]