"""
Utility functions for Langevin regression

Jared Callaham (2020)
"""

import numpy as np
from time import time
from scipy.optimize import minimize

# Return a single expression from a list of expressions and coefficients
#   Note this will give a SymPy expression and not a function
def sindy_model(Xi, expr_list):
    return sum([Xi[i]*expr_list[i] for i in range(len(expr_list))])


def ntrapz(I, dx):
    if isinstance(dx, int) or isinstance(dx, float) or len(dx)==1:
        return np.trapz(I, dx=dx, axis=0)
    else:
        return np.trapz( ntrapz(I, dx[1:]), dx=dx[0])
    

def kl_divergence(p_in, q_in, dx=1, tol=None):
    """
    Approximate Kullback-Leibler divergence for arbitrary dimensionality
    """
    if tol==None:
        tol = max( min(p_in.flatten()), min(q_in.flatten()))
    q = q_in.copy()
    p = p_in.copy()
    q[q<tol] = tol
    p[p<tol] = tol
    return ntrapz(p*np.log(p/q), dx)

def KM_avg(X, bins, stride, dt):
    
    Y = X[::stride] 
    tau = stride*dt
    dY = (Y[1:] - Y[:-1])/tau  # Step (like a finite-difference derivative estimate)
    dY2 = (Y[1:] - Y[:-1])**2/tau  # Conditional variance
    
    f_KM = np.zeros(len(bins)-1)
    a_KM = np.zeros(f_KM.shape)
    f_err = np.zeros(f_KM.shape)
    a_err = np.zeros(f_KM.shape)
    
    # At each histogram bin, find time series points where the state falls into this bin
    for i in range(len(bins)-1):
        mask = np.nonzero( (Y[:-1] > bins[i]) * (Y[:-1] < bins[i+1]) )[0]

        if len(mask) > 0:
            f_KM[i] = np.mean(dY[mask]) # Conditional average  ~ drift
            a_KM[i] = 0.5*np.mean(dY2[mask]) # Conditional variance  ~ diffusion

            # Estimate error by variance of samples in the bin
            a_KM[i] = 0.5*np.mean(dY2[mask]) # Conditional average
            a_err[i] = np.std(dY2[mask])/np.sqrt(len(mask))

        else:
            f_KM[i] = np.nan
            f_err[i] = np.nan
            a_KM[i] = np.nan
            a_err[i] = np.nan
            
    return f_KM, a_KM, f_err, a_err


# Return optimal coefficients for finite-time correctio
def AFP_opt(cost, params):
    ### RUN OPTIMIZATION PROBLEM
    start_time = time()
    Xi0 = params["Xi0"]

    is_complex = np.iscomplex(Xi0[0])
    
    if is_complex:
        Xi0 = np.concatenate((np.real(Xi0), np.imag(Xi0)))  # Split vector in two for complex
        opt_fun = lambda Xi: cost(Xi[:len(Xi)//2] + 1j*Xi[len(Xi)//2:], params)

    else:
        opt_fun = lambda Xi: cost(Xi, params)

    res = minimize(opt_fun, Xi0, method='nelder-mead',
              options={'disp': False, 'maxfev':int(1e4)})
    print('%%%% Optimization time: {0} seconds,   Cost: {1} %%%%'.format(time() - start_time, res.fun) )
    
    # Return coefficients and cost function
    if is_complex:
        # Return to complex number
        return res.x[:len(res.x)//2] + 1j*res.x[len(res.x)//2:], res.fun
    else:
        return res.x, res.fun


    
def SSR_loop(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)
    """
    
    # Lists of candidate expressions... coefficients are optimized
    f_expr, s_expr = params['f_expr'].copy(), params['s_expr'].copy()  
    lib_f, lib_s = params['lib_f'].copy(), params['lib_s'].copy()
    Xi0 = params['Xi0'].copy()
    
    m = len(f_expr) + len(s_expr)
    
    Xi = np.zeros((m, m-1), dtype=Xi0.dtype)  # Output results
    V = np.zeros((m-1))      # Cost at each step
    
    # Full regression problem as baseline
    Xi[:, 0], V[0] = opt_fun(params)
    
    # Start with all candidates
    active = np.array([i for i in range(m)])
    
    # Iterate and threshold
    for k in range(1, m-1):
        # Loop through remaining terms and find the one that increases the cost function the least
        min_idx = -1
        V[k] = 1e8
        for j in range(len(active)):
            tmp_active = active.copy()
            tmp_active = np.delete(tmp_active, j)  # Try deleting this term
            
            # Break off masks for drift/diffusion
            f_active = tmp_active[tmp_active < len(f_expr)]
            s_active = tmp_active[tmp_active >= len(f_expr)] - len(f_expr)
            print(f_active)
            print(s_active)
        
            print(f_expr[f_active], s_expr[s_active])
            params['f_expr'] = f_expr[f_active]
            params['s_expr'] = s_expr[s_active]
            params['lib_f'] = lib_f[:, f_active]
            params['lib_s'] = lib_s[:, s_active]
            params['Xi0'] = Xi0[tmp_active]
        
            # Ensure that there is at least one drift and diffusion term left
            if len(s_active) > 0 and len(f_active) > 0:
                tmp_Xi, tmp_V = opt_fun(params)

                # Keep minimum cost
                if tmp_V < V[k]:
                    # Ensure that there is at least one drift and diffusion term left
                    #if (IS_DRIFT and len(f_active)>1) or (not IS_DRIFT and len(a_active)>1):
                    min_idx = j
                    V[k] = tmp_V
                    min_Xi = tmp_Xi
            
        print("Cost: {0}".format(V[k]))
        # Delete least important term
        active = np.delete(active, min_idx)  # Remove inactive index
        Xi0[active] = min_Xi  # Re-initialize with best results from previous
        Xi[active, k] = min_Xi
        print(Xi[:, k])
        
    return Xi, V

def cost(Xi, params):
    """
    Least-squares cost function for optimization
    This version is only good in 1D, but could be extended pretty easily
    Xi - current coefficient estimates
    param - inputs to optimization problem: grid points, list of candidate expressions, regularizations
        W, f_KM, a_KM, x_pts, y_pts, x_msh, y_msh, f_expr, a_expr, l1_reg, l2_reg, kl_reg, p_hist, etc
    """
    
    # Unpack parameters
    W = params['W']  # Optimization weights
    
    # Kramers-Moyal coefficients
    f_KM, a_KM = params['f_KM'].flatten(), params['a_KM'].flatten()
    
    fp, afp = params['fp'], params['afp'] # Fokker-Planck solvers
    lib_f, lib_s = params['lib_f'], params['lib_s']
    N = params['N']
    
    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    f_vals = lib_f @ Xi[:lib_f.shape[1]]
    a_vals = 0.5*(lib_s @ Xi[lib_f.shape[1]:])**2
        
    # Solve AFP equation to find finite-time corrected drift/diffusion
    #    corresponding to the current parameters Xi
    afp.precompute_operator(np.reshape(f_vals, N), np.reshape(a_vals, N))
    f_tau, a_tau = afp.solve(params['tau'])
            
    # Histogram points without data have NaN values in K-M average - ignore these in the average
    mask = np.nonzero(np.isfinite(f_KM))[0]
    V = np.sum(W[0, mask]*abs(f_tau[mask] - f_KM[mask])**2) \
      + np.sum(W[1, mask]*abs(a_tau[mask] - a_KM[mask])**2)
    
    # Include PDF constraint via Kullbeck-Leibler divergence regularization
    if params['kl_reg'] > 0:
        p_hist = params['p_hist']  # Empirical PDF
        p_est = fp.solve(f_vals, a_vals)  # Solve Fokker-Planck equation for steady-state PDF
        kl = utils.kl_divergence(p_hist, p_est, dx=fp.dx, tol=1e-6)
        kl = max(0, kl)  # Numerical integration can occasionally produce small negative values
        V += params['kl_reg']*kl
    return V


# 1D Markov test
def markov_test(X, lag, N=32, L=2):
    # Lagged time series
    X1 = X[:-2*lag:lag]
    X2 = X[lag:-lag:lag]
    X3 = X[2*lag::lag]
    
    # Two-time joint pdfs
    bins = np.linspace(-L, L, N+1)
    dx = bins[1]-bins[0]
    p12, _, _ = np.histogram2d(X1, X2, bins=[bins, bins], density=True)
    p23, _, _ = np.histogram2d(X2, X3, bins=[bins, bins], density=True)
    p2, _ = np.histogram(X2, bins=bins, density=True)
    p2[p2<1e-4] = 1e-4
    
    # Conditional PDF (Markov assumption)
    pcond_23 = p23.copy()
    for j in range(pcond_23.shape[1]):
        pcond_23[:, j] = pcond_23[:, j]/p2
        
    # Three-time PDFs
    p123, _ = np.histogramdd(np.array([X1, X2, X3]).T, bins=np.array([bins, bins, bins]), density=True)
    p123_markov = np.einsum('ij,jk->ijk',p12, pcond_23)
    
    # Chi^2 value
    #return utils.ntrapz( (p123 - p123_markov)**2, [dx, dx, dx] )/(np.var(p123.flatten()) + np.var(p123_markov.flatten()))
    return kl_divergence(p123, p123_markov, dx=[dx, dx, dx], tol=1e-6)



### FAST AUTOCORRELATION FUNCTION
# From https://dfm.io/posts/autocorr/

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n
    
    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf