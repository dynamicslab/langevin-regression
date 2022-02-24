"""
Utility functions for Langevin regression on axisymmetric wake
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

def KM_weights(f_err, a_err):
    W = np.array((f_err.flatten(), a_err.flatten()))
    W[np.less(abs(W), 1e-12, where=np.isfinite(W))] = 1e6  # Set zero entries to large weights
    W[np.logical_not(np.isfinite(W))] = 1e6  # Set NaN entries to large numbers (small weights)
    W = 1/W  # Invert error for weights
    return W/np.nansum(W.flatten())  # Normalize for better numerics


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


def calc_cop(p):
    """
    # Compute center of pressure (adapted from George's MATLAB code)
    Reads in a snapshot p: (azimuthal) x (radial)

    % Rxi = 1/Sum[Pi*dAi] * Sum [Pi*xi*dAi]
    % Ryi = 1/Sum[Pi*dAi] * Sum [Pi*yi*dAi]

    % xi  = ri*cos(thetai)
    % yi  = ri*sin(thetai)
    % dAi = ri*dr*dtheta
    """
    # location of the 64 pressure taps
    D = 196.5
    rESP=np.linspace(11,88,8)/D            # non-dimensional with D
    thetaESP= np.array([270, 315, 0, 45, 90, 135, 180, 225]) #[degrees]

    p = p.T # nr x nth
    nr, nth = p.shape

    # Find ri , dAi
    dAi = np.zeros(nr)
    ri  = np.zeros(nr)
    dr = rESP[1]-rESP[0]
    dth = 2*np.pi/8
    for i in range(1, len(rESP)):
        ri[i] = 0.5*(rESP[i]+rESP[i-1])
        dAi[i] =   ri[i]*dr*dth

    # Construct P, xi,yi dAi
    Ptrapz = 0.5*(p[:,:-1]+p[:,1:] ) # 8 x 7
    dAi = dAi[1:]  # 1x7
    ri = ri[1:]    # 1x7

    xi= np.outer( np.cos(thetaESP*np.pi/180), ri)  # 8 x 7
    yi= np.outer( np.sin(thetaESP*np.pi/180), ri)    # 8 x 7

    dA = np.repeat(dAi[None, :], 8, axis=0)

    SumPA = sum(sum(Ptrapz*dA))

    Rxi = 1/SumPA*sum(sum(Ptrapz*xi*dA))
    Ryi = 1/SumPA*sum(sum(Ptrapz*yi*dA))

    return (Rxi, Ryi)


def load_new_expt(file):
    """Rearranges order of arrays to be as expected in python"""
    import scipy.io as sio
    data = sio.loadmat(file)
    P = np.concatenate(data['P'][0], axis=0).T
    P2 = P.copy()

    P2[24:32] = P[32:40]
    P2[32:40] = P[48:56]
    P2[40:48] = P[24:32]
    P2[48:56] = P[40:48]

    P2 = np.reshape(P2, [8, 8, -1], order='F')  # azimuthal x radius
    P2 = np.flip(P2, axis=0)
    return P2