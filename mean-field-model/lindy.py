import numpy as np
import sympy
from time import time

from scipy import sparse, linalg
from scipy.optimize import minimize
from numpy.linalg import lstsq

import utils
import fpsolve

# freestream parameters for non-dimensionalization
Uinf = 15                # freestream velocity [m/sec]
Pinf = 0.5*1.2041*15**2  # rho_air = 1.2041 (20C) -> Pinf = 135.4612
D = 196.5                # Diameter of the body [mm]

# Sampling rate
fs = 225  # Hz   
dt = 1/fs

class LRModel():
    def __init__(self, data, dim=1, dt=1/225):
        self.data = data
        self.dim = dim
        self.dt = dt
        
    def compute_KM(self, lag):
        self.f_KM = None
        self.a_KM = None
        self.W = None
        self.tau = lag*self.dt
        raise NotImplementedError
        
    def check_KM_initialized(self):
        return ( hasattr(self, 'f_KM') and hasattr(self, 'a_KM') and
                 hasattr(self, 'W') and hasattr(self, 'tau') )
    
    def compute_PDF(self):
        self.p_hist = None
        raise NotImplementedError
        
    def check_PDF_initialized(self):
        return hasattr(self, 'p_hist')
    
    def precompute_libraries(self, x_sym, f_expr, s_expr, x_eval):
        self.f_expr = f_expr
        self.s_expr = s_expr
        N = len(x_eval)
        self.lib_f = np.zeros([N, len(f_expr)], dtype=x_eval.dtype)
        for k in range(len(f_expr)):
            lamb_expr = sympy.lambdify(x_sym, f_expr[k])
            self.lib_f[:, k] = lamb_expr(x_eval)

        self.lib_s = np.zeros([N, len(s_expr)])
        for k in range(len(s_expr)):
            lamb_expr = sympy.lambdify(x_sym, s_expr[k])
            self.lib_s[:, k] = lamb_expr(x_eval)
            
    def check_libs_initialized(self):
        return hasattr(self, 'lib_f') and hasattr(self, 'lib_s')
            
    def initial_coeffs(self):
        self.Xi0 = None
        raise NotImplementedError
        
    def check_coeffs_initialized(self):
        return hasattr(self, 'Xi0')
    
    def initialize_fpsolve(self, centers, dim=1):
        self.afp = fpsolve.AdjFP(centers, ndim=dim, method="exp") # Adjoint solver
        self.fp = fpsolve.SteadyFP(centers, ndim=dim, method="galerkin") # Forward solver
        
    def check_fpsolve_initialized(self):
        return hasattr(self, 'afp') and hasattr(self, 'fp')
    
    def initialized(self):
        return (self.check_KM_initialized() and 
                self.check_PDF_initialized() and
                self.check_libs_initialized() and
                self.check_coeffs_initialized() and
                self.check_fpsolve_initialized() )
    
    def cost(self, Xi, kl_reg):
        return NotImplementedError
    
    def fit(self, kl_reg=1e-3):
        assert self.initialized()
        print(f"KL Regularization: {kl_reg}")
        
        start_time = time()
        is_complex = np.iscomplex(self.Xi0[0])

        if is_complex:
            Xi0 = np.concatenate((np.real(self.Xi0), np.imag(self.Xi0)))  # Split vector in two for complex
            opt_fun = lambda Xi: self.cost(Xi[:len(Xi)//2] + 1j*Xi[len(Xi)//2:], kl_reg)

        else:
            Xi0 = self.Xi0
            opt_fun = lambda Xi: self.cost(Xi, kl_reg)

        res = minimize(opt_fun, Xi0, method='nelder-mead',
                  options={'disp': False, 'maxfev':int(1e4)})
        print('%%%% Optimization time: {0} seconds,   Cost: {1} %%%%'.format(time() - start_time, res.fun) )

        # Return coefficients and cost function
        if is_complex:
            return res.x[:len(res.x)//2] + 1j*res.x[len(res.x)//2:], res.fun
        else:
            return res.x, res.fun
        
        
class A_model(LRModel):
    def __init__(self, data, dt=1/225, nbins=24, domain_width=2, lag=200):
        super().__init__(data, dim=2, dt=dt)
        self.data = data
        self.nbins = nbins
        self.domain_width = domain_width
        
        self.compute_PDF()
        self.compute_KM(lag)
        self.precompute_libraries()
        self.initial_coeffs()
        self.initialize_fpsolve()
        
    def compute_PDF(self):
        #########################################################
        ### PROBABILITY DENSITIES
        #########################################################
        A = self.data
        self.bins = np.linspace(-self.domain_width, self.domain_width, self.nbins+1)
        dx = self.bins[1]-self.bins[0]
        self.centers1d = (self.bins[:-1]+self.bins[1:])/2

        YY, XX = np.meshgrid(self.centers1d, self.centers1d)
        self.centers2d = [XX, YY]

        # Histogram: [real, imag]
        self.p_hist, _, _ = np.histogram2d(np.real(A), np.imag(A), bins=[self.bins, self.bins], density=True)

        
    def compute_KM(self, lag):
        self.tau = lag*self.dt
        A = self.data

        dA = (A[lag:] - A[:-lag])/self.tau  # Cartesian step (finite-difference derivative estimate)
        dA2 = self.tau*(np.real(dA)**2 + 1j*np.imag(dA)**2) # Multivariate variance (assuming diagonal)

        N = len(self.bins)-1
        f_KM = np.zeros((N, N), dtype=np.complex64)
        f_err = np.zeros(f_KM.shape)
        a_KM = np.zeros((f_KM.shape), dtype=np.complex64)
        a_err = np.zeros((f_KM.shape))

        for i in range(N):
            for j in range(N):
                    # Find where signal falls into this bin
                    mask = (np.real(A[:-lag]) > self.bins[i]) * (np.real(A[:-lag]) < self.bins[i+1]) * \
                            (np.imag(A[:-lag]) > self.bins[j]) * (np.imag(A[:-lag]) < self.bins[j+1])

                    mask_idx = np.nonzero(mask)[0]
                    nmask = len(mask_idx)

                    if nmask > 0:
                        # Conditional mean
                        f_KM[i, j] = np.mean(dA[mask_idx]) # Conditional average
                        f_err[i, j] = np.std(abs(dA[mask_idx]))/np.sqrt(nmask)

                        # Conditional variance  (assumes diagonal forcing)
                        a_KM[i, j] = 0.5*np.mean(dA2[mask_idx]) # Conditional average
                        a_err[i, j] = np.std(dA2[mask_idx])/np.sqrt(nmask)

                    else:
                        f_KM[i, j] = np.nan
                        f_err[i, j] = np.nan
                        a_KM[i, j] = np.nan
                        a_err[i, j] = np.nan

        self.f_KM = np.array([np.real(f_KM), np.imag(f_KM)])
        self.a_KM = np.array([np.real(a_KM), np.imag(a_KM)])
        f_err = np.array([np.real(f_err), np.imag(f_err)])
        a_err = np.array([np.real(a_err), np.imag(a_err)])
        
        self.W = utils.KM_weights(f_err, a_err)
        
    def precompute_libraries(self):
        # Initialize sympy expression
        z = sympy.symbols('z') # real, imag
        
        # Lists of candidate functions
        self.f_expr = np.array([z, z*abs(z)**2])
        self.s_expr = np.array([z**0, abs(z)**2])
        
        ZZ = self.centers2d[0] + 1j*self.centers2d[1]
        LRModel.precompute_libraries(self, z, self.f_expr, self.s_expr, ZZ.flatten())
        
        
    def initial_coeffs(self):
        self.Xi0 = np.zeros((len(self.f_expr) + len(self.s_expr)), dtype=np.complex64)
        lhs = (self.f_KM[0, :, :] + 1j*self.f_KM[1, :, :]).flatten()
        mask = np.nonzero(np.isfinite(lhs))[0]
        self.Xi0[:len(self.f_expr)] = lstsq( self.lib_f[mask, :], lhs[mask], rcond=None)[0]

        lhs = np.sqrt(2*self.a_KM[0, :, :].flatten()) + 1j*np.sqrt(2*self.a_KM[1, :, :].flatten())
        self.Xi0[len(self.f_expr):] = lstsq( self.lib_s[mask, :], lhs[mask], rcond=None)[0]
        
    def initialize_fpsolve(self):
        LRModel.initialize_fpsolve(self, centers=[self.centers1d, self.centers1d], dim=2)
        
    def cost(self, Xi, kl_reg):
        r"""Least-squares cost function for optimization"""
        f_KM, a_KM = self.f_KM[0].flatten(), self.a_KM[0].flatten()

        Xi_f = Xi[:self.lib_f.shape[1]]
        Xi_s = Xi[self.lib_f.shape[1]:]

        f_vals = self.lib_f @ Xi_f
        s_vals = self.lib_s @ Xi_s
        a_vals = 0.5*( np.real(s_vals)**2 + 1j*(np.imag(s_vals))**2 )

        # Solve adjoint Fokker-Planck equation
        self.afp.precompute_operator([np.real(f_vals), np.imag(f_vals)],
                                     [np.real(a_vals), np.imag(a_vals)])
        f_tau, a_tau = self.afp.solve(self.tau, d=0)  # Assumes real/imag symmetry

        mask = np.nonzero(np.isfinite(f_KM))[0]
        V = np.sum(self.W[0, mask]*abs(f_tau[mask] - f_KM[mask])**2) \
          + np.sum(self.W[1, mask]*abs(a_tau[mask] - a_KM[mask])**2)

        if kl_reg > 0:
            p_est = self.fp.solve(
                [np.reshape(np.real(f_vals), self.fp.N), np.reshape(np.imag(f_vals), self.fp.N)],
                [np.reshape(np.real(a_vals), self.fp.N), np.reshape(np.imag(a_vals), self.fp.N)]
            )

            kl = utils.kl_divergence(self.p_hist, p_est, dx=self.fp.dx, tol=1e-6)
            kl = max(0, kl)
            V += kl_reg*kl

        if not np.isfinite(V):
            print('Error in cost function')
            print(Xi)
            print(f)
            print(a)
            return None

        return V
        
        
class B_model(LRModel):
    def __init__(self, data, dt=1/225, nbins=40, domain_width=4, lag=200):
        super().__init__(data, dim=1, dt=dt)
        self.data = data
        self.nbins = nbins
        self.domain_width = domain_width
        
        self.compute_PDF()
        self.compute_KM(lag)
        self.precompute_libraries()
        self.initial_coeffs()
        self.initialize_fpsolve()
        
    def compute_PDF(self):
        self.bins = np.linspace(-self.domain_width, self.domain_width, self.nbins+1)
        dx = self.bins[1]-self.bins[0]
        self.centers1d = (self.bins[:-1]+self.bins[1:])/2

        # Histogram: [real, imag]
        self.p_hist = np.histogram(self.data, bins=self.bins, density=True)[0]

        
    def compute_KM(self, lag):
        self.tau = lag*self.dt
        B = self.data
        
        N = len(self.bins)-1
        self.f_KM = np.zeros((N))
        f_err = np.zeros(self.f_KM.shape)
        self.a_KM = np.zeros((self.f_KM.shape))
        a_err = np.zeros((self.f_KM.shape))

        dB = np.real(B[lag:] - B[:-lag])/self.tau  # Step (finite-difference derivative estimate)
        dB2 = self.tau*dB**2  # Variance

        for i in range(N):
            # Find where signal falls into this bin
            mask = (B[:-lag] > self.bins[i]) * (B[:-lag] < self.bins[i+1])
            mask_idx = np.nonzero(mask)[0]

            if len(mask_idx) > 0:
                # Conditional mean
                self.f_KM[i] = np.mean(dB[mask_idx]) # Conditional average
                f_err[i] = np.std(dB[mask_idx])/np.sqrt(len(mask_idx))

                # Conditional variance
                self.a_KM[i] = 0.5*np.mean(dB2[mask_idx]) # Conditional average
                a_err[i] = np.std(dB2[mask_idx])/np.sqrt(len(mask_idx))
            
            else:
                self.f_KM[i] = np.nan
                f_err[i] = np.nan
                self.a_KM[i] = np.nan
                a_err[i] = np.nan
                
        self.W = utils.KM_weights(f_err, a_err)
        
    def precompute_libraries(self):
        # Initialize sympy expression
        z = sympy.symbols('z')
        
        # Lists of candidate functions
        self.f_expr = np.array([z])
        self.s_expr = np.array([z**0, z**2])
        
        ZZ = self.centers1d
        LRModel.precompute_libraries(self, z, self.f_expr, self.s_expr, ZZ)
        
    def initial_coeffs(self):
        self.Xi0 = np.zeros((len(self.f_expr) + len(self.s_expr)))
        lhs = self.f_KM
        mask = np.nonzero(np.isfinite(lhs))[0]
        self.Xi0[:len(self.f_expr)] = lstsq( self.lib_f[mask, :], lhs[mask], rcond=None)[0]

        lhs = np.sqrt(2*self.a_KM)
        self.Xi0[len(self.f_expr):] = lstsq( self.lib_s[mask, :], lhs[mask], rcond=None)[0]
        
    def initialize_fpsolve(self):
        LRModel.initialize_fpsolve(self, centers=self.centers1d, dim=1)
        
    def cost(self, Xi, kl_reg):
        f_KM, a_KM = self.f_KM, self.a_KM

        Xi_f = Xi[:self.lib_f.shape[1]]
        Xi_s = Xi[self.lib_f.shape[1]:]

        f_vals = self.lib_f @ Xi_f
        s_vals = self.lib_s @ Xi_s
        a_vals = 0.5*s_vals**2

        # Solve adjoint Fokker-Planck equation
        self.afp.precompute_operator(f_vals, a_vals)
        f_tau, a_tau = self.afp.solve(self.tau, d=0)

        mask = np.nonzero(np.isfinite(f_KM))[0]
        V = np.sum(self.W[0, mask]*abs(f_tau[mask] - f_KM[mask])**2) \
          + np.sum(self.W[1, mask]*abs(a_tau[mask] - a_KM[mask])**2)

        if kl_reg > 0:
            p_est = self.fp.solve(f_vals, a_vals)

            kl = utils.kl_divergence(self.p_hist, p_est, dx=self.fp.dx, tol=1e-6)
            kl = max(0, kl)
            V += kl_reg*kl

        return V
        
        

class cop_model(B_model):
    
    def compute_PDF(self):
        self.bins = np.linspace(0, self.domain_width, self.nbins+1)
        dx = self.bins[1]-self.bins[0]
        self.centers1d = (self.bins[:-1]+self.bins[1:])/2

        # Histogram: [real, imag]
        self.p_hist = np.histogram(self.data, bins=self.bins, density=True)[0]
        
    def precompute_libraries(self):
        # Initialize sympy expression
        z = sympy.symbols('z')
        
        # Lists of candidate functions
        self.f_expr = np.array([z**i for i in [1, 3]])  # Polynomial library for drift
        self.s_expr = np.array([z**i for i in [0, 2]])  # Polynomial library for diffusion
        
        ZZ = self.centers1d
        LRModel.precompute_libraries(self, z, self.f_expr, self.s_expr, ZZ)
        
    def cost(self, Xi, kl_reg):
        f_KM, a_KM = self.f_KM, self.a_KM

        Xi_f = Xi[:self.lib_f.shape[1]]
        Xi_s = Xi[self.lib_f.shape[1]:]

        f_vals = self.lib_f @ Xi_f
        
        s_vals = self.lib_s @ Xi_s
        a_vals = 0.5*s_vals**2
        f_vals += a_vals/self.centers1d  # Diffusion-induced drift from polar change of variables

        # Solve adjoint Fokker-Planck equation
        self.afp.precompute_operator(f_vals, a_vals)
        f_tau, a_tau = self.afp.solve(self.tau, d=0)

        mask = np.nonzero(np.isfinite(f_KM))[0]
        V = np.sum(self.W[0, mask]*abs(f_tau[mask] - f_KM[mask])**2) \
          + np.sum(self.W[1, mask]*abs(a_tau[mask] - a_KM[mask])**2)

        if kl_reg > 0:
            p_est = self.fp.solve(f_vals, a_vals)

            kl = utils.kl_divergence(self.p_hist, p_est, dx=self.fp.dx, tol=1e-6)
            kl = max(0, kl)
            V += kl_reg*kl

        return V