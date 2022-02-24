"""
Package to solve Fokker-Planck equations

* Steady-state Fourier-space solver for the PDF
    - Galerkin projection for 1D/2D
    - Arnoldi iteration for higher dimensions (slower but easier to scale)
* Adjoint finite difference solver for first/second moments

Jared Callaham (2020)
"""

import numpy as np
from numpy.fft import fft, ifft, fftn, fftfreq, ifftn
from scipy import linalg, sparse

from scipy.sparse.linalg import LinearOperator, eigs
import utils

class SteadyFP:
    """
    Solver object for steady-state Fokker-Planck equation

    Initializing this independently avoids having to re-initialize all of the indexing arrays
      for repeated loops with different drift and diffusion

    Jared Callaham (2020)
    """

    def __init__(self, x, ndim=1, method=None):
        """
        ndim - number of dimensions
        N - array of ndim ints: grid resolution N[0] x N[1] x ... x N[ndim-1]
        dx - grid spacing (array of floats)
        
        method - Galerkin or Arnoldi
            if not selected, defaults to Galerkin for ndim < 3, Arnoldi otherwise
        """
        self.ndim = ndim
        
        if (method is None) and (ndim < 3):
            method = "galerkin"
        elif (method is None):
            method = "arnoldi"
        
        if self.ndim == 1:
            self.N = len(x)
            self.dx = x[1]-x[0]
            self.x = x
            self.k = 2*np.pi*fftfreq(self.N, self.dx)
        else:
            self.x = x
            self.N = [len(x[i]) for i in range(len(x))]
            self.dx = [x[i][1]-x[i][0] for i in range(len(x))]
            self.k = [2*np.pi*fftfreq(self.N[i], self.dx[i]) for i in range(self.ndim)]
            self.KK = np.meshgrid(*self.k, indexing='ij')
            
        if method=="arnoldi":
            self.solve = self.arnoldi_solve
        elif method=="galerkin":
            self.galerkin_init()
            self.solve = self.galerkin_solve
        else:
            print("Not implemented")
         
    def FP_operator(self, p, f, a):
        """
        Linear Fokker-Planck operator with functions for f and a
        
        q = L*p
        """
        
        p = np.reshape(p, self.N)
        q = np.sum(np.array([ ifft(-1j*self.KK[d]*fft(f[d]*p, axis=d) - self.KK[d]**2*fft(a[d]*p, axis=d), axis=d)
                                    for d in range(self.ndim) ]), axis=0)
        return q.flatten()
    
    def arnoldi_operator(self, f, a):
        self.L = LinearOperator((np.prod(self.N), np.prod(self.N)),
                           matvec=lambda p: self.FP_operator(p, f, a))
        
    def arnoldi_solve(self, f, a):
        """
        Solve Fokker-Planck equation from input drift/diffusion coefficients
        
        Higher-dimensional version using Arnoldi iteration to estimate leading eigenvector
        """
        self.arnoldi_operator(f, a)
        
        evals, evecs = eigs(self.L, k=1, which="LR")
        p_sol = np.reshape( np.real(evecs[:, 0]), self.N)
        p_sol /= utils.ntrapz(p_sol, self.dx)
        return p_sol
    
    
    def galerkin_init(self):
        # Set up indexing matrices for ndim=1, 2
        if self.ndim == 1:
            self.k = 2*np.pi*fftfreq(self.N, self.dx)
            self.idx = np.zeros((self.N, self.N), dtype=np.int32)
            for i in range(self.N):
                self.idx[i, :] = i-np.arange(self.N)

        elif self.ndim == 2:
            # Fourier frequencies
            self.k = [2*np.pi*fftfreq(self.N[i], self.dx[i]) for i in range(self.ndim)]
            self.idx = np.zeros((2, self.N[0], self.N[1], self.N[0], self.N[1]), dtype=np.int32)
            
            for m in range(self.N[0]):
                for n in range(self.N[1]):
                    self.idx[0, m, n, :, :] = m-np.tile(np.arange(self.N[0]), [self.N[1], 1]).T
                    self.idx[1, m, n, :, :] = n-np.tile(np.arange(self.N[1]), [self.N[0], 1])

        else:
            print("WARNING: NOT IMPLEMENTED FOR HIGHER DIMENSIONS - USE ARNOLDI")
        
    def galerkin_operator(self, f, a):
        """
        f - array of drift coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        a - array of diffusion coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        NOTE: To generalize to covariate noise, would need to add a dimension to a
        """
        
        if self.ndim == 1:
            f_hat = self.dx*fftn(f)
            a_hat = self.dx*fftn(a)

            # Set up spectral projection operator
            self.L = np.einsum('i,ij->ij', -1j*self.k, f_hat[self.idx]) \
                   + np.einsum('i,ij->ij', -self.k**2, a_hat[self.idx])

        """
        # KEEP FOR REFERENCE: naive implementation with N^4 loops  (~3:30 for N=64, ~14s for N=32)

        A = np.zeros((Nx, Ny, Nx, Ny), dtype=np.complex64)
        # First two loops are over projected variables (k')
        for m in range(Nx):
            for n in range(Ny):
                # Then loop over k
                for i in range(Nx):
                    for j in range(Ny):
                        A[m, n, i, j] = -1j*(kx[m]*f_hat[0, m-i, n-j] + ky[n]*f_hat[1, m-i, n-j])
                        A[m, n, i, j] -= (kx[m]**2*a_hat[0, m-i, n-j] + ky[n]**2*a_hat[1, m-i, n-j])
        """
        if self.ndim == 2:
            # Initialize Fourier transformed coefficients
            f_hat = np.zeros(np.append([self.ndim], self.N), dtype=np.complex64)
            a_hat = np.zeros(f_hat.shape, dtype=np.complex64)
            for i in range(self.ndim):
                f_hat[i] = np.prod(self.dx)*fftn(f[i])
                a_hat[i] = np.prod(self.dx)*fftn(a[i])

            self.L = -1j*np.einsum('i,ijkl->ijkl', self.k[0], f_hat[0, self.idx[0], self.idx[1]]) \
                     -1j*np.einsum('j,ijkl->ijkl', self.k[1], f_hat[1, self.idx[0], self.idx[1]]) \
                     -np.einsum('i,ijkl->ijkl', self.k[0]**2, a_hat[0, self.idx[0], self.idx[1]]) \
                     -np.einsum('j,ijkl->ijkl', self.k[1]**2, a_hat[1, self.idx[0], self.idx[1]])

            self.L = np.reshape(self.L, (np.prod(self.N), np.prod(self.N)))
        
        
    def galerkin_solve(self, f, a):
        """
        Solve Fokker-Planck equation from input drift coefficients
        
        Fourier-Galerkin projection for fast solving in 1 or 2 dimensions
        """
        self.galerkin_operator(f, a)
        
        q_hat = np.linalg.lstsq(self.L[1:, 1:], -self.L[1:, 0], rcond=1e-6)[0]
        q_hat = np.append([1], q_hat)
        return np.real(ifftn( np.reshape(q_hat, self.N) ))/np.prod(self.dx)
        



class AdjFP:
    """
    Solver object for adjoint Fokker-Planck equation

    Jared Callaham (2020)
    """
    
    @staticmethod
    def unit(N, idx):
        """
        Construct an n-dimensional "unit matrix"
           where the only nonzero element is given by idx
        """
        # Here idx should be a linear idx (output will be reshaped to matrix)
        e = np.zeros(np.prod(N))
        e[idx] = 1
        return np.reshape(e, N)


    @staticmethod
    def deriv1(u, dx, axis=0, bc="extrapolate"):
        """
        First derivative of u
            Extrapolated boundary conditions

        u = n-dimensional array
        dx = n-length delta-x values
        dim = dimension to differentiate
        bc - boundary conditions
        """
        u = np.moveaxis(u, axis, 0)  # Move so that the axis being differentiated is first
        du = np.zeros_like(u)

        du[1:-1] = (u[2:]-u[:-2])  # Centered

        if bc=="extrapolate":
            du[0] = -3*u[0]+4*u[1]-u[2]
            du[-1] = 3*u[-1]-4*u[-2]+u[-3]
        elif bc=="dirichlet":
            du[0] = np.ones_like(u[0])
            du[-1] = np.ones_like(u[-1])
        else:
            raise NotImplementedError

        u = np.moveaxis(u, 0, axis)  # Move axis back
        du = np.moveaxis(du, 0, axis)/(2*dx[axis])
        return du

    @staticmethod
    def deriv2(u, dx, axis=0, bc="extrapolate"):
        """
        Second derivative of u
            Extrapolated boundary conditions

        u = n-dimensional array
        dx = n-length delta-x values
        dim = dimension to differentiate
        """
        u = np.moveaxis(u, axis, 0)  # Move so that the axis being differentiated is first
        du = np.zeros_like(u)

        du[1:-1] = (u[2:]-2*u[1:-1]+u[:-2])  # Centered

        if bc=="extrapolate":
            du[0] = 2*u[0]-5*u[1]+4*u[2]-u[3]
            du[-1] = 2*u[-1]-5*u[-2]+4*u[-3]-u[-4]
        elif bc=="dirichlet":
            du[0] = np.ones_like(u[0])
            du[-1] = np.ones_like(u[-1])
        else:
            raise NotImplementedError

        u = np.moveaxis(u, 0, axis)  # Move axis back to original location
        du = np.moveaxis(du, 0, axis)/(dx[axis]**2)
        return du

    @staticmethod
    def fd_op(N, dx, deriv, axis, bc="extrapolate"):
        """
        Compute sparse derivative operator by evaluating finite-differences on "unit matrices"
        deriv - finite-differencing function (deriv1 or deriv2)
        axis - axis along which to differentiate
        """
        row_idx = []
        col_idx = []
        entries = []

        for k in range(np.prod(N)):
            col = deriv(AdjFP.unit(N, k), dx, axis=axis, bc=bc).flatten()  # "Unit vector"
            cur_idx = np.flatnonzero(col)
            row_idx = np.concatenate([row_idx, cur_idx])
            col_idx = np.concatenate([col_idx, k*np.ones_like(cur_idx)])  # Current column for entries
            entries = np.concatenate([entries, col[cur_idx]])

        D = sparse.coo_matrix((entries, (row_idx, col_idx)), shape=(np.prod(N), np.prod(N)))
        return sparse.csc_matrix(D)
    

    def __init__(self, x, ndim=1, method="step"):
        """
        x - uniform grid (array of floats)
        """
        self.ndim = ndim
        
        if self.ndim == 1:
            self.N = [len(x)]
            self.dx = [x[1]-x[0]]
            self.x = [x]
        else:
            self.x = x
            self.N = [len(x[i]) for i in range(len(x))]
            self.dx = [x[i][1]-x[i][0] for i in range(len(x))]
            
        # Precompute and store sparse finite-difference matrices
        self.precompute_matrices()

        # Compute grid
        self.XX = np.meshgrid(*self.x, indexing='ij')
        self.XX = [XX.flatten() for XX in self.XX]
        
        if method=="step":
            self.solve = self.step_solve  # Euler time-stepping
        elif method=="exp":
            self.solve = self.exp_solve   # Matrix exponential
        

    def precompute_matrices(self):
        self.Dx = [AdjFP.fd_op(self.N, self.dx, AdjFP.deriv1, axis) for axis in range(self.ndim)]
        self.Dxx = [AdjFP.fd_op(self.N, self.dx, AdjFP.deriv2, axis) for axis in range(self.ndim)]
    
    def precompute_operator(self, f, a):
        if self.ndim==1:
            f, a = [f], [a]
        self.L = np.sum([ sparse.diags(f[i]) @ self.Dx[i] + sparse.diags(a[i]) @ self.Dxx[i]
                                 for i in range(self.ndim)])
        
        
    def exp_solve(self, tau, dt=None, d=0):
        """
        Solve with matrix exponential
        """
        if self.L is None:
            print("Need to initialize operator")
            return None
        
        L_tau = linalg.expm(self.L.todense()*tau)
        
        w1 = L_tau @ self.XX[d]
        w2 = L_tau @ self.XX[d]**2

        f_tau = (w1 - self.XX[d])/tau
        a_tau = (w2 - 2*self.XX[d]*w1 + self.XX[d]**2)/(2*tau)
        
        return f_tau, a_tau

    def step_solve(self, tau, dt, d=0):
        """
        Solve with forward Euler time-stepping
        """
        # Evolve observables (zero-centered moments)
        w1 = self.XX[d].copy()
        w2 = self.XX[d].copy()**2
        for i in range(int(tau//dt)):
            w1 += dt*(self.L @ w1)
            w2 += dt*(self.L @ w2)
            
        # Evaluate finite-time moments for drift/diffusion
        f_tau = (w1 - self.XX[d])/tau
        a_tau = (w2 - 2*self.XX[d]*w1 + self.XX[d]**2)/(2*tau)
        return f_tau, a_tau
    
    
    