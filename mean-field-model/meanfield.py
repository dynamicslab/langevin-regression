import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq

### GLOBAL VARIABLES
Uinf = 15              # freestream velocity [m/sec]
Pinf = 0.5*1.2041*15**2  # rho_air = 1.2041 (20C) -> Pinf = 135.4612
D = 196.5              # Diameter of the body [mm]

# location of the 64 pressure taps
rESP=np.linspace(11,88,8)/D            # non-dimensional with D
thetaESP= np.array([270, 315, 0, 45, 90, 135, 180, 225]) #[degrees]
nr = len(rESP)
nth = len(thetaESP)

# Sampling rate
fs = 225  # Hz   
dt = 1/fs


# Mass matrix
dth = 2*np.pi/8
dr  = rESP[1]-rESP[0]
w   = (rESP*dr)
W   = np.diag(w**0.5)  # Sqrt of mass matrix
W_inv = np.linalg.inv(W)
ip = lambda p, q: (W @ p).T @ (W @ q).conj()

# For full-field (nr*nth)
W_full   = np.repeat(rESP*dr*dth,8)
W_full   = np.diag(W_full**0.5)


def order_parameter(q):
    """
    Derive order parameter from integrated m=1 amplitudre
    q = [nr x nth x T]
    """
    qhat = fft(q, axis=1, norm='ortho')
    A = np.trapz( rESP * qhat[:, 1, :].T, dx=dr, axis=1)
    return A
    
def phase_align(q, A, rot_frame=False):
    """
    Phase-align with order parameter
    q = [nr x nth x T]
    """
    qhat = fft(q, axis=1, norm='ortho')
    m = fftfreq(nth, d=1/nth)  # Azimuthal wavenumber
    qhat_rot = np.zeros_like(qhat)
    phi = np.angle(A)
    for k in range(nth):
        if rot_frame:
            # Align with rotating frame
            qhat_rot[:, k, :] = qhat[:, k, :]*np.exp(-1j*np.sign(m[k])*phi)
        else:
            # Full phase alignment
            qhat_rot[:, k, :] = qhat[:, k, :]*np.exp(-1j*m[k]*phi)
    return np.real(ifft(qhat_rot, axis=1, norm='ortho'))


def cond_avg(q, A, edges):
    qd = np.zeros([nr, nth, len(edges)-1])
    qd_err = np.zeros_like(qd)
    for i in range(len(edges)-1):
        mask = np.nonzero((abs(A)>edges[i]) * (abs(A)<edges[i+1]))[0]
        qd[:, :, i] = np.mean(q[:, :, mask], axis=2)
        qd_err[:, :, i] = np.std(q[:, :, mask], axis=2)/np.sqrt(len(mask))
        
    return qd, qd_err

def err_full(p, p_est):
    ip = lambda p, q: (W_full @ p).T @ (W_full @ q).conj()
    dp = p - p_est
    norm = lambda p: ( ip(p.flatten('F'), p.flatten('F')) )
    if len(p.shape)==3:
        return np.array( [norm(dp[:, :, t_idx])/norm(p[:, :, t_idx]) for t_idx in range(p.shape[2])] )
    else:
        return norm(dp)/norm(p)
    
    
"""
Plotting functions
"""


import matplotlib.pyplot as plt
from scipy.interpolate import griddata

### INTERPOLATION FOR PLOTTING
TT, RR = np.meshgrid(thetaESP, rESP)
XX = RR * np.cos(2*np.pi*TT/360)
YY = RR * np.sin(2*np.pi*TT/360)

theta_interp = np.linspace(0, 2*np.pi, 100)
r_interp = np.linspace(0, 0.41, 100)
TT_interp, RR_interp = np.meshgrid(theta_interp, r_interp)
XX_interp = RR_interp * np.cos(TT_interp)
YY_interp = RR_interp * np.sin(TT_interp)
    
def plot(q, colorbar=True, vmin=None, vmax=None, cm="RdGy"):
    field = griddata( (XX.flatten(), YY.flatten()), q.flatten(), (XX_interp, YY_interp), method='cubic')
    plt.pcolormesh(XX_interp, YY_interp, field,
                   cmap=cm, shading='gouraud', vmin=vmin, vmax=vmax)
    if colorbar: plt.colorbar()
    plt.scatter(XX.flatten('F'), YY.flatten('F'), c='k', s=5)

    cyl = plt.Circle((0, 0), 0.5, edgecolor='k', facecolor='none', ls='--')
    plt.gcf().gca().add_artist(cyl)
    plt.xlim([-0.55, 0.55])
    plt.ylim([-0.55, 0.55])
    plt.xticks([])
    plt.yticks([])
