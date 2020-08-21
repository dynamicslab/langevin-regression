# Nonlinear stochastic modeling with Langevin regression
___________
__J. L. Callaham, J.-C. Loiseau, G. Rigas, and S. L. Brunton (2020)__

Code and details for the manuscript on data-driven stochastic modeling with Langevin regression. The repository should have everything needed to reproduce the results of the simulated systems in the paper and demonstrate the basics of Langevin regression. The main ideas are:

1. Estimating drift and diffusion from data with Kramers-Moyal averaging
2. Correcting finite-time sampling effects with the adjoint Fokker-Planck equation
3. Enforcing consistency with the empirical PDF via the steady-state Fokker-Planck equation

The full optimization problem combines all of these, along with a sparsification routine called SSR that was [previously proposed](https://arxiv.org/abs/1712.02432) to extend [SINDy](https://github.com/dynamicslab/pysindy) to stochastic systems.

The repository several Python packages:

* `utils.py`: A number of functions to do the main work of Langevin regression.  There is code to compute the Kramers-Moyal average, the Langevin regression "cost function", a wrapper around the Nelder-Mead optimization, an implementation of SSR, etc.
*  `fpsolve.py`: A library of Fokker-Planck solvers (steady-state and adjoint).  The details are in Appendix B of the paper.
*  `dwutils.py`: Several utility functions specifically for the notebook on the 1D double-well potential (see below).  These do things like compute the dwell time distribution for the metastable states.

To demonstrate, we include two notebooks for the simulated examples in the paper:

### 1. Pitchfork normal form driven by colored noise

To explore the effects of correlated forcing and finite-time sampling, we look at the bistable normal form of a pitchfork bifurcation, driven by an Ornstein-Uhlenbeck process:
$$
\dot{x} = \lambda x - \mu x^3 + \eta
$$
$$
\dot{\eta} = - \alpha \eta + \sigma w(t),
$$
where $w(t)$ is a white noise process (see paper for details).
With coarse sampling rates $\tau \gg \alpha^{-1}$ and adjoint corrections, we show that a statistically consistent model can be identified with only white noise forcing.

### 2. One-dimensional particle in a double-well potential

The true dynamics of the system are given by the second-order Langevin equation
$$
\ddot{x} + \gamma \dot{x} + U'(x) = \sqrt{2 \gamma \kB T} w(t),
$$
with double-well potential
$$
U(x) = -\frac{\alpha}{2} x^2 + \frac{\beta}{4} x^4.
$$
See the paper for nondimensionalization and further discussion.

On short time scales, the dynamics of the state $x$ are smooth, since the direct white noise forcing is integrated twice.  However, the macroscopic behavior displays metastable switching similar to the pitchfork normal form. In fact, for the weakly supercritical system we show in Appendix C that a stochastic normal form model can be derived analytically.

Far from the bifurcation we can continue to model the macroscopic dynamics, but we have to resort to data-driven methods.
This notebook demonstrates the use of Langevin regression to reduce the dynamics to a first-order system with consistent statistics over a wide range of parameters.


References
----------------------
-  Jared L. Callaham,
   Jean-Christophe Loiseau,
   Georgios Rigas,
   and Steven L. Brunton
   *Nonlinear stochastic modeling with Langevin regression.* arXiv preprint arXiv:2004.08424 (2020)
   `[arXiv] <https://arxiv.org/abs/2004.08424>`_

-  Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz.
   *Discovering governing equations from data by sparse identification
   of nonlinear dynamical systems.* Proceedings of the National
   Academy of Sciences 113.15 (2016): 3932-3937.
   `[DOI] <http://dx.doi.org/10.1073/pnas.1517384113>`__