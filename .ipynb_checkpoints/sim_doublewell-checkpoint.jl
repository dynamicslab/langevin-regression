"""
Julia code to simulate 1D particle in a double-well potential

Jared Callaham (2020)
"""
using DifferentialEquations

# Noisy double well potential

### Physical dynamics
function f_fn(dx, x, p, t)
    ϵ, σ = p
    dx[1] = x[2]
    dx[2] = -2*x[2] + ϵ*x[1] - x[1]^3
end

function σ_fn(dx, x, p, t)
    ϵ, σ = p
    dx[1] = 0
    dx[2] = σ
end

### Normal form dynamics (pitchfork bifurcation)
function fn_fn(dx, x, p, t)
    λ, μ, σ = p
    dx[1] = λ*x[1] + μ*x[1]^3
end

function σn_fn(dx, x, p, t)
    λ, μ, σ = p
    dx[1] = σ
end

function run_nf(ϵ, μ, σ, dt, tmax)
    prob = SDEProblem(fn_fn, σn_fn, [0.0] , (0.0, tmax), [ϵ, μ, σ])
    sol = solve(prob, SRIW1(), dt=dt, adaptive=false);
    return sol.t, sol[:, :]
end

function run_sim(η, σ, dt, tmax)
    prob = SDEProblem(f_fn,σ_fn, [0.0, 0.0], (0.0, tmax), [η, σ])
    sol = solve(prob, SRIW1(), dt=dt, adaptive=false)
    return sol.t, sol[:, :]
end