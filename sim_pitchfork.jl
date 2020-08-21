"""
Julia code to simulate pitchfork bifurcation normal form forced by colored noise

Jared Callaham (2020)
"""
using DifferentialEquations
using MAT

# Drift component
μ = 1  # Fixed points at +/- 1
λ = 1

r0 = sqrt(λ/μ) # Equilibrium point
α = Int(1e2)  # Inverse autocorrelation time of the forcing (smaller than lambda for scale separation)
σ = α*0.5

f_ex(x) = λ*x - μ*x^3

# Simulate SDE
function f_fn(dx, x, p, t)
    dx[1] = f_ex(x[1]) + x[2]
    dx[2] = -α*x[2]
end

function σ_fn(dx, x, p, t)
    dx[1] = 0
    dx[2] = σ
end

dt = 0.001
x0 = [0.0, 0.0]
tspan = (0.0, Int(1e4))
prob = SDEProblem(f_fn, σ_fn, x0, tspan)

# EM() or SRIW1()
sol = solve(prob, EM(), dt=dt);
t = sol.t
X = sol[1, :]

matwrite("./data/pitchfork.mat", Dict(
    "X" => X,
    "dt" => dt,
    "lamb" => λ,
    "mu" => μ
); compress = true)