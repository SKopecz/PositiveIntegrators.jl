# The modified Patankar-Euler (MPE) scheme is a first order scheme which is unconditionally positive
# and conservative when applied to a positive and conservative PDS.
# For linear PDS the MPE schemes is equivalent to the implicit Euler scheme.
#
# Literature:
# 1) A high-order conservative Patankar-type discretisation for stiff systems of production-destruction
#    equations; Burchard et al.; APNUM 2003
# 2) On order conditions for modified Patankar-Runge-Kutta schemes; S. Kopecz, A. Meister; APNUM 2018
#
# In this script we use the MPE scheme to solve a linear PDS and check that the solution is equivalent
# to the solution of the implicit Euler scheme.

# Install packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()

# load new problem type for production-destruction systems and MPRK algorithms
using PositiveIntegrators

using OrdinaryDiffEq

# load utility function for the assesment of the order of numerical schemes
include("utilities.jl")

# problem data
u0 = [0.9, 0.1]
tspan = (0.0, 2.0)
p = [5.0, 1.0]

# linear model problem - out-of-place
linmodP(u,p,t) = [0.0 p[2]*u[2]; p[1]*u[1] 0.0]
linmodD(u,p,t) = [0.0; 0.0]
# analytic solution
function f_analytic(u0,p,t)
    u₁⁰, u₂⁰ = u0
    a, b = p
    c = a+b
    return ((u₁⁰+u₂⁰)*[b; a] + exp(-c*t)*(a*u₁⁰-b*u₂⁰)*[1;-1])/c
end
PD_op = ProdDestFunction(linmodP, linmodD; analytic=f_analytic)
prob_op = ProdDestODEProblem(PD_op, u0, tspan, p)

#=
# solution
sol_MPE_op = solve(prob_op, MPE(), dt=0.25)
sol_IE_op = solve(prob_op, ImplicitEuler(), dt=0.25, adaptive=false)

# check that MPE and IE solutions are equivalent
@assert sol_MPE_op.u ≈ sol_IE_op.u

# plot solutions
using Plots
p1 = myplot(sol_MPE_op,"MPE",true)
p2 = myplot(sol_IE_op,"IE",true)
plot(p1,p2,plot_title="out-of-place")

# check convergence order
using DiffEqDevTools, PrettyTables
convergence_tab_plot(prob_op, [MPE(); ImplicitEuler()])
=#

# linear model problem - in-place
function linmodP!(P,u,p,t)
    P .= 0
    P[1, 2] = u[2]
    P[2, 1] = 5.0*u[1]
    return nothing
end
function linmodD!(D,u,p,t)
    D .= 0
    return nothing
end
PD_ip = ProdDestFunction(linmodP!,linmodD!, p_prototype=zeros(2,2), d_prototype=zeros(2,1),
            analytic=f_analytic)
#BUG: prob_ip cannot be solved if prototypes are not given.
prob_ip = ProdDestODEProblem(PD_ip, u0, tspan, p)

#solutions
sol_MPE_ip = solve(prob_ip, MPE(), dt=0.25)
sol_IE_ip = solve(prob_ip,ImplicitEuler(autodiff=false), dt=0.25, adaptive=false) #autodiff does not work here

# check that MPE and IE solutions are equivalent
@assert sol_MPE_ip.u ≈ sol_IE_ip.u

# plots solutinos
p1 = myplot(sol_MPE_ip, "MPE", true)
p2 = myplot(sol_IE_ip, "IE", true)
plot(p1, p2, plot_title="in-place")

# check convergence order
convergence_tab_plot(prob_ip, [MPE(); ImplicitEuler(autodiff=false)])

# try different linear solvers
using LinearSolve
sol_MPE_ip_linsol1 = solve(prob_ip, MPE(), dt=0.25)
sol_MPE_ip_linsol2 = solve(prob_ip, MPE(linsolve=RFLUFactorization()), dt=0.25)
sol_MPE_ip_linsol3 = solve(prob_ip, MPE(linsolve=LUFactorization()), dt=0.25)
@assert sol_MPE_ip_linsol1.u ≈ sol_MPE_ip_linsol2.u ≈ sol_MPE_ip_linsol3.u