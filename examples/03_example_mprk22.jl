# The family of MPRK22 schemes are second order schemes which are unconditionally positive
# and conservative when applied to a positive and conservative PDS. Numerical experiments show that
# that these schemes can be used to integrate stiff problems. The Patankar-weight σ used in the
# approximation step is first order accurate and thus σ can be used to estimate the error in an adaptive
# timestepping algorithm without additional cost.
#
# Literature:
# On order conditions for modified Patankar-Runge-Kutta schemes; S. Kopecz, A. Meister; APNUM 2018
#
# In this script we use the several MPRK22 schemes to solve non-stiff and stiff PDS.

# Install packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()

# load new problem type for production-destruction systems and MPRK algorithms
using PositiveIntegrators

using OrdinaryDiffEq

# load utility functions
include("utilities.jl")

### linear model problem ######################################################################################

# problem data
u0 = [0.9,0.1]
tspan = (0.0, 1.5)
p = [5.0,1.0]

# analytic solution
function f_analytic(u0,p,t)
    y₁⁰ ,y₂⁰ = u0
    a, b = p
    c = a+b
    return ((y₁⁰+y₂⁰)*[b; a] + exp(-c*t)*(a*y₁⁰-b*y₂⁰)*[1;-1])/c
end

# out-of-place implementation
linmodP(u,p,t) = [0.0 p[2]u[2]; p[1]*u[1] 0.0]
linmod_op = ConservativePDSProblem(linmodP, u0, tspan, p; analytic=f_analytic)

# solutions with constant equispaced time steps
dt0 = 0.25
sol_linmod_op_MPE = solve(linmod_op, MPE(), dt=dt0);
sol_linmod_op_MPRK22_1 = solve(linmod_op, MPRK22(1.0), dt=dt0, adaptive=false);
sol_linmod_op_MPRK22_½ = solve(linmod_op, MPRK22(0.5), dt=dt0, adaptive=false);

# plots
using Plots
p1 = myplot(sol_linmod_op_MPE, "MPE", true)
p2 = myplot(sol_linmod_op_MPRK22_1, "MPRK22(1.0)", true)
p3 = myplot(sol_linmod_op_MPRK22_½, "MPRK22(0.5)", true)
plot(p1,p2,p3,plot_title="oop")

# convergence order
using DiffEqDevTools
using PrettyTables
convergence_tab_plot(linmod_op,[MPE(); MPRK22(0.5); MPRK22(1.0)])

# in-place implementation
function linmodP!(P,u,p,t)
    P .= 0
    P[1, 2] = u[2]
    P[2, 1] = 5.0*u[1]
    return nothing
end
linmod_ip = ConservativePDSProblem(linmodP!, u0, tspan, p; analytic=f_analytic)

# solutions with constant equispaced time steps
dt0 = 0.25
sol_linmod_ip_MPE = solve(linmod_ip, MPE(), dt=dt0);
sol_linmod_ip_MPRK22_1 = solve(linmod_ip, MPRK22(1.0), dt=dt0, adaptive=false);
sol_linmod_ip_MPRK22_½ = solve(linmod_ip, MPRK22(0.5), dt=dt0, adaptive=false);

# plots
using Plots
p1 = myplot(sol_linmod_ip_MPE, "MPE", true)
p2 = myplot(sol_linmod_ip_MPRK22_1, "MPRK22(1.0)", true)
p3 = myplot(sol_linmod_ip_MPRK22_½, "MPRK22(0.5)", true)
plot(p1,p2,p3,plot_title="in-place")

# convergence order
using DiffEqDevTools
using PrettyTables
convergence_tab_plot(linmod_ip,[MPE(); MPRK22(0.5); MPRK22(1.0)])

###############################################################################################################
### nonlinear model problem - non stiff

# problem data
u0 = [9.98; 0.01; 0.01]
tspan = (0.0, 30.0)

# out-of-place implementation
nonlinmodP(u,p,t) = [0.0 0.0 0.0; u[2]*u[1]/(u[1]+1.0) 0.0 0.0; 0.0 0.3*u[2] 0.0]
nonlinmod_op = ConservativePDSProblem(nonlinmodP, u0, tspan, p)

# solutions with constant equispaced time steps
dt0 = 1.0
sol_nonlinmod_op_Tsit5 = solve(nonlinmod_op, Tsit5(), dt=dt0, abstol=1e-2, reltol=1e-3);
sol_nonlinmod_op_MPRK22_1 = solve(nonlinmod_op, MPRK22(1.0), dt=dt0, abstol=1e-2, reltol=1e-3);
sol_nonlinmod_op_MPRK22_½ = solve(nonlinmod_op, MPRK22(0.5), dt=dt0, abstol=1e-2, reltol=1e-3);

# plots
using Plots
p1 = myplot(sol_nonlinmod_op_Tsit5, "Tsit5")
p2 = myplot(sol_nonlinmod_op_MPRK22_1, "MPRK22(1.0)")
p3 = myplot(sol_nonlinmod_op_MPRK22_½, "MPRK22(0.5)")
plot(p1,p2,p3,plot_title="oop")

# in-place implementation
function nonlinmodP!(P,u,p,t)
    P .= 0
    P[2, 1] = u[2]*u[1]/(u[1]+1.0)
    P[3, 2] = 0.3*u[2]
    return nothing
end
nonlinmod_ip = ConservativePDSProblem(nonlinmodP!, u0, tspan, p)

# solutions with constant equispaced time steps
dt0 = 1.0
sol_nonlinmod_ip_Tsit5 = solve(nonlinmod_ip, Tsit5(), dt=dt0, abstol=1e-2, reltol=1e-3);
sol_nonlinmod_ip_MPRK22_1 = solve(nonlinmod_ip, MPRK22(1.0), dt=dt0, abstol=1e-2, reltol=1e-3);
sol_nonlinmod_ip_MPRK22_½ = solve(nonlinmod_ip, MPRK22(0.5), dt=dt0, abstol=1e-2, reltol=1e-3);

# plots
using Plots
p1 = myplot(sol_nonlinmod_ip_Tsit5, "Tsit5")
p2 = myplot(sol_nonlinmod_ip_MPRK22_1, "MPRK22(1.0)")
p3 = myplot(sol_nonlinmod_ip_MPRK22_½, "MPRK22(0.5)")
plot(p1,p2,p3,plot_title="in-place")
###############################################################################################################

###############################################################################################################
### robertson problem - stiff

# problem data
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 1.0e11)
p = [1e4,4e-2, 3e7]

# out-of-place
robertsonP(u,p,t) = [0.0 1e4*u[2]*u[3] 0.0; 4e-2*u[1] 0.0 0.0; 0.0 3e7*u[2]^2 0.0]
robertson_op = ConservativePDSProblem(robertsonP, u0, tspan, p)

# Test constant time step size
sol_robertson_op_RB23 = solve(robertson_op, Rosenbrock23(), abstol=1e-3, reltol=1e-2);
sol_robertson_op_MPRK22_1 = solve(robertson_op, MPRK22(1.1), abstol=1e-3, reltol=1e-2);

# plot solution
p1 = plot(sol_robertson_op_RB23[2:end],idxs = [(0, 1), ((x,y)-> (x,1e4.*y) , 0, 2), (0, 3)], color=palette(:default)[1:3]',legend=:right, xaxis=:log)
plot!(sol_robertson_op_RB23[2:end],idxs = [(0, 1), ((x,y)-> (x,1e4.*y) , 0, 2), (0, 3)], color=palette(:default)[1:3]',denseplot=false,markershape=:circle,markerstrokecolor=palette(:default)[1:2]',linecolor = invisible(),label="")
title!("Rosenbrock23")
p2 = plot(sol_robertson_op_MPRK22_1[2:end],idxs = [(0, 1), ((x,y)-> (x,1e4.*y) , 0, 2), (0, 3)], color=palette(:default)[1:3]',legend=:right, xaxis=:log)
plot!(sol_robertson_op_MPRK22_1[2:end],idxs = [(0, 1), ((x,y)-> (x,1e4.*y) , 0, 2), (0, 3)], color=palette(:default)[1:3]',denseplot=false,markershape=:circle,markerstrokecolor=palette(:default)[1:2]',linecolor = invisible(),label="")
title!("MPRK22(1)")
plot(p1, p2, plot_title="oop")

# in-place
function robertsonP!(P,u,p,t)
    P .= 0
    P[1, 2] = 1e4*u[2]*u[3]
    P[2, 1] = 4e-2*u[1]
    P[3, 2] = 3e7*u[2]^2
    return nothing
end
robertson_ip = ConservativePDSProblem(robertsonP!, u0, tspan, p)

# Test constant time step size
sol_robertson_ip_RB23 = solve(robertson_ip, Rosenbrock23(autodiff=false), abstol=1e-3, reltol=1e-2);
sol_robertson_ip_MPRK22_1 = solve(robertson_ip, MPRK22(1.1), abstol=1e-3, reltol=1e-2);

# plot solution
p1 = plot(sol_robertson_op_RB23[2:end],idxs = [(0, 1), ((x,y)-> (x,1e4.*y) , 0, 2), (0, 3)], color=palette(:default)[1:3]',legend=:right, xaxis=:log)
plot!(sol_robertson_op_RB23[2:end],idxs = [(0, 1), ((x,y)-> (x,1e4.*y) , 0, 2), (0, 3)], color=palette(:default)[1:3]',denseplot=false,markershape=:circle,markerstrokecolor=palette(:default)[1:2]',linecolor = invisible(),label="")
title!("Rosenbrock23")
p2 = plot(sol_robertson_op_MPRK22_1[2:end],idxs = [(0, 1), ((x,y)-> (x,1e4.*y) , 0, 2), (0, 3)], color=palette(:default)[1:3]',legend=:right, xaxis=:log)
plot!(sol_robertson_op_MPRK22_1[2:end],idxs = [(0, 1), ((x,y)-> (x,1e4.*y) , 0, 2), (0, 3)], color=palette(:default)[1:3]',denseplot=false,markershape=:circle,markerstrokecolor=palette(:default)[1:2]',linecolor = invisible(),label="")
title!("MPRK22(1)")
plot(p1, p2, plot_title="oop")

###############################################################################################################