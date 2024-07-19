# [Tutorial: Solution of Robertson problem](@id tutorial-robertson)

In this tutorial we show that MPRK schemes can be used to integrate stiff problems. 
We also show how callbacks can be used to change the time step in non-adaptive schemes.

## Definition of the production-destruction system

The well known Robertson problem is given by
```math
\begin{aligned}
u_1' &= -0.04u_1+10^4 u_2u_3, & u_1(0)&=1,\\
u_2' &=  0.04u_1-10^4 u_2u_3-3⋅10^7 u_2^2, & u_2(0)&=0 \\
u_3' &= 3⋅10^7 u_2^2, & u_3(0)&=0.
\end{aligned}
```
The time domain of interest is ``t\in[0,10^{11}]``, which in general requires adaptive time stepping. 

The model can be represented as a conservative PDS with production terms
```math
\begin{aligned}
p_{12} &= 10^4u_2u_3,&
p_{21} &= 0.04u_1, &
p_{32} &= 3⋅10^7u_2^2.
\end{aligned}
```
Since the PDS is conservative, we have ``d_{i,j}=p_{j,i}`` and the system is fully determined by the production matrix ``\mathbf P``.

## Solution of the production-destruction system

Now we are ready to define a `ConservativePDSProblem` and to solve this problem with any method of [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl) or [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) which is suited for stiff problems.

Since this PDS consists of only three differential equations we provide an out-of-place implementation for the production matrix. Furthermore, we use static arrays for additional efficiency. See also the tutorial on [the solution of an NPZD model](@ref tutorial-npzd).

```@example robertson
using PositiveIntegrators, StaticArrays

function prod(u, p, t)
    @SMatrix [0.0 1e4*u[2]*u[3] 0.0; 
              4e-2*u[1] 0.0 0.0; 
              0.0 3e7*u[2]^2 0.0]
end
u0 = @SVector [1.0, 0.0, 0.0]
tspan = (0.0, 1.0e11)
prob = ConservativePDSProblem(prod, u0, tspan)

sol = solve(prob, MPRK43I(1.0, 0.5))
nothing #hide
```
```@example robertson
using Plots

plot(sol, tspan = (1e-6, 1e11),  xaxis = :log,
     idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)],
     label = ["u_1" "1e4u_2" "u_3"])
```
### Using callbacks to solve the Robertson problem with non-adatpive schemes

The `SSPMPRK43()` scheme is only available with fixed time stepping. With a scheme like this it would take an enoumours amount of time to solve the Robertson problem accurately, since the time step must be chosen very small to accurately solve the problem in the initial phase. But the use of a `callback` allows us to modify the time steps size after each step.

In the following example the `callback` doubles the time step size after each time step.
```@example roberston
sol_cb = solve(prob, SSPMPRK43(); dt = Inf, 
               callback = DiscreteCallback(Returns(true), 
                          integrator -> set_proposed_dt!(integrator, 2 * get_proposed_dt(integrator));
               save_positions = (false, false),
               initialize = (c, u, t, integrator) -> set_proposed_dt!(integrator, 1.0e-5)));
```
```@example robertson
plot(sol_cb, tspan = (1e-6, 1e11),  xaxis = :log,
     idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)],
     label = ["u₁" "10⁴u₂" "u₃"])
```

## Package versions

These results were obtained using the following versions.
```@example robertson
using InteractiveUtils
versioninfo()
println()

using Pkg
Pkg.status(["PositiveIntegrators", "StaticArrays", "LinearSolve", "OrdinaryDiffEq"],
           mode=PKGMODE_MANIFEST)
nothing # hide
```
