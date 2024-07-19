# [Tutorial: Solution of an NPZD model](@id tutorial-npzd)

This tutorial is about the efficient solution of production-destruction systems (PDS) with a small number of differential equations.
We will compare the use of standard arrays and static arrays from [StaticArrays.jl](https://juliaarrays.github.io/StaticArrays.jl/stable/) and assess their efficiency.

## Definition of the production-destruction system

The NPZD model we want to solve was described by Burchard, Deleersnijder and Meister in [Application of modified Patankar schemes to stiff biogeochemical models for the water column](https://doi.org/10.1007/s10236-005-0001-x). The model reads
```math
\begin{aligned}
N' &= 0.01N + 0.01Z + 0.003D - \\frac{NP}{0.01 + N},\\\\
P' &= \\frac{NP}{0.01 + N}- 0.01P - 0.5( 1 - e^{-1.21P^2})Z - 0.05P,\\\\
Z' &= 0.5(1 - e^{-1.21P^2})Z - 0.01Z - 0.02Z,\\\\
D' &= 0.05P + 0.02Z - 0.003D
\end{aligned}
```
and we consider the initial conditions ``N=8``, ``P=2``, ``Z=1`` and ``D=4``. The time domain of interest is ``t\in[0,10]``. 

The model can be represented as a conservative PDS with production terms
```math
\begin{aligned}
p_{12} &= 0.01 P,\\
p_{13} &= 0.01 Z,\\
p_{14} &= 0.003 D,\\
p_{21} &= \frac{NP}{0.01 + N},\\
p_{32} &= 0.5  (1.0 - e^{-1.21  P^2})  Z,\\
p_{42} &= 0.05  P,\\
p_{43} &= 0.02  Z.
\end{aligned}
```
Since the PDS is conservative, we have ``d_{i,j}=p_{j,i}`` and the system is fully determined by the production matrix ``\mathbf P``.

## Solution of the production-destruction system

Now we are ready to define a `ConservativePDSProblem` and to solve this problem with a method of [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl) or [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). 

As mentioned above, we will try two approaches to solve this PDS and compare their efficiency. These are
1. an out-of-place implementation with standard (dynamic) matrices and vectors,
2. an out-of-place implementation with static matrices and vectors from [StaticArrays.jl](https://juliaarrays.github.io/StaticArrays.jl/stable/).

### Standard out-of-place implementation

Here we create a function to compute the production matrix with return type `Matrix{Float64}`.

```@example NPZD
using PositiveIntegrators # load ConservativePDSProblem

function prod(u, p, t)
    N, P, Z, D = u

    p12 = 0.01 * P
    p13 = 0.01 * Z
    p14 = 0.003 * D
    p21 = N / (0.01 + N) * P
    p32 = 0.5 * (1.0 - exp(-1.21 * P^2)) * Z
    p42 = 0.05 * P
    p43 = 0.02 * Z

    return [0.0 p12 p13 p14;
            p21 0.0 0.0 0.0;
            0.0 p32 0.0 0.0;
            0.0 p42 p43 0.0]
end
nothing #hide
```
The solution of the NPZD model can now be computed as follows.
```@example NPZD
u0 = [8.0, 2.0, 1.0, 4.0] # initial values
tspan = (0.0, 10.0) # time domain
prob = ConservativePDSProblem(prod, u0, tspan) # create the PDS

sol = solve(prob, MPRK43I(1.0, 0.5))

nothing #hide
```
```@example NPZD
using Plots

plot(sol; label = ["N" "P" "Z" "D"], xguide = "t")
```

### Using static arrays
For PDS with a small number of differential equations like the NPZD model the use of static arrays will be more efficient. To create a function which computes the production matrix and returns a static matrix, we only need to add the `@SMatrix` macro.

```@example NPZD
using StaticArrays

function prod_static(u, p, t)
    N, P, Z, D = u

    p12 = 0.01 * P
    p13 = 0.01 * Z
    p14 = 0.003 * D
    p21 = N / (0.01 + N) * P
    p32 = 0.5 * (1.0 - exp(-1.21 * P^2)) * Z
    p42 = 0.05 * P
    p43 = 0.02 * Z

    return @SMatrix [0.0 p12 p13 p14;
                     p21 0.0 0.0 0.0;
                     0.0 p32 0.0 0.0;
                     0.0 p42 p43 0.0]
end
nothing #hide
```
In addition we also want to use a static vector to hold the initial conditions.
```@example NPZD
u0_static = @SVector [8.0, 2.0, 1.0, 4.0] # initial values
prob_static = ConservativePDSProblem(prod_static, u0_static, tspan) # create the PDS

sol_static = solve(prob_static, MPRK43I(1.0, 0.5))

nothing #hide
```
```@example NPZD
using Plots

plot(sol_static; label = ["N" "P" "Z" "D"], xguide = "t")
```
### Performance comparison

Finally, we use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
to show the benefit of using static arrays.

```@example NPZD
using BenchmarkTools
@benchmark solve(prob, MPRK43I(1.0, 0.5))
```

```@example NPZD
@benchmark solve(prob_static, MPRK43I(1.0, 0.5))
```

## Package versions

These results were obtained using the following versions.
```@example NPZD
using InteractiveUtils
versioninfo()
println()

using Pkg
Pkg.status(["PositiveIntegrators", "StaticArrays", "LinearSolve", "OrdinaryDiffEq"],
           mode=PKGMODE_MANIFEST)
nothing # hide
```
