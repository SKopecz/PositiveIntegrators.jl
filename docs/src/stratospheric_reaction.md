# [Tutorial: Solution of a stratospheric reaction problem](@id tutorial-stratos)

This tutorial is about the efficient solution of a stiff non-autonomous and non-conservative production-destruction systems (PDS) with a small number of differential equations. 
We will compare the use of standard arrays and static arrays from [StaticArrays.jl](https://juliaarrays.github.io/StaticArrays.jl/stable/) and assess their efficiency.

## Definition of the production-destruction system

This stratospheric reaction problem was described by Adrian Sandu in [Positive Numerical Integration Methods for Chemical Kinetic Systems](https://doi.org/10.1006/jcph.2001.6750), see also the paper [Positivity-preserving adaptive Runge–Kutta methods](https://doi.org/10.2140/camcos.2021.16.155) by Stefan Nüßlein, Hendrik Ranocha and David I. Ketcheson. The goverining equations are
```math
\begin{aligned}
O^{1D}' &= r_5 - r_6 -  r_7,\\
O' &= 2r_1 - r_2 + r_3 - r_4 + r_6 - r_9 + r_{10} - r_{11},\\
O_3' &= r_2 - r_3 - r_4 - r_5 - r_7 - r_8,\\
O_2' &= -r_1 -r_2 + r_3 + 2r_4+r_5+2r_7+r_8+r_9,\\
NO' &= -r_8+r_9+r_{10}-r_{11},\\
NO_2' &= r_8-r_9-r_{10}+r_{11},
\end{aligned}
```
with reaction rates
```math
\begin{aligned}
r_1 &=2.643⋅ 10^{-10}σ^3 O_2, & r_2 &=8.018⋅10^{-17}O O_2 , & r_3 &=6.12⋅10^{-4}σ O_3,\\
r_4 &=1.567⋅10^{-15}O_3 O , & r_5 &= 1.07⋅ 10^{-3}σ^2O_3,  & r_6 &= 7.11⋅10^{-11}⋅ 8.12⋅10^6 O^{1D},\\
r_7 &= 1.2⋅10^{-10}O^{1D} O_3, & r_8 &= 6.062⋅10^{-15}O_3 NO, & r_9 &= 1.069⋅10^{-11}NO_2 O,\\
r_{10} &= 1.289⋅10^{-2}σ NO_2, & r_{11} &= 10^{-8}NO O,
\end{aligned}
```
where
```math
\begin{aligned}
T &= t/3600 \mod 24,\quad T_r=4.5,\quad T_s = 19.5,\\
σ(T) &= \begin{cases}1, & T_r≤ T≤ T_s,\\0, & \text{otherwise}.\end{cases}
\end{aligned}
```
Setting ``\mathbf u = (O^{1D}, O, O_3, O_2, NO, NO_2)`` the initial value is ``\mathbf{u}_0 = (9.906⋅10^1, 6.624⋅10^8, 5.326⋅10^{11}, 1.697⋅10^{16}, 4⋅10^6, 1.093⋅10^9)^T`` and the time domain is ``(4.32⋅ 10^{4}, 3.024⋅10^5)``.
There are two independent linear invariants, e.g. ``u_1+u_2+3u_3+2u_4+u_5+2u_6=(1,1,3,2,1,2)\\cdot\\mathbf{u}_0`` and ``u_5+u_6 = 1.097⋅10^9``.

The stratospheric reaction problem can be represented as a (non-conservative) PDS with production terms
```math
\begin{aligned}
p_{13} &= r_5, & p_{21} &= r_6, & p_{22} &= r_1+r_{10},\\
p_{23} &= r_3, & p_{24} &= r_1,& p_{32} &= r_2,\\
p_{41} &= r_7, & p_{42}&= r_4+r_9, & p_{43}&= r_4+r_7+r_8,\\
p_{44} &= r_3+r_5, & p_{56}=r_9+r_{10}, & p_{65}&=r_8+r_{11}.
\end{aligned}
```
and additional destruction terms
```math
\begin{aligned}
d_{22}&= r11, & d_{44}&=r2.
```

## Solution of the production-destruction system

Now we are ready to define a `PDSProblem` and to solve this problem with a method of [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl) or [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). 

As mentioned above, we will try two approaches to solve this PDS and compare their efficiency. These are
1. an out-of-place implementation with standard (dynamic) matrices and vectors,
2. an out-of-place implementation with static matrices and vectors from [StaticArrays.jl](https://juliaarrays.github.io/StaticArrays.jl/stable/).

### Standard out-of-place implementation

Here we create a function to compute the production matrix with return type `Matrix{Float64}`.

```@example stratreac
using PositiveIntegrators # load PDSProblem

function prod(u, p, t)
    O1D, O, O3, O2, NO, NO2 = u

    Tr = 4.5
    Ts = 19.5
    T = mod(t / 3600, 24)
    if (Tr <= T) && (T <= Ts)
        Tfrac = (2 * T - Tr - Ts) / (Ts - Tr)
        sigma = 0.5 + 0.5 * cos(pi * abs(Tfrac) * Tfrac)
    else
        sigma = 0.0
    end

    M = 8.120e16

    k1 = 2.643e-10 * sigma^3
    k2 = 8.018e-17
    k3 = 6.120e-4 * sigma
    k4 = 1.567e-15
    k5 = 1.070e-3 * sigma^2
    k6 = 7.110e-11
    k7 = 1.200e-10
    k8 = 6.062e-15
    k9 = 1.069e-11
    k10 = 1.289e-2 * sigma
    k11 = 1.0e-8

    r1 = k1 * O2
    r2 = k2 * O * O2
    r3 = k3 * O3
    r4 = k4 * O3 * O
    r5 = k5 * O3
    r6 = k6 * M * O1D
    r7 = k7 * O1D * O3
    r8 = k8 * O3 * NO
    r9 = k9 * NO2 * O
    r10 = k10 * NO2
    r11 = k11 * NO * O

    return [0.0 0.0 r5 0.0 0.0 0.0;
            r6 r1+r10 r3 r1 0.0 0.0;
            0.0 r2 0.0 0.0 0.0 0.0;
            r7 r4+r9 r4+r7+r8 r3+r5 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 r9+r10;
            0.0 0.0 0.0 0.0 r8+r11 0.0]
end

function dest(u, p, t)
    O1D, O, O3, O2, NO, NO2 = u

    k2 = 8.018e-17
    k11 = 1.0e-8

    r2 = k2 * O * O2
    r11 = k11 * NO * O

    return [0.0, r11, 0.0, r2, 0.0, 0.0]
end
nothing #hide
```
The solution of the stratospheric reaction problem can now be computed as follows.
```@example stratreac
u0 = [9.906e1, 6.624e8, 5.326e11, 1.697e16, 4e6, 1.093e9] # initial values
tspan = (4.32e4, 3.024e5) # time domain
prob = PDSProblem(prod, u0, tspan) # create the PDS

sol = solve(prob, MPRK43I(1.0, 0.5))

nothing #hide
```
```@example stratreac
using Plots

xticks = [tspan[1], tspan[2]]
legend = :outertop
p1 = plot(sol; idxs = (0, 1), label = "O¹ᴰ", xticks, legend, ylims=(-10, 100))
p2 = plot(sol; idxs = (0, 2), label = "O", xticks, legend, ylims = (-1e8, 8e8))
p3 = plot(sol; idxs = (0, 3), label = "O₃", xticks, legend, ylims = (2e11, 6e11))
p4 = plot(sol; idxs = (0, 4), label = "O₂", xticks, legend, ylims=(1.69698e16, 1.69705e16))
p5 = plot(sol; idxs = (0, 5), label = "NO", xticks, legend, ylims = (-5e6, 15e6))
p6 = plot(sol; idxs = (0, 6), label = "NO₂", xticks, legend, ylims = (1.08e9, 1.1e9))
plot(p1, p2, p3, p4, p5, p6)
```

### Using static arrays
For PDS with a small number of differential equations like the NPZD model the use of static arrays will be more efficient. To create a function which computes the production matrix and returns a static matrix, we only need to add the `@SMatrix` macro.

```@example stratreac
using StaticArrays

function prod_static(u, p, t)
    O1D, O, O3, O2, NO, NO2 = u

    Tr = 4.5
    Ts = 19.5
    T = mod(t / 3600, 24)
    if (Tr <= T) && (T <= Ts)
        Tfrac = (2 * T - Tr - Ts) / (Ts - Tr)
        sigma = 0.5 + 0.5 * cos(pi * abs(Tfrac) * Tfrac)
    else
        sigma = 0.0
    end

    M = 8.120e16

    k1 = 2.643e-10 * sigma^3
    k2 = 8.018e-17
    k3 = 6.120e-4 * sigma
    k4 = 1.567e-15
    k5 = 1.070e-3 * sigma^2
    k6 = 7.110e-11
    k7 = 1.200e-10
    k8 = 6.062e-15
    k9 = 1.069e-11
    k10 = 1.289e-2 * sigma
    k11 = 1.0e-8

    r1 = k1 * O2
    r2 = k2 * O * O2
    r3 = k3 * O3
    r4 = k4 * O3 * O
    r5 = k5 * O3
    r6 = k6 * M * O1D
    r7 = k7 * O1D * O3
    r8 = k8 * O3 * NO
    r9 = k9 * NO2 * O
    r10 = k10 * NO2
    r11 = k11 * NO * O

    return @SMatrix [0.0 0.0 r5 0.0 0.0 0.0;
            r6 r1+r10 r3 r1 0.0 0.0;
            0.0 r2 0.0 0.0 0.0 0.0;
            r7 r4+r9 r4+r7+r8 r3+r5 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 r9+r10;
            0.0 0.0 0.0 0.0 r8+r11 0.0]
end

function dest_static(u, p, t)
    O1D, O, O3, O2, NO, NO2 = u

    k2 = 8.018e-17
    k11 = 1.0e-8

    r2 = k2 * O * O2
    r11 = k11 * NO * O

    return @SVector [0.0, r11, 0.0, r2, 0.0, 0.0]
end
nothing #hide
```
In addition we also want to use a static vector to hold the initial conditions.
```@example stratreac
u0_static = @SVector [9.906e1, 6.624e8, 5.326e11, 1.697e16, 4e6, 1.093e9] # initial values
prob_static = PDSProblem(prod_static, dest_static, u0_static, tspan) # create the PDS

sol_static = solve(prob_static, MPRK43I(1.0, 0.5))

nothing #hide
```
```@example stratreac
using Plots

xticks = [tspan[1], tspan[2]]
legend = :outertop
p1 = plot(sol; idxs = (0, 1), label = "O¹ᴰ", xticks, legend, ylims=(-10, 100))
p2 = plot(sol; idxs = (0, 2), label = "O", xticks, legend, ylims = (-1e8, 8e8))
p3 = plot(sol; idxs = (0, 3), label = "O₃", xticks, legend, ylims = (2e11, 6e11))
p4 = plot(sol; idxs = (0, 4), label = "O₂", xticks, legend, ylims=(1.69698e16, 1.69705e16))
p5 = plot(sol; idxs = (0, 5), label = "NO", xticks, legend, ylims = (-5e6, 15e6))
p6 = plot(sol; idxs = (0, 6), label = "NO₂", xticks, legend, ylims = (1.08e9, 1.1e9))
plot(p1, p2, p3, p4, p5, p6)
```
### Performance comparison

Finally, we use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
to show the benefit of using static arrays.

```@example stratreac
using BenchmarkTools
@benchmark solve(prob, MPRK43I(1.0, 0.5))
```

```@example stratreac
@benchmark solve(prob_static, MPRK43I(1.0, 0.5))
```

## Package versions

These results were obtained using the following versions.
```@example stratreac
using InteractiveUtils
versioninfo()
println()

using Pkg
Pkg.status(["PositiveIntegrators", "StaticArrays", "LinearSolve", "OrdinaryDiffEq"],
           mode=PKGMODE_MANIFEST)
nothing # hide
```
