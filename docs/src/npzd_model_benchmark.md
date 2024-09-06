# [Benchmark: Solution of an NPZD model](@id benchmark-npzd)

We use the NPZD model [`prob_pds_npzd`](@ref) to assess the efficiency of different solvers.

Standard procedures have difficulties to solve this problem accurately, at least for low tolerances.

```@example NPZD
using OrdinaryDiffEq, PositiveIntegrators
using Plots

prob = prob_pds_npzd
abstol = 1e-2
reltol = 1e-1

sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol)
sol_MPRK = solve(prob, SSPMPRK22(0.5, 1.0); abstol, reltol)

plot(plot(sol_Ros23; denseplot = false, markers = :circle),
     plot(sol_MPRK; denseplot = false, markers = :circle))
```

Nevertheless, [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used to guarantee nonnegative solutions.

The solution of the NPZD model can now be computed as follows.
```@example NPZD

sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative) #reject negative solutions

plot(sol_Ros23; denseplot = false, markers = :circle)
```

## Work-Precision diagrams

First we compare different (adaptive) MPRK schemes described in the literature.
```@example NPZD
using DiffEqDevTools #load WorkPrecisionSet

# create reference solution
sol = solve(prob, Vern7(), abstol = 1 / 10^14, reltol = 1 / 10^13)
ref_sol = TestSolution(sol)

# choose methods to compare
setups = [Dict(:alg => MPRK22(0.5))
          Dict(:alg => MPRK22(2.0 / 3.0))
          Dict(:alg => MPRK22(1.0))
          Dict(:alg => SSPMPRK22(0.5, 1.0))
          Dict(:alg => MPRK43I(1.0, 0.5))
          Dict(:alg => MPRK43I(0.5, 0.75))
          Dict(:alg => MPRK43II(0.5))
          Dict(:alg => MPRK43II(2.0 / 3.0))]
labels = ["MPRK22(0.5)"
          "MPPRK22(2/3)"
          "MPRK22(1.0)"
          "SSPMPRK22(0.5,1.0)"
          "MPRK43I(1.0,0.5)"
          "MPRK43I(0.5,0.75)"
          "MPRK43II(0.5)"
          "MPRK43II(2.0/3.0)"]


abstols = 1.0 ./ 10.0 .^ (2:8)
reltols = 1.0 ./ 10.0 .^ (1:7)
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      error_estimate = :l∞, appxsol = ref_sol,
                      names = labels, print_names = true,
                      verbose = false)

plot(wp, title = "NPZD benchmark", legend = :topright,
     color = permutedims([repeat([1],3)...,2,repeat([3],2)...,repeat([4],2)...]))
```

## Literature
- Kopecz, Meister 2nd order
- Kopecz, Meister 3rd order
- Huang, Shu 2nd order

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
