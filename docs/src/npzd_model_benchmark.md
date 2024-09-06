# [Benchmark: Solution of an NPZD model](@id benchmark-npzd)

We use the NPZD model [`prob_pds_npzd`](@ref) to assess the efficiency of different solvers.

Standard procedures have difficulties to solve this problem accurately, at least for low tolerances.

```@example NPZD
using OrdinaryDiffEq, PositiveIntegrators

prob = prob_pds_npzd
abstol = 1e-2
reltol = 1e-1

sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol)
sol_MPRK = solve(prob, SSPMPRK22(0.5, 1.0); abstol, reltol)

plot(
    plot(sol_Ros23; denseplot = false, markers = :circle),
    plot(sol_MPRK; denseplot = false, markers = :circle)
)
```

Nevertheless, [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used to guarantee nonnegative solutions.

The solution of the NPZD model can now be computed as follows.
```@example NPZD

sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative #reject negative solutions
                  )

plot(sol_Ros23; denseplot = false, markers = :circle)
```

## Work-Precision diagrams

```@example NPZD
using DiffEqDevTools #load Work

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
