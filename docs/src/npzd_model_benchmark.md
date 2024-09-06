# [Benchmark: Solution of an NPZD model](@id benchmark-npzd)

We use the NPZD model [`prob_pds_npzd`](@ref) to assess the efficiency of different solvers.

Standard procedures have difficulties to solve this problem accurately, at least for low tolerances.

```@example NPZD
using OrdinaryDiffEq, PositiveIntegrators
using Plots

# select problem
prob = prob_pds_npzd

# compute reference solution (standard tolerances are too low)
ref_sol = solve(prob, Vern7(); abstol = 1e-14, reltol = 1e-13)

# compute solutions with low tolerances
abstol = 1e-2
reltol = 1e-1
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol)
sol_MPRK = solve(prob, MPRK22(1.0); abstol, reltol)

# plot solutions
p1 = plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(p1, sol_Ros23; denseplot = false, markers = :circle, ylims = (-1.0, 10.0),
      title = "Rosenbrock23")
p2 = plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(p2, sol_MPRK; denseplot = false, markers = true, ylims = (-1.0, 10.0),
     title = "MPRK22(1.0)")
plot(p1, p2)
```

Nevertheless, [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used to guarantee nonnegative solutions.

```@example NPZD
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative) #reject negative solutions

plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(sol_Ros23; denseplot = false, markers = :circle, ylims = (-1.0, 10.0),
          title = "Rosenbrock23")
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

# set tolerances and error
abstols = 1.0 ./ 10.0 .^ (2:8)
reltols = 1.0 ./ 10.0 .^ (1:7)
err_est = :l∞

# compute work-precision
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      error_estimate = err_est, appxsol = ref_sol,
                      names = labels, print_names = true,
                      verbose = false)

#plot
plot(wp, title = "NPZD benchmark", legend = :topright,
     color = permutedims([repeat([1],3)...,2,repeat([3],2)...,repeat([4],2)...]))
```

The second- and third-order methods behave very similarly. For comparisons with other schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) we choose the schemes with the smallest error for the initial tolerances, respectively. These are `SSPMPRK22(0.5, 1.0)` and `MPRK43I(1.0, 0.5)`.

```@example NPZD
sol_SSPMPRK22 = solve(prob, SSPMPRK22(0.5, 1.0); abstol, reltol)
sol_MPRK43 = solve(prob, MPRK43I(1.0, 0.5); abstol, reltol)

p1 = plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(p1, sol_SSPMPRK22; denseplot = false, markers = :circle, ylims = (-1.0, 10.0),
      title = "SSPMPRK22(0.5, 1.0)")
p2 = plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(p2, sol_MPRK43; denseplot = false, markers = true, ylims = (-1.0, 10.0),
     title = "MPRK43I(1.0, 0.5)")
plot(p1, p2)
```

```@example NPZD
# select methods
setups = [Dict(:alg => SSPMPRK22(0.5, 1.0)),
    Dict(:alg => MPRK43I(1.0, 0.5)),
    Dict(:alg => Midpoint(), :isoutofdomain => isnegative),
    Dict(:alg => Heun(), :isoutofdomain => isnegative),
    Dict(:alg => Ralston(), :isoutofdomain => isnegative),
    Dict(:alg => TRBDF2(), :isoutofdomain => isnegative),
    Dict(:alg => SDIRK2(), :isoutofdomain => isnegative),
    Dict(:alg => Kvaerno3(), :isoutofdomain => isnegative),
    Dict(:alg => KenCarp3(), :isoutofdomain => isnegative),
    Dict(:alg => Rodas3(), :isoutofdomain => isnegative),
    Dict(:alg => ROS2(), :isoutofdomain => isnegative),
    Dict(:alg => ROS3(), :isoutofdomain => isnegative)]

labels = ["SSPMPRK22(0.5,1.0)"
          "MPRK43I(1.0,0.5)"
          "Midpoint"
          "Heun"
          "Ralston"
          "TRBDF2"
          "SDIRK2"
          "Kvearno3"
          "KenCarp3"
          "Rodas3"
          "ROS2"
          "ROS3"]

# compute work-precision
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      error_estimate = err_est, appxsol = ref_sol,
                      names = labels, print_names = true,                      
                      verbose = false)
plot(wp, title = "NPZD benchmark", legend = :topright,
     color = permutedims([2, 3, repeat([4], 3)..., repeat([5], 4)..., repeat([6], 3)...]))

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
