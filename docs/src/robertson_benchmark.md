# [Benchmark: Solution of the Robertson problem](@id benchmark-robertson)

We use the stiff Robertson model [`prob_pds_npzd`](@ref) to assess the efficiency of different solvers.


```@example ROBER
using OrdinaryDiffEq, PositiveIntegrators
using Plots

# select problem
prob = prob_pds_robertson

# compute reference solution 
ref_sol = solve(prob, Rodas4P(); abstol = 1e-14, reltol = 1e-13)

# compute solutions with low tolerances
abstol = 1e-2
reltol = 1e-1
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol)
sol_MPRK = solve(prob, MPRK22(1.0); abstol, reltol)

# plot solutions
p1 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right)
plot!(p1, sol_Ros23; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "Rosenbrock23", xticks =10.0 .^ (-6:4:10))
p2 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right)
plot!(p2, sol_MPRK; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "MPRK22(1.0)", xticks =10.0 .^ (-6:4:10))
plot(p1, p2)
```

Nevertheless, [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used to guarantee nonnegative solutions.

```@example ROBER
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative) #reject negative solutions

plot(ref_sol, tspan = (1e-7, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right)
plot!(sol_Ros23; tspan = (1e-7, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "Rosenbrock23", xticks =10.0 .^ (-6:4:10))
```

## Work-Precision diagrams

First we compare different (adaptive) MPRK schemes described in the literature. The chosen `l∞` error computes the maximum of the absolute values of the difference between the numerical solution and the reference solution over all components and all time steps.

```@example ROBER
using DiffEqDevTools #load WorkPrecisionSet

# choose methods to compare
setups = [#Dict(:alg => MPRK22(0.5))  FAIL!
          Dict(:alg => MPRK22(2.0 / 3.0))
          Dict(:alg => MPRK22(1.0))
          #Dict(:alg => SSPMPRK22(0.5, 1.0))  FAIL!
          Dict(:alg => MPRK43I(1.0, 0.5))
          Dict(:alg => MPRK43I(0.5, 0.75))
          Dict(:alg => MPRK43II(0.5))
          Dict(:alg => MPRK43II(2.0 / 3.0))]

labels = [#"MPRK22(0.5)"
          "MPPRK22(2/3)"
          "MPRK22(1.0)"
          #"SSPMPRK22(0.5,1.0)"
          "MPRK43I(1.0,0.5)"
          "MPRK43I(0.5,0.75)"
          "MPRK43II(0.5)"
          "MPRK43II(2.0/3.0)"]

# set tolerances and error
abstols = 1.0 ./ 10.0 .^ (2:0.5:8)
reltols = 1.0 ./ 10.0 .^ (1:0.5:7)
err_est = :l∞

# create reference solution for `WorkPrecisionSet`
test_sol = TestSolution(ref_sol)

# compute work-precision
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      error_estimate = err_est, appxsol = test_sol,
                      names = labels, print_names = true,
                      verbose = false)

#plot
plot(wp, title = "Robertson benchmark", legend = :topright,
     color = permutedims([repeat([1],2)...,repeat([3],2)...,repeat([4],2)...]),
     ylims = (10 ^ -5, 10 ^ -2), yticks = 10.0 .^ (-5:.5:-1), minorticks=10,
     xlims = (2 *10 ^ -6, 2*10 ^ -2), xticks =10.0 .^ (-5:1:0))
```

All methods behave similarly. For comparisons with other schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) we choose the second order scheme `MPRK22(1.0)` and the third order scheme `MPRK43I(1.0, 0.5)`.

```@example ROBER
sol_MPRK22 = solve(prob, MPRK22(1.0); abstol, reltol)
sol_MPRK43 = solve(prob, MPRK43I(1.0, 0.5); abstol, reltol)

p1 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right)
plot!(p1, sol_MPRK22; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "MPRK22(1.0)", xticks =10.0 .^ (-6:4:10))
p2 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right)
plot!(p2, sol_MPRK43; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "MPRK43I(1.0, 0.5)", xticks =10.0 .^ (-6:4:10))
plot(p1, p2)
```

# <span style="color:red">**We need relative errors!**</span>.

Next we compare `MPRK22(1.0)` and `MPRK43I(1.0, 0.5)` with some second and third order methods from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). To guarantee positive solutions with these methods, we must select the solver option `isoutofdomain = isnegative`.

```@example ROBER
# select methods
setups = [Dict(:alg => MPRK22(1.0)),
    Dict(:alg => MPRK43I(1.0, 0.5)),
    Dict(:alg => TRBDF2(), :isoutofdomain => isnegative),
    Dict(:alg => SDIRK2(), :isoutofdomain => isnegative),
    Dict(:alg => Kvaerno3(), :isoutofdomain => isnegative),
    Dict(:alg => KenCarp3(), :isoutofdomain => isnegative),
    Dict(:alg => Rodas3(), :isoutofdomain => isnegative),
    Dict(:alg => ROS2(), :isoutofdomain => isnegative),
    Dict(:alg => ROS3(), :isoutofdomain => isnegative),
    Dict(:alg => Rosenbrock23(), :isoutofdomain => isnegative)]

labels = ["MPRK22(1.0)"
          "MPRK43I(1.0,0.5)"
          "TRBDF2"
          "SDIRK2"
          "Kvearno3"
          "KenCarp3"
          "Rodas3"
          "ROS2"
          "ROS3"
          "Rosenbrock23"]

# compute work-precision
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      error_estimate = err_est, appxsol = test_sol,
                      names = labels, print_names = true,                     
                      verbose = false)
plot(wp, title = "Robertson benchmark", legend = :topright,
     color = permutedims([2, 3, repeat([5], 4)..., repeat([6], 4)...]),
     ylims = (10 ^ -5, 10 ^ -2), yticks = 10.0 .^ (-5:.5:-1), minorticks=10,
     xlims = (2 *10 ^ -6, 2*10 ^ -2), xticks =10.0 .^ (-5:1:0))
```

Comparison to recommend solvers.
```@example ROBER
setups = [Dict(:alg => MPRK22(1.0)),
    Dict(:alg => MPRK43I(1.0, 0.5)),
    Dict(:alg => TRBDF2(), :isoutofdomain => isnegative),
    Dict(:alg => Rosenbrock32(), :isoutofdomain => isnegative),
    Dict(:alg => Rodas5P(), :isoutofdomain => isnegative),
    Dict(:alg => Rodas4P(), :isoutofdomain => isnegative)]

labels = ["MPRK22(1.0)"
          "MPRK43I(1.0,0.5)"
          "TRBDF2"
          "Rosenbrock23"
          "Rodas5P"
          "Rodas4P"]

# compute work-precision
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      error_estimate = err_est, appxsol = test_sol,
                      names = labels, print_names = true,
                      verbose = false)

#plot                      
plot(wp, title = "Robertson benchmark", legend = :topright,
     color = permutedims([2, 3, 5, repeat([6], 3)...]),
     ylims = (10 ^ -5, 10 ^ -2), yticks = 10.0 .^ (-5:.5:-1), minorticks=10,
     xlims = (2 *10 ^ -6, 2*10 ^ -2), xticks =10.0 .^ (-5:1:0))
```

## Literature
- Kopecz, Meister 2nd order
- Kopecz, Meister 3rd order
- Huang, Shu 2nd order


## Package versions

These results were obtained using the following versions.
```@example ROBER
using InteractiveUtils
versioninfo()
println()

using Pkg
Pkg.status(["PositiveIntegrators", "StaticArrays", "LinearSolve", "OrdinaryDiffEq"],
           mode=PKGMODE_MANIFEST)
nothing # hide
```
