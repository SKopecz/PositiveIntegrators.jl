# [Benchmark: Solution of a stratospheric reaction problem](@id benchmark-stratos)

We use the stiff stratospheric reacation problem [`prob_pds_stratreac`](@ref) to assess the efficiency of different solvers.


```@example stratreac
using OrdinaryDiffEq, PositiveIntegrators
using Plots
include("docs/src/utilities.jl")

# select problem
prob = prob_pds_stratreac

# compute reference solution 
tspan = prob.tspan
dt_ref = (last(tspan) - first(tspan)) ./ 1e5
sol_ref = solve(prob, Rodas4P(); dt = dt_ref, adaptive = false, save_everystep = false);
sol_ref = sol_ref.u[end]


dt0 = 48 * 60 #48 minutes
dts = dt0 ./ 2 .^ (0:3)

algs = [MPE()
        MPRK22(1.0)]

names = ["MPE"
         "MPRK22(1.0)"]

wp = workprecision_fixed(prob, algs, names, sol_ref, dts)


plot(wp, names,
     color = permutedims([repeat([1], 2)...]), legend = :top)

```

```@example
ref_sol = solve(prob, Rodas4P(); abstol = 1e-11, reltol = 1e-10);

# compute solutions with low tolerances
abstol = 1e-2
reltol = 1e-1
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol);
sol_MPRK = solve(prob, MPRK22(1.0); abstol, reltol);

# plot solutions
tspan = prob.tspan
plot(ref_sol, layout=(3,2),
    xguide = "t [h]", xguidefontsize = 8,
    xticks = (range(first(tspan), last(tspan), 4), range(12.0, 84.0, 4)), tickfontsize = 7,
    yguide=["O¹ᴰ" "O" "O₃" "O₂" "NO" "NO₂"],    
    linestyle = :dash, label = "",    
    legend = :outertop, legend_column = -1,
    widen = true);
plot!(sol_Ros23, label = "Ros23", denseplot = false, markers = :circle);    
plot!(sol_MPRK, label = "MPRK22(1.0)", denseplot = false, markers = :circle)
```

Although not visible in the plots, the `Rosenbrock23` solution contains negative values.

```@example stratreac
isnonnegative(sol_Ros23)
```

[OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used to guarantee nonnegative solutions.

To improve the MPRK22 result we inrecase the method's `small_constant`.

```@example stratreac
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative); #reject negative solutions
sol_MPRK = solve(prob, MPRK22(1.0, small_constant = 1e-6); abstol, reltol);

plot(ref_sol, layout=(3,2),
    xguide = "t [h]", xguidefontsize = 8,
    xticks = (range(first(tspan), last(tspan), 4), range(12.0, 84.0, 4)), tickfontsize = 7,
    yguide=["O¹ᴰ" "O" "O₃" "O₂" "NO" "NO₂"],    
    linestyle = :dash, label = "",    
    legend = :outertop, legend_column = -1,
    widen = true);
plot!(sol_Ros23, label = "Ros23", denseplot = false, markers = :circle);    
plot!(sol_MPRK, label = "MPRK22(1.0)", denseplot = false, markers = :circle)
```

## Work-Precision diagrams

First we compare different (adaptive) MPRK schemes described in the literature. The chosen `l∞` error computes the maximum of the absolute values of the difference between the numerical solution and the reference solution over all components and all time steps.

```@example stratreac
using DiffEqDevTools #load WorkPrecisionSet

# choose methods to compare
setups = [#Dict(:alg => MPRK22(0.5)) # fail
          #Dict(:alg => MPRK22(0.5, small_constant = 1e-6)) #fail
          #Dict(:alg => MPRK22(2.0 / 3.0)) #fail
          #Dict(:alg => MPRK22(2.0 / 3.0, small_constant = 1e-6)) #fail
          Dict(:alg => MPRK22(1.0))
          Dict(:alg => MPRK22(1.0, small_constant = 1e-6))
          #Dict(:alg => SSPMPRK22(0.5, 1.0)) # takes too long
          Dict(:alg => MPRK43I(1.0, 0.5))
          Dict(:alg => MPRK43I(1.0, 0.5, small_constant = 1e-6))
          Dict(:alg => MPRK43I(0.5, 0.75))
          Dict(:alg => MPRK43I(0.5, 0.75, small_constant = 1e-6))
          Dict(:alg => MPRK43II(0.5))
          Dict(:alg => MPRK43II(0.5, small_constant = 1e-6))
          Dict(:alg => MPRK43II(2.0 / 3.0))
          Dict(:alg => MPRK43II(2.0 / 3.0, small_constant = 1e-6))
          ]

labels = [#"MPRK22(0.5)"
          #"MPRK22(0.5, sc=1e-6)"
          #"MPPRK22(2/3)"
          #"MPPRK22(2/3, sc=1e-6)"
          "MPRK22(1.0)"
          "MPRK22(1.0, sc=1e-6)"
          #"SSPMPRK22(0.5,1.0)"
          "MPRK43I(1.0,0.5)"
          "MPRK43I(1.0,0.5, sc=1e-6)"
          "MPRK43I(0.5,0.75)"
          "MPRK43I(0.5,0.75, sc=1e-6)"
          "MPRK43II(0.5)"
          "MPRK43II(0.5, sc=1e-6)"
          "MPRK43II(2.0/3.0)"
          "MPRK43II(2.0/3.0, sc=1e-6)"
          ]

# set tolerances and error
abstols = 1.0 ./ 10.0 .^ (2:0.5:5)
reltols = 1.0 ./ 10.0 .^ (1:0.5:4)
err_est = :l∞

# create reference solution for `WorkPrecisionSet`
test_sol = TestSolution(ref_sol)

# compute work-precision
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      error_estimate = err_est, appxsol = test_sol,
                      names = labels, print_names = true,
                      verbose = false)

#plot
plot(wp, title = "Stratospheric reaction benchmark", legend = :bottomleft,
     color = permutedims([repeat([1],2)...,repeat([3],4)...,repeat([4],4)...]),
     #ylims = (10 ^ -5, 10 ^ -1), yticks = 10.0 .^ (-5:.5:-1), minorticks=10,
     #xlims = (2 *10 ^ -6, 2*10 ^ -2), xticks =10.0 .^ (-5:1:0),
     )
```


All methods using `small_constant = 1e-6` behave similar, irrespective of the method's order.
For comparisons with other second and third order schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) we choose the third order scheme `MPRK43I(1.0, 0.5)`. To guarantee positive solutions of the [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) methods, we must select the solver option `isoutofdomain = isnegative`.

```@example stratreac
# select methods
setups = [
    Dict(:alg => MPRK43I(1.0, 0.5)),
    Dict(:alg => TRBDF2(), :isoutofdomain => isnegative),
    Dict(:alg => SDIRK2(), :isoutofdomain => isnegative),
    Dict(:alg => Kvaerno3(), :isoutofdomain => isnegative),
    Dict(:alg => KenCarp3(), :isoutofdomain => isnegative),
    Dict(:alg => Rodas3(), :isoutofdomain => isnegative),
    Dict(:alg => ROS2(), :isoutofdomain => isnegative),
    Dict(:alg => ROS3(), :isoutofdomain => isnegative),
    Dict(:alg => Rosenbrock23(), :isoutofdomain => isnegative)]

labels = ["MPRK43I(1.0,0.5)"
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
plot(wp, title = "Stratospheric reaction benchmark", legend = :topright,
     color = permutedims([3, repeat([5], 4)..., repeat([6], 4)...]),
     #ylims = (10 ^ -5, 10 ^ 0), yticks = 10.0 .^ (-5:.5:0), minorticks=10,
     #xlims = (1 *10 ^ -8, 2*10 ^ -2), xticks =10.0 .^ (-7:1:0)
     )
```

Comparison to recommend solvers.
```@example stratreac
setups = [Dict(:alg => MPRK43I(1.0, 0.5)),
    Dict(:alg => TRBDF2(), :isoutofdomain => isnegative),
    Dict(:alg => Rosenbrock23(), :isoutofdomain => isnegative),
    Dict(:alg => Rodas5P(), :isoutofdomain => isnegative),
    Dict(:alg => Rodas4P(), :isoutofdomain => isnegative)]

labels = ["MPRK43I(1.0,0.5)"
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
     color = permutedims([3, 5, repeat([6], 3)...]),
     #ylims = (10 ^ -5, 10 ^ 0), yticks = 10.0 .^ (-5:.5:0), minorticks=10,
     #xlims = (1 *10 ^ -9, 2*10 ^ -2), xticks =10.0 .^ (-8:1:0)
     )
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
