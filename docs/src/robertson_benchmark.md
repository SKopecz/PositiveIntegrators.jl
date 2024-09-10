# [Benchmark: Solution of the Robertson problem](@id benchmark-robertson)

We use the stiff Robertson model [`prob_pds_robertson`](@ref) to assess the efficiency of different solvers.


```@example ROBER
using OrdinaryDiffEq, PositiveIntegrators
using Plots

# select problem
prob = prob_pds_robertson

# compute reference solution 
ref_sol = solve(prob, Rodas4P(); abstol = 1e-14, reltol = 1e-13);

# compute solutions with low tolerances
abstol = 1e-2
reltol = 1e-1
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol);
sol_MPRK = solve(prob, MPRK22(1.0); abstol, reltol);

# plot solutions
p1 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right);
plot!(p1, sol_Ros23; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "Rosenbrock23", xticks =10.0 .^ (-6:4:10));
p2 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right);
plot!(p2, sol_MPRK; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "MPRK22(1.0)", xticks =10.0 .^ (-6:4:10));
plot(p1, p2)
```

Nevertheless, [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used to guarantee nonnegative solutions.

```@example ROBER
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative) #reject negative solutions

plot(ref_sol, tspan = (1e-7, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right);
plot!(sol_Ros23; tspan = (1e-7, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "Rosenbrock23", xticks =10.0 .^ (-6:4:10))
```

## Work-Precision diagrams

```@example ROBER
# compute reference solution
tspan = prob.tspan
sol_ref = solve(prob, Rodas4P(); abstol = 1e-15, reltol = 1e-14, save_everystep = false);
sol_ref = sol_ref.u[end]

# define error functions
l2_error(sol, sol_ref) = sqrt(sum(((sol .- sol_ref) ./ sol_ref) .^ 2) / length(sol_ref))
l∞_error(sol, sol_ref) = maximum(abs.((sol .- sol_ref) ./ sol_ref))
nothing #hide output
```
### Adaptive time stepping

```@example ROBER
abstols = 1.0 ./ 10.0 .^ (2:1:11)
reltols = 1.0 ./ 10.0 .^ (1:1:10)
nothing # hide output
```

#### L∞ errors

First we compare different (adaptive) MPRK schemes described in the literature. 

```@example ROBER
algs = [#MPRK22(0.5)
        MPRK22(2.0 / 3.0)
        MPRK22(1.0)
        #SSPMPRK22(0.5, 1.0)
        MPRK43I(1.0, 0.5)
        MPRK43I(0.5, 0.75)
        MPRK43II(0.5)
        MPRK43II(2.0 / 3.0)]

names = [#"MPRK22(0.5)"
         "MPPRK22(2/3)"
         "MPRK22(1.0)"
         #"SSPMPRK22(0.5,1.0)"
         "MPRK43I(1.0,0.5)"
         "MPRK43I(0.5,0.75)"
         "MPRK43II(0.5)"
         "MPRK43II(2.0/3.0)"]

# compute work-precision
wp_l∞ = workprecision_adaptive(prob, algs, names, sol_ref, abstols, reltols;
                               compute_error = l∞_error)

plot(wp_l∞, names; title = "Robertson benchmark (l∞)", legend = :bottomleft,     
     color = permutedims([repeat([1], 2)..., repeat([3], 2)..., repeat([4], 2)...]),
     xlims = (10^-5, 10^0), xticks = 10.0 .^ (-8:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10
     )
```

Besides `SSPMPRK22` and `MPRK22(0.5)` all methods behave similarly. `SSPMPRK22` generates oscillatory solutions.

```@example ROBER
sol1 = solve(prob, SSPMPRK22(0.5, 1.0), abstol=1e-5, reltol = 1e-4);

# plot solutions
p1 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right);
plot!(p1, sol1; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "SSPMPRK22(1.0, 0.5)", xticks =10.0 .^ (-6:4:10))
```

With `abstol=1e-6` and `reltol = 1e-5` the `MPRK22(0.5)` schemes needs over 800.000 steps to integrate the Robertson problem. In comparison, `MPRK22(0.5)` needs less than 1.000.

```@example ROBER
sol1 = solve(prob, MPRK22(1.0), abstol=1e-6, reltol = 1e-5);
sol2 = solve(prob, MPRK22(0.5), abstol=1e-6, reltol = 1e-5);

length(sol1), length(sol2)
```

For comparisons with other schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) we choose the second order scheme `MPRK22(1.0)` and the third order scheme `MPRK43I(0.5, 0.75)`.

```@example ROBER
sol_MPRK22 = solve(prob, MPRK22(1.0); abstol, reltol)
sol_MPRK43 = solve(prob, MPRK43I(0.5, 0.75); abstol, reltol)

p1 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right);
plot!(p1, sol_MPRK22; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "MPRK22(1.0)", xticks =10.0 .^ (-6:4:10));
p2 = plot(ref_sol, tspan = (1e-6, 1e11),  xaxis = :log, idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], linestyle = :dash, label = "", legend = :right);
plot!(p2, sol_MPRK43; tspan = (1e-6, 1e11),  xaxis = :log, denseplot = false, markers = :circle, ylims = (-0.2, 1.2), idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], title = "MPRK43I(0.5, 0.75)", xticks =10.0 .^ (-6:4:10));
plot(p1, p2)
```

Next we compare `MPRK22(1.0)` and `MPRK43I(0.5, 0.75)` with some second and third order methods from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). To guarantee positive solutions with these methods, we must select the solver option `isoutofdomain = isnegative`.

```@example ROBER
# select methods
algs1 = [MPRK22(1.0),
         MPRK43I(0.5, 0.75)]

algs2 = [TRBDF2()
         Kvaerno3()
         KenCarp3()
         Rodas3()
         ROS2()
         ROS3()
         Rosenbrock23()]

names1 = ["MPRK22(1.0)"
          "MPRK43I(0.5,0.75)"]

names2 = ["TRBDF2"
          "Kvearno3"
          "KenCarp3"
          "Rodas3"
          "ROS2"
          "ROS3"
          "Rosenbrock23"]

# compute work-precision
compute_error = l∞_error
wp_l∞ = workprecision_adaptive(prob, algs1, names1, sol_ref, abstols, reltols;
                               compute_error)
workprecision_adaptive!(wp_l∞, prob, algs2, names2, sol_ref, abstols, reltols;
                               compute_error, isoutofdomain=isnegative)

plot(wp_l∞, [names1; names2]; title = "Robertson benchmark (l∞)", legend = :bottomleft,
     color = permutedims([2, 3, repeat([5], 3)..., repeat([6], 4)...]),
     xlims = (10^-14, 10^2), xticks = 10.0 .^ (-14:2:2),
     ylims = (10^-5, 1.5*10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

Comparison to recommend solvers.
```@example ROBER
algs3 = [TRBDF2()
         Rosenbrock23()
         Rodas5P()
         Rodas4P()]

names3 = ["TRBDF2"
          "Rosenbrock23"
          "Rodas5P"
          "Rodas4P"]

# compute work-precision
compute_error = l∞_error
wp_l∞ = workprecision_adaptive(prob, algs1, names1, sol_ref, abstols, reltols;
                               compute_error)                             
workprecision_adaptive!(wp_l∞, prob, algs3, names3, sol_ref, abstols, reltols;
                               compute_error, isoutofdomain=isnegative)

plot(wp_l∞, [names1; names3]; title = "NPZD benchmark (l∞)", legend = :topright,
     color = permutedims([2, 3, 5, repeat([6], 3)...]),
     xlims = (10^-8, 2*10^0), xticks = 10.0 .^ (-11:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)

```

#### L2 errors

```@example ROBER
# compute work-precision
wp_l2 = workprecision_adaptive(prob, algs, names, sol_ref, abstols, reltols;
                               compute_error = l2_error)

plot(wp_l2, names; title = "Robertson benchmark (l2)", legend = :bottomleft,     
     color = permutedims([repeat([1], 2)..., repeat([3], 2)..., repeat([4], 2)...]),
     xlims = (10^-5, 10^0), xticks = 10.0 .^ (-8:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10
     )
```

```@example ROBER
# compute work-precision
compute_error = l2_error
wp_l2 = workprecision_adaptive(prob, algs1, names1, sol_ref, abstols, reltols;
                               compute_error)
workprecision_adaptive!(wp_l2, prob, algs2, names2, sol_ref, abstols, reltols;
                               compute_error, isoutofdomain=isnegative)

plot(wp_l2, [names1; names2]; title = "Robertson benchmark (l2)", legend = :bottomleft,
     color = permutedims([2, 3, repeat([5], 3)..., repeat([6], 4)...]),
     xlims = (10^-14, 10^2), xticks = 10.0 .^ (-14:2:2),
     ylims = (10^-5, 1.5*10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

```@example ROBER

# compute work-precision
compute_error = l2_error
wp_l2 = workprecision_adaptive(prob, algs1, names1, sol_ref, abstols, reltols;
                               compute_error)                             
workprecision_adaptive!(wp_l2, prob, algs3, names3, sol_ref, abstols, reltols;
                               compute_error, isoutofdomain=isnegative)

plot(wp_l2, [names1; names3]; title = "NPZD benchmark (l2)", legend = :topright,
     color = permutedims([2, 3, 5, repeat([6], 3)...]),
     xlims = (10^-8, 2*10^0), xticks = 10.0 .^ (-11:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)

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
