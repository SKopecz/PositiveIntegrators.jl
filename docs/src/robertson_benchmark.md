# [Benchmark: Solution of the Robertson problem](@id benchmark-robertson)

Here we use the stiff Robertson problem [`prob_pds_robertson`](@ref) to assess the efficiency of different solvers from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) and [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl).

```@example ROBER
using OrdinaryDiffEq, PositiveIntegrators

# select Robertson problem
prob = prob_pds_robertson
nothing # hide
```

To keep the following code as clear as possible, we define a helper function `robertson_plot` that we use for plotting.

```@example ROBER
using Plots

robertson_plot = function (sol, sol_ref = nothing, title = "")
    colors = palette(:default)[1:3]'
    if !isnothing(sol_ref)
        p = plot(sol_ref, tspan = (1e-6, 1e11), xaxis = :log,
                 idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)],
                 linestyle = :dash, label = "", color = colors, linewidth = 2)
        plot!(p, sol; tspan = (1e-6, 1e11), xaxis = :log, denseplot = false,
              markers = :circle, ylims = (-0.2, 1.2),
              idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)],
              title, xticks = 10.0 .^ (-6:4:10), color = colors,
              linewidht = 2, legend = :right, label = ["u₁" "u₂" "u₃"])
    else
        p = plot(sol; tspan = (1e-6, 1e11), xaxis = :log, denseplot = false,
                 markers = :circle, ylims = (-0.2, 1.2),
                 idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)],
                 title, xticks = 10.0 .^ (-6:4:10), color = colors,
                 linewidht = 2, legend = :right, label = ["u₁" "u₂" "u₃"])
    end
    return p
end
nothing # hide
```

For this stiff problem the computation of negative approximations may lead to inaccurate solutions. 
This typically occurs when adaptive time stepping uses loose tolerances.

```@example ROBER
# compute reference solution for plotting
ref_sol = solve(prob, Rodas4P(); abstol = 1e-14, reltol = 1e-13);

# compute solutions with loose tolerances
abstol = 1e-2
reltol = 1e-1
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol);
sol_MPRK = solve(prob, MPRK22(1.0); abstol, reltol);

# plot solutions
p1 = robertson_plot(sol_Ros23, ref_sol, "Rosenbrock23");
p2 = robertson_plot(sol_MPRK, ref_sol, "MPRK22(1.0)");
plot(p1, p2)
```

Nevertheless, [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used in combination with [`isnegative`](@ref) to guarantee nonnegative solutions. 

```@example ROBER
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative) #reject negative solutions

robertson_plot(sol_Ros23, ref_sol, "Rosenbrock23")
```

## Work-Precision diagrams

In the following we show several work-precision diagrams, which compare different methods with respect to computing time and the respective error. 
We focus solely on adaptive methods, since the time interval ``(0, 10^{11})`` is too large to generate accurate solutions with fixed step sizes.

Since the Robertson problem is stiff, we need to use a suited implicit scheme to compute a reference solution, see the [solver guide](https://docs.sciml.ai/DiffEqDocs/dev/solvers/ode_solve/#Stiff-Problems). Note that we cannot use the recommended method `radau()`, since [`prob_pds_robertson`](@ref) uses [StaticArrays](https://juliaarrays.github.io/StaticArrays.jl/stable/) instead of arrays of type `Float64`.

```@example ROBER
# select solver to compute reference solution
alg_ref = Rodas4P()
nothing # hide
```

We use the functions [`work_precision_adaptive`](@ref) and [`work_precision_adaptive!`](@ref) to compute the data for the diagrams.
Furthermore, the following absolute and relative tolerances are used.

```@example ROBER
# set absolute and relative tolerances
abstols = 1.0 ./ 10.0 .^ (2:1:10)
reltols = abstols .* 10.0
nothing # hide
```

### Relative maximum error at the final time

In this section the chosen error is the relative maximum error at the final time ``t = 10^{11}``.

```@example ROBER
# select relative maximum error at the end of the problem's time span.
compute_error = rel_max_error_tend
nothing # hide
```

We start with a comparison of different adaptive MPRK schemes.
```@example ROBER
# choose methods to compare
algs = [MPRK22(0.5); MPRK22(2.0 / 3.0); MPRK22(1.0); MPRK43I(1.0, 0.5); MPRK43I(0.5, 0.75);
        MPRK43II(0.5); MPRK43II(2.0 / 3.0)]
labels = ["MPRK22(0.5)"; "MPRK22(2/3)"; "MPRK22(1.0)"; "MPRK43I(1.0,0.5)"; "MPRK43I(0.5,0.75)";
         "MPRK43II(0.5)"; "MPRK43II(2.0/3.0)"]

# compute work-precision data
wp = work_precision_adaptive(prob, algs, labels, abstols, reltols, alg_ref;
                            adaptive_ref = true, compute_error)

# plot work-precision diagram
plot(wp, labels; title = "Robertson benchmark", legend = :topright,     
     color = permutedims([repeat([1], 3)..., repeat([3], 2)..., repeat([4], 2)...]),
     xlims = (10^-10, 10^0), xticks = 10.0 .^ (-10:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

We see that the second and third order schemes perform very similar, with the exception of `MPRK22(0.5)`.
This superior performance of `MPRK22(0.5)` cannot be seen in other benchmarks is therefore an exception here.

The scheme `SSPMPRK22(0.5, 1.0)` has not been considered above, since it generates oscillatory solutions.

```@example ROBER
sol1 = solve(prob, SSPMPRK22(0.5, 1.0), abstol=1e-5, reltol = 1e-4);

# plot solutions
robertson_plot(sol1, ref_sol, "SSPMPRK22(0.5, 1.0)")
```

For comparisons with schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) we choose the second order schemes `MPRK22(0.5)` and `MPRK22(1.0)` as well as the third order scheme `MPRK43I(0.5, 0.75)`.

```@example ROBER
sol_MPRK22_½ = solve(prob, MPRK22(0.5); abstol, reltol)
sol_MPRK22_1 = solve(prob, MPRK22(1.0); abstol, reltol)
sol_MPRK43 = solve(prob, MPRK43I(0.5, 0.75); abstol, reltol)

p1 = robertson_plot(sol_MPRK22_½, ref_sol, "MPRK22(0.5)");
p2 = robertson_plot(sol_MPRK22_1, ref_sol, "MPRK22(1.0)");
p3 = robertson_plot(sol_MPRK43, ref_sol, "MPRK43I(0.5, 0.75)");
plot(p1, p2, p3)
```

Now we compare these three schemes with a selection of second and third order stiff solvers from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). To guarantee nonnegative solutions, we use the solver option `isoutofdomain = isnegative`.

```@example ROBER
# select reference MPRK methods
algs1 = [MPRK22(0.5); MPRK22(1.0); MPRK43I(0.5, 0.75)]
labels1 = ["MPRK22(0.5)"; "MPRK22(1.0)"; "MPRK43I(0.5,0.75)"]

# select methods from OrdinaryDiffEq
algs2 = [TRBDF2(); Kvaerno3(); KenCarp3(); Rodas3(); ROS2(); ROS3(); Rosenbrock23()]
labels2 = ["TRBDF2"; "Kvearno3"; "KenCarp3"; "Rodas3"; "ROS2"; "ROS3"; "Rosenbrock23"]

# compute work-precision data
wp = work_precision_adaptive(prob, algs1, labels1, abstols, reltols, alg_ref;
                               adaptive_ref = true, compute_error)
# add work-precision data with isoutofdomain = isnegative                               
work_precision_adaptive!(wp, prob, algs2, labels2, abstols, reltols, alg_ref;
                               adaptive_ref = true, compute_error, isoutofdomain=isnegative)

# plot work-precision diagram
plot(wp, [labels1; labels2]; title = "Robertson benchmark", legend = :topright,
     color = permutedims([repeat([1], 2)..., 3, repeat([5], 3)..., repeat([6], 4)...]),
     xlims = (10^-10, 10^3), xticks = 10.0 .^ (-14:1:3),
     ylims = (10^-6, 10^1), yticks = 10.0 .^ (-6:1:0), minorticks = 10)
```

We see that `MPRK22(1.0)` and `MPRK43I(0.5, 0.75)` perform similar to `Ros3()` or `Rosenbrock23()` and are a good choice as long as low accuracy is acceptable. For high accuracy we should employ a scheme like `KenCarp3()`. As for `MPRK22(0.5)` the  superior performance of `Rodas3()` seems to be an exception here.

In addition,  we compare `MPRK22(1.0)` and `MPRK43I(0.5, 0.75)` to some [recommended solvers](https://docs.sciml.ai/DiffEqDocs/dev/solvers/ode_solve/) of higher order from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). Again, to guarantee positive solutions we select the solver option `isoutofdomain = isnegative`.

```@example ROBER
algs3 = [Rodas5P(); Rodas4P(); RadauIIA5()]
labels3 = ["Rodas5P"; "Rodas4P"; "RadauIIA5"]

# compute work-precision data
wp = work_precision_adaptive(prob, algs1, labels1, abstols, reltols, alg_ref;
                               adaptive_ref = true, compute_error)
# add work-precision data with isoutofdomain = isnegative                                 
work_precision_adaptive!(wp, prob, algs3, labels3, abstols, reltols, alg_ref;
                               adaptive_ref = true, compute_error, isoutofdomain=isnegative)

# plot work-precision diagram
plot(wp, [labels1; labels3]; title = "Robertson benchmark", legend = :topright,
     color = permutedims([repeat([1],2)..., 3, repeat([4], 2)..., 5]),
     xlims = (10^-10, 2*10^0), xticks = 10.0 .^ (-10:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

Again, we see that the MPRK schemes are in general only beneficial if low accuracy is acceptable.

### Relative maximum error over all time steps

In this section we do not compare the relative maximum errors at the final time ``t = 10^{11}``, but the relative maximum errors over all time steps. 

```@example ROBER
# select relative maximum error at the end of the problem's time span.
compute_error = rel_max_error_overall
nothing # hide
```

First, we compare different MPRK schemes. As above, we omit `SSPMPRK22(0.5, 1.0)`.

```@example ROBER
# compute work-precision data
wp = work_precision_adaptive(prob, algs, labels, abstols, reltols, alg_ref;
                            adaptive_ref = true, compute_error)

# plot work-precision diagram
plot(wp, labels; title = "Robertson benchmark", legend = :top,     
     color = permutedims([repeat([1], 3)..., repeat([3], 2)..., repeat([4], 2)...]),
     xlims = (10^-4, 5*10^1), xticks = 10.0 .^ (-5:1:2),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

Notably, the error of the second-order methods does not decrease when stricter tolerances are used. We choose the second order scheme `MPRK22(1.0)` and the third order scheme `MPRK43I(0.5, 0.75)` for comparison with solvers from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). To guarantee nonnegative solutions of these methods, we select the solver option `isoutofdomain = isnegative`.

```@example ROBER
# select reference MPRK methods
algs1 = [MPRK22(1.0); MPRK43I(0.5, 0.75)]
labels1 = ["MPRK22(1.0)"; "MPRK43I(0.5,0.75)"]

# compute work-precision data
wp = work_precision_adaptive(prob, algs1, labels1, abstols, reltols, alg_ref;
                               adaptive_ref = true, compute_error)
# add work-precision data with isoutofdomain = isnegative                               
work_precision_adaptive!(wp, prob, algs2, labels2, abstols, reltols, alg_ref;
                               adaptive_ref = true, compute_error, isoutofdomain=isnegative)

# plot work-precision diagram
plot(wp, [labels1; labels2]; title = "Robertson benchmark", legend = :bottomleft,
     color = permutedims([1, 3, repeat([5], 3)..., repeat([6], 4)...]),
     xlims = (10^-5, 10^2), xticks = 10.0 .^ (-14:1:3),
     ylims = (10^-6, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

Here too, some methods show that the error does not decrease even though stricter tolerances are used.

Finally, we compare `MPRK43I(0.5, 0.75)` and `MPRK22(1.0)` to [recommended solvers](https://docs.sciml.ai/DiffEqDocs/dev/solvers/ode_solve/) of higher order from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). Again, to guarantee positive solutions we select the solver option `isoutofdomain = isnegative`.

```@example ROBER
# compute work-precision data
wp = work_precision_adaptive(prob, algs1, labels1, abstols, reltols, alg_ref;
                               adaptive_ref = true, compute_error)
# add work-precision data with isoutofdomain = isnegative                                 
work_precision_adaptive!(wp, prob, algs3, labels3, abstols, reltols, alg_ref;
                               adaptive_ref = true, compute_error, isoutofdomain=isnegative)

# plot work-precision diagram
plot(wp, [labels1; labels3]; title = "Robertson benchmark", legend = :topright,
     color = permutedims([1, 3, repeat([4], 2)..., 5]),
     xlims = (10^-4, 2*10^0), xticks = 10.0 .^ (-11:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

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
