# [Benchmark: Solution of an NPZD model](@id benchmark-npzd)

We use the NPZD model [`prob_pds_npzd`](@ref) to assess the efficiency of different solvers from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) and [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl).

```@example NPZD
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqTsit5, OrdinaryDiffEqVerner
using PositiveIntegrators

# select NPZD problem
prob = prob_pds_npzd
nothing # hide
```

To keep the following code as clear as possible, we define a helper function `npzd_plot` that we use for plotting.

```@example NPZD
using Plots

npzd_plot = function(sol, sol_ref = nothing, title = "")
     colors = palette(:default)[1:4]'
     if !isnothing(sol_ref)
          p = plot(sol_ref, linestyle = :dash, label = "", color = colors, 
          linewidth = 2)
          plot!(p, sol; denseplot = false, markers = :circle, ylims = (-1.0, 10.0),
          color = colors, title, label = ["N" "P" "Z" "D"], legend = :right,
          linewidth = 2);
     else
          p = plot(sol; denseplot = false, markers = :circle, ylims = (-1.0, 10.0),
          color = colors, title, label = ["N" "P" "Z" "D"], legend = :right, 
          linewidths = 2);
     end
     return p
end
nothing  # hide
```

Standard methods have difficulties to solve the NPZD problem accurately for loose tolerances or large time step sizes.
The reason for this is that there is only a tiny margin for negative values in ``N``. 
In most cases negative values of ``N``, will directly lead to a further decrease in ``N`` and thus completely inaccurate solutions.  

```@example NPZD
# compute reference solution for plotting
ref_sol = solve(prob, Vern7(); abstol = 1e-14, reltol = 1e-13);

# compute solutions with loose tolerances
abstol = 1e-2
reltol = 1e-1
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol);
sol_MPRK = solve(prob, MPRK22(1.0); abstol, reltol);

# plot solutions
p1 = npzd_plot(sol_Ros23, ref_sol, "Rosenbrock23"); # helper function defined above
p2 = npzd_plot(sol_MPRK, ref_sol, "MPRK22(1.0)"); 
plot(p1, p2)
```

Nevertheless, [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used in combination with [`isnegative`](@ref) to guarantee nonnegative solutions. 

```@example NPZD
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative); #reject negative solutions

npzd_plot(sol_Ros23, ref_sol) #auxiliary function defined above
```

## Work-Precision diagrams

In the following sections we show several work-precision diagrams, which compare different methods with respect to computing time and error. 
First we focus on adaptive methods, afterwards we also show results obtained with fixed time step sizes.

Since the NPZD problem is not stiff, we can use an explicit high-order scheme to compute a reference solution.

```@example NPZD
# select solver to compute reference solution
alg_ref = Vern7()
nothing  # hide
```

### Adaptive schemes

We use the functions [`work_precision_adaptive`](@ref) and [`work_precision_adaptive!`](@ref) to compute the data for the diagrams.
Furthermore, the following absolute and relative tolerances are used.

```@example NPZD
# set absolute and relative tolerances
abstols = 1.0 ./ 10.0 .^ (2:1:8)
reltols = abstols .* 10.0
nothing  # hide
```

#### Relative maximum error at the final time

In this section the chosen error is the relative maximum error at the final time ``t = 10.0``.

```@example NPZD
# select relative maximum error at the end of the problem's time span.
compute_error = rel_max_error_tend
nothing # hide
```

We start with a comparison of different adaptive MPRK schemes.

```@example NPZD
# choose methods to compare
algs = [MPRK22(0.5); MPRK22(2.0 / 3.0); MPRK22(1.0); SSPMPRK22(0.5, 1.0); 
        MPRK43I(1.0, 0.5); MPRK43I(0.5, 0.75); MPRK43II(0.5); MPRK43II(2.0 / 3.0)]
labels = ["MPRK22(0.5)"; "MPPRK22(2/3)"; "MPRK22(1.0)"; "SSPMPRK22(0.5,1.0)"; 
          "MPRK43I(1.0, 0.5)"; "MPRK43I(0.5, 0.75)"; "MPRK43II(0.5)"; "MPRK43II(2.0/3.0)"]

# compute work-precision data
wp = work_precision_adaptive(prob, algs, labels, abstols, reltols, alg_ref;
                               compute_error)

# plot work-precision diagram
plot(wp, labels; title = "NPZD benchmark", legend = :topright,     
     color = permutedims([repeat([1], 3)..., 2, repeat([3], 2)..., repeat([4], 2)...]),
     xlims = (10^-7, 2*10^-1), xticks = 10.0 .^ (-8:1:0),
     ylims = (10^-6, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

The second- and third-order methods behave very similarly. For comparisons with other schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) we choose `MPRK22(1.0)` and `MPRK43I(1.0, 0.5)`.

```@example NPZD
sol_MPRK22 = solve(prob, MPRK22(1.0); abstol, reltol)
sol_MPRK43 = solve(prob, MPRK43I(1.0, 0.5); abstol, reltol)

p1 = npzd_plot(sol_MPRK22, ref_sol, "MPRK22(1.0)");
p2 = npzd_plot(sol_MPRK43, ref_sol, "MPRK43I(1.0, 0.5)");
plot(p1, p2)
```

Next we compare `MPRK22(1.0)` and `MPRK43I(1.0, 0.5)` to explicit and implicit methods of second and third order from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). 
To guarantee nonnegative solutions, we select the solver option `isoutofdomain = isnegative`.

```@example NPZD
# select MPRK methods for reference
algs1 = [MPRK22(1.0); MPRK43I(1.0, 0.5)]
labels1 = ["MPRK22(1.0)"; "MPRK43I(1.0,0.5)"]

# select methods from OrdinaryDiffEq
algs2 = [Midpoint(); Heun(); Ralston(); TRBDF2(); SDIRK2(); Kvaerno3(); KenCarp3(); Rodas3();
         ROS2(); ROS3(); Rosenbrock23()]
labels2 = ["Midpoint"; "Heun"; "Ralston"; "TRBDF2"; "SDIRK2"; "Kvearno3"; "KenCarp3"; "Rodas3";
          "ROS2"; "ROS3"; "Rosenbrock23"]

# compute work-precision data
wp = work_precision_adaptive(prob, algs1, labels1, abstols, reltols, alg_ref;
                               compute_error)
# add work-precision data with isoutofdomain=isnegative
work_precision_adaptive!(wp, prob, algs2, labels2, abstols, reltols, alg_ref;
                               compute_error, isoutofdomain=isnegative)

plot(wp, [labels1; labels2]; title = "NPZD benchmark", legend = :topright,
     color = permutedims([1, 3, repeat([4], 3)..., repeat([5], 4)..., repeat([6], 4)...]),
     xlims = (5*10^-8, 2*10^-1), xticks = 10.0 .^ (-8:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

We see that for the NPZD problem the use of adaptive MPRK schemes is only beneficial when using the loosest tolerances.

Now we compare `MPRK22(1.0)` and `MPRK43I(1.0, 0.5)` to [recommended solvers](https://docs.sciml.ai/DiffEqDocs/dev/solvers/ode_solve/) from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). Again, to guarantee positive solutions we select the solver option `isoutofdomain = isnegative`.

```@example NPZD
algs3 = [Tsit5(); BS3(); Vern6(); Vern7(); Vern8(); TRBDF2(); Rosenbrock23(); 
         Rodas5P(); Rodas4P()]
labels3 = ["Tsit5"; "BS3"; "Vern6"; "Vern7"; "Vern8"; "TRBDF2"; "Rosenbrock23";
          "Rodas5P"; "Rodas4P"]

# compute work-precision data
wp = work_precision_adaptive(prob, algs1, labels1, abstols, reltols, alg_ref;
                               compute_error) 
# add work-precision data with isoutofdomain = isnegative
work_precision_adaptive!(wp, prob, algs3, labels3, abstols, reltols, alg_ref;
                               compute_error, isoutofdomain = isnegative)

# plot work-precision diagram
plot(wp, [labels1; labels3]; title = "NPZD benchmark", legend = :topright,
     color = permutedims([1, 3, repeat([4], 5)...,5, repeat([6], 1)...,repeat([7],2)...]),
     xlims = (10^-11, 10^1), xticks = 10.0 .^ (-11:1:1),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

We see that it is advisable to use a high order explicit method like `Vern7()` with `isoutofdomain = isnegative` to obtain nonnegative solutions of such a non-stiff problem.

#### Relative maximum error over all time steps

In this section we do not compare the relative maximum errors at the final time ``t = 10.0``, but the relative maximum errors over all time steps. 

```@example NPZD
# select relative maximum error over all time steps
compute_error = rel_max_error_overall
nothing  # hide
```

The results are very similar to those from above. 
We therefore only show the work-precision diagrams without further comments. 
The main difference are significantly increased errors which mainly occur around time ``t = 2.0`` where there is a sharp kink in the solution.

```@example NPZD
# compute work-precision data
wp = work_precision_adaptive(prob, algs, labels, abstols, reltols, alg_ref;
                               compute_error)

# plot work-precision diagram
plot(wp, labels; title = "NPZD benchmark", legend = :topright,     
          color = permutedims([repeat([1], 3)..., 2, repeat([3], 2)..., repeat([4], 2)...]),
          xlims = (10^-5, 10^4), xticks = 10.0 .^ (-5:1:4),
          ylims = (10^-6, 10^-1), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

```@example NPZD
# compute work-precision data
wp = work_precision_adaptive(prob, algs1, labels1, abstols, reltols, alg_ref;
                               compute_error)
# add work-precision data with isoutofdomain = isnegative
work_precision_adaptive!(wp, prob, algs2, labels2, abstols, reltols, alg_ref;
                               compute_error, isoutofdomain=isnegative)

# plot work-precision diagram
plot(wp, [labels1; labels2]; title = "NPZD benchmark", legend = :topright,
     color = permutedims([1, 3, repeat([4], 3)..., repeat([5], 4)..., repeat([6], 4)...]),
     xlims = (10^-5, 10^4), xticks = 10.0 .^ (-5:1:4),
     ylims = (10^-6, 10^0), yticks = 10.0 .^ (-6:1:0), minorticks = 10)
```

```@example NPZD
# compute work-precision data
wp = work_precision_adaptive(prob, algs1, labels1, abstols, reltols, alg_ref;
                               compute_error)
# add work-precision data with isoutofdomain = isnegative                             
work_precision_adaptive!(wp, prob, algs3, labels3, abstols, reltols, alg_ref;
                               compute_error, isoutofdomain=isnegative)

# plot work-precision diagram
plot(wp, [labels1; labels3]; title = "NPZD benchmark", legend = :topright,
     color = permutedims([1, 3, repeat([4], 5)...,5, repeat([6], 1)...,repeat([7],2)...]),
     xlims = (10^-7, 10^5), xticks = 10.0 .^ (-7:1:5),
     ylims = (10^-6, 10^0), yticks = 10.0 .^ (-6:1:0), minorticks = 10)
```

### Fixed time steps sizes

Here we use fixed time step sizes instead of adaptive time stepping. 
Similar to the adaptive situation above, standard schemes are likely to compute negative solutions for the NPZD problem. 

```@example NPZD
sol_Ros23 = solve(prob, Rosenbrock23(), dt = 1.0, adaptive = false);
sol_MPRK = solve(prob, MPRK22(1.0), dt = 1.0, adaptive = false);

p1 = npzd_plot(sol_Ros23, ref_sol, "Rosenbrock23");
p2 = npzd_plot(sol_MPRK, ref_sol, "MPRK22(1.0)");
plot(p1, p2)
```

We use the functions [`work_precision_fixed`](@ref) and [`work_precision_fixed!`](@ref) to compute the data for the diagrams.
Please note that these functions set error and computing time to `Inf`, whenever a solution contains negative elements. 
Consequently, such cases are not visible in the work-precision diagrams.

Within the work-precision diagrams we use the following time step sizes.

```@example NPZD
# set time step sizes
dts = 1.0 ./ 2.0 .^ (0:1:12)
nothing # hide
```

#### Relative maximum error at the end of the problem's time span

Again, we start with the relative maximum error at the final time ``t = 10.0``.

```@example NPZD
# select relative maximum error at the end of the problem's time span.
compute_error = rel_max_error_tend
nothing  # hide
```

First, we compare different MPRK methods. 
For fixed time step sizes we can also consider `MPE()` and `SSPMPRK43()`.

```@example NPZD
# choose MPRK methods to compare
algs = [MPE(); algs; SSPMPRK43()]
labels = ["MPE()"; labels; "SSPMPRK43"]

# compute work-precision data
wp = work_precision_fixed(prob, algs, labels, dts, alg_ref;
                               compute_error)

# plot work-precision diagram
plot(wp, labels; title = "NPZD benchmark", legend = :bottomleft,     
     color = permutedims([5,repeat([1], 3)..., 2, repeat([3], 2)..., repeat([4], 2)...,6]),
     xlims = (10^-10, 1*10^0), xticks = 10.0 .^ (-10:1:0),
     ylims = (1*10^-6, 10^-1), yticks = 10.0 .^ (-6:1:0), minorticks = 10)
```

Apart from `MPE()` the schemes behave very similar and a difference in order can only be observed for the smaller step sizes. 
We choose `MPRK22(1.0)` and `MPRK43I(1.0, 0.5)` for comparisons with other second and third order schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/).

```@example NPZD
# compute work-precision data
wp = work_precision_fixed(prob, [algs1; algs2], [labels1; labels2], dts, alg_ref;
                               compute_error)

# plot work-precision diagram
plot(wp, [labels1; labels2]; title = "NPZD benchmark", legend = :topright,
     color = permutedims([1, 3, repeat([4], 3)..., repeat([5],4)...,repeat([6],4)...]),
     xlims = (10^-13, 10^2), xticks = 10.0 .^ (-12:2:6),
     ylims = (10^-6, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)         
```

We see that the MPRK schemes are to be preferred for the rather large step sizes ``\Delta t \in\lbrace 1.0, 0.5, 0.25, 0.125\rbrace``, for which the other schemes cannot provide nonnegative solutions.

```@example NPZD
# solution computed with MPRK43I(1.0, 0.5) and dt = 0.125
sol_MPRK = solve(prob, MPRK43I(1.0, 0.5); dt = dts[4], adaptive = false);

# plot solution
npzd_plot(sol_MPRK, ref_sol)
```

Finally, we show a comparison of `MPRK22(1.0)`, `MPRK43I(1.0, 0.5)` and [recommended solvers](https://docs.sciml.ai/DiffEqDocs/dev/solvers/ode_solve/) from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/).

```@example NPZD
# compute work-precision data
wp = work_precision_fixed(prob, [algs1; algs3], [labels1; labels3], dts, alg_ref;
                               compute_error)

# plot work-precision diagram
plot(wp, [labels1; labels3]; title = "NPZD benchmark", legend = :topright,
     color = permutedims([1, 3, repeat([4], 3)..., repeat([5],4)...,repeat([6],4)...]),
     xlims = (10^-14, 10^0), xticks = 10.0 .^ (-14:2:10),
     ylims = (10^-6, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)         
```

#### Relative maximum error over all time steps

As for the adaptive schemes, we also show work-precisions diagrams where the error is the relative maximum error over all time steps.

```@example NPZD
# select relative maximum error over all time steps
compute_error = rel_max_error_overall
nothing  # hide
```

```@example NPZD

# compute work-precision
wp = work_precision_fixed(prob, algs, labels, dts, alg_ref;
                               compute_error)

#plot work-precision diagram
plot(wp, labels; title = "NPZD benchmark", legend = :bottomleft,     
     color = permutedims([5,repeat([1], 3)..., 2, repeat([3], 2)..., repeat([4], 2)...,6]),
     xlims = (10^-4, 10^5), xticks = 10.0 .^ (-4:1:5),
     ylims = (10^-6, 10^-1), yticks = 10.0 .^ (-6:1:0), minorticks = 10)
```

```@example NPZD
wp = work_precision_fixed(prob, algs1, labels1, dts, alg_ref;
                               compute_error)
work_precision_fixed!(wp, prob, algs2, labels2, dts, alg_ref;
                     compute_error)                               

plot(wp, [labels1; labels2]; title = "NPZD benchmark", legend = :topright,
     color = permutedims([1, 3, repeat([4], 3)..., repeat([5], 4)..., repeat([6], 4)...]),
     xlims = (10^-4, 10^6), xticks = 10.0 .^ (-12:1:6),
     ylims = (10^-6, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)         
```

```@example NPZD
wp = work_precision_fixed(prob, algs1, labels1, dts, alg_ref;
                               compute_error)
work_precision_fixed!(wp, prob, algs3, labels3, dts, alg_ref;
                     compute_error)                               

plot(wp, [labels1; labels3]; title = "NPZD benchmark", legend = :bottomleft,
     color = permutedims([1, 3, repeat([4], 5)..., 5, repeat([7], 3)...]),
     xlims = (10^-12, 10^6), xticks = 10.0 .^ (-12:2:6),
     ylims = (10^-6, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)         
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
