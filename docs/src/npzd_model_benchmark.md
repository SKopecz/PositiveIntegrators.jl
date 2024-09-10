# [Benchmark: Solution of an NPZD model](@id benchmark-npzd)

We use the NPZD model [`prob_pds_npzd`](@ref) to assess the efficiency of different solvers.

Standard methods have difficulties to solve this problem accurately, at least for low tolerances.

```@example NPZD
using OrdinaryDiffEq, PositiveIntegrators
using Plots

# select problem
prob = prob_pds_npzd

# compute reference solution (standard tolerances are too low)
ref_sol = solve(prob, Vern7(); abstol = 1e-14, reltol = 1e-13)

# compute solutions with lose tolerances
abstol = 1e-2
reltol = 1e-1
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol)
sol_MPRK = solve(prob, MPRK22(1.0); abstol, reltol)

# plot solutions
p1 = plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(p1, sol_Ros23; denseplot = false, markers = :circle, ylims = (-1.0, 10.0),
      title = "Rosenbrock23", label = ["N" "P" "Z" "D"])
p2 = plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(p2, sol_MPRK; denseplot = false, markers = true, ylims = (-1.0, 10.0),
     title = "MPRK22(1.0)", label = ["N" "P" "Z" "D"])
plot(p1, p2)
```

Nevertheless, [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) provides the solver option `isoutofdomain`, which can be used to guarantee nonnegative solutions.

```@example NPZD
sol_Ros23 = solve(prob, Rosenbrock23(); abstol, reltol, 
                  isoutofdomain = isnegative) #reject negative solutions

plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(sol_Ros23; denseplot = false, markers = :circle, ylims = (-1.0, 10.0),
          title = "Rosenbrock23", label = ["N" "P" "Z" "D"])
```

## Work-Precision diagrams
```@example NPZD
# compute reference solution
tspan = prob.tspan
dt_ref = (last(tspan) - first(tspan)) ./ 1e5
sol_ref = solve(prob, Vern7(); dt = dt_ref, adaptive = false, save_everystep = false);
sol_ref = sol_ref.u[end]

# define error functions
l2_error(sol, sol_ref) = sqrt(sum(((sol .- sol_ref) ./ sol_ref) .^ 2) / length(sol_ref))
l∞_error(sol, sol_ref) = maximum(abs.((sol .- sol_ref) ./ sol_ref))
```

### Fixed time steps sizes

### Adaptive schemes
First we compare different (adaptive) MPRK schemes described in the literature. The chosen `l∞` error computes the maximum of the absolute values of the difference between the numerical solution and the reference solution over all components and all time steps.
```@example
# set tolerances and error
abstols = 1.0 ./ 10.0 .^ (2:1:8)
reltols = abstols ./ 10.0
```

#### L∞ error

```@example NPZD
# choose methods to compare
algs = [MPRK22(0.5)
        MPRK22(2.0 / 3.0)
        MPRK22(1.0)
        SSPMPRK22(0.5, 1.0)
        MPRK43I(1.0, 0.5)
        MPRK43I(0.5, 0.75)
        MPRK43II(0.5)
        MPRK43II(2.0 / 3.0)]

names = ["MPRK22(0.5)"
         "MPPRK22(2/3)"
         "MPRK22(1.0)"
         "SSPMPRK22(0.5,1.0)"
         "MPRK43I(1.0,0.5)"
         "MPRK43I(0.5,0.75)"
         "MPRK43II(0.5)"
         "MPRK43II(2.0/3.0)"]

# compute work-precision
wp_l∞ = workprecision_adaptive(prob, algs, names, sol_ref, abstols, reltols,
                               compute_error = l∞_error)

plot(wp_l∞, names; title = "NPZD benchmark (l∞)", legend = :topright,     
     color = permutedims([repeat([1], 3)..., 2, repeat([3], 2)..., repeat([4], 2)...]),
     xlims = (10^-8, 10^-1), xticks = 10.0 .^ (-8:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10,)
```

The second- and third-order methods behave very similarly. For comparisons with other schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) we choose the schemes with the smallest error for the initial tolerances, respectively. These are `SSPMPRK22(0.5, 1.0)` and `MPRK43I(1.0, 0.5)`.

```@example NPZD
sol_SSPMPRK22 = solve(prob, SSPMPRK22(0.5, 1.0); abstol, reltol)
sol_MPRK43 = solve(prob, MPRK43I(1.0, 0.5); abstol, reltol)

p1 = plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(p1, sol_SSPMPRK22; denseplot = false, markers = :circle, ylims = (-1.0, 10.0),
      title = "SSPMPRK22(0.5, 1.0)", label = ["N" "P" "Z" "D"])
p2 = plot(ref_sol, linestyle = :dash, label = "", legend = :right)
plot!(p2, sol_MPRK43; denseplot = false, markers = true, ylims = (-1.0, 10.0),
     title = "MPRK43I(1.0, 0.5)", label = ["N" "P" "Z" "D"])
plot(p1, p2)
```

Although the SSPMPRK22 solution seems to be more accurate at first glance, the `l∞`-error of the SSPMPRK22 scheme is 0.506359, whereas the `l∞`-error of the MPRK43 scheme is 0.413915. Both errors occurs at approximately $t=2$, where there is a sharp kink in the first component.


Next we compare `SSPMPRK22(0.5, 1.0)` and `MPRK43I(1.0, 0.5)` with some second and third order methods from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). To guarantee positive solutions with these methods, we must select the solver option `isoutofdomain = isnegative`.

```@example NPZD
# select methods
algs1 = [SSPMPRK22(0.5, 1.0)
         MPRK43I(1.0, 0.5)]

algs2 = [Midpoint()
         Heun()
         Ralston()
         TRBDF2()
         SDIRK2()
         Kvaerno3()
         KenCarp3()
         Rodas3()
         ROS2()
         ROS3()
         Rosenbrock23()]

names1 = ["SSPMPRK22(0.5,1.0)"
          "MPRK43I(1.0,0.5)"]

names2 = ["Midpoint"
          "Heun"
          "Ralston"
          "TRBDF2"
          "SDIRK2"
          "Kvearno3"
          "KenCarp3"
          "Rodas3"
          "ROS2"
          "ROS3"
          "Rosenbrock23"]

compute_error = l∞_error
wp_l∞ = workprecision_adaptive(prob, algs1, names1, sol_ref, abstols, reltols;
                               compute_error)
workprecision_adaptive!(wp_l∞, prob, algs2, names2, sol_ref, abstols, reltols;
                               compute_error, isoutofdomain=isnegative)

plot(wp_l∞, [names1; names2]; title = "NPZD benchmark (l∞)", legend = :topright,
     color = permutedims([2, 3, repeat([4], 3)..., repeat([5], 4)..., repeat([6], 4)...]),
     xlims = (10^-8, 10^-1), xticks = 10.0 .^ (-8:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)

```

Comparison to recommend solvers.
```@example NPZD
algs2 = [Tsit5(),
    BS3(),
    Vern6(),
    Vern7(),
    Vern8(),
    TRBDF2(),
    Rosenbrock32(),
    Rodas5P(),
    Rodas4P()]

names2 = ["Tsit6"
          "BS3"
          "Vern6"
          "Vern7"
          "Vern8"
          "TRBDF2"
          "Rosenbrock23"
          "Rodas5P"
          "Rodas4P"]

compute_error = l∞_error
wp_l∞ = workprecision_adaptive(prob, algs1, names1, sol_ref, abstols, reltols;
                               compute_error)                             
workprecision_adaptive!(wp_l∞, prob, algs2, names2, sol_ref, abstols, reltols;
                               compute_error, isoutofdomain=isnegative)

plot(wp_l∞, [names1; names2]; title = "NPZD benchmark (l∞)", legend = :topright,
     color = permutedims([2, 3, repeat([4], 3)..., repeat([5], 4)..., repeat([6], 4)...]),
     xlims = (10^-11, 10^-1), xticks = 10.0 .^ (-11:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)

```

#### L2 error
```@example
wp_l2 = workprecision_adaptive(prob, algs, names, sol_ref, abstols, reltols,
                               compute_error = l2_error)

plot(wp_l2, names; title = "NPZD benchmark (l2)", legend = :topright,     
          color = permutedims([repeat([1], 3)..., 2, repeat([3], 2)..., repeat([4], 2)...]),
          xlims = (1 * 10^-8, 10^-1), xticks = 10.0 .^ (-8:1:0),
          ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10,)
```
```@example NPZD
# select methods
algs1 = [SSPMPRK22(0.5, 1.0)
         MPRK43I(1.0, 0.5)]

algs2 = [Midpoint()
         Heun()
         Ralston()
         TRBDF2()
         SDIRK2()
         Kvaerno3()
         KenCarp3()
         Rodas3()
         ROS2()
         ROS3()
         Rosenbrock23()]

names1 = ["SSPMPRK22(0.5,1.0)"
          "MPRK43I(1.0,0.5)"]

names2 = ["Midpoint"
          "Heun"
          "Ralston"
          "TRBDF2"
          "SDIRK2"
          "Kvearno3"
          "KenCarp3"
          "Rodas3"
          "ROS2"
          "ROS3"
          "Rosenbrock23"]

compute_error = l2_error
wp_l2 = workprecision_adaptive(prob, algs1, names1, sol_ref, abstols, reltols;
                               compute_error)
workprecision_adaptive!(wp_l2, prob, algs2, names2, sol_ref, abstols, reltols;
                               compute_error, isoutofdomain=isnegative)

plot(wp_l2, [names1; names2]; title = "NPZD benchmark (l2)", legend = :topright,
     color = permutedims([2, 3, repeat([4], 3)..., repeat([5], 4)..., repeat([6], 4)...]),
     xlims = (10^-8, 10^-1), xticks = 10.0 .^ (-8:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)

```
```@example NPZD
algs2 = [Tsit5(),
    BS3(),
    Vern6(),
    Vern7(),
    Vern8(),
    TRBDF2(),
    Rosenbrock32(),
    Rodas5P(),
    Rodas4P()]

names2 = ["Tsit6"
          "BS3"
          "Vern6"
          "Vern7"
          "Vern8"
          "TRBDF2"
          "Rosenbrock23"
          "Rodas5P"
          "Rodas4P"]

compute_error = l2_error
wp_l2 = workprecision_adaptive(prob, algs1, names1, sol_ref, abstols, reltols;
                               compute_error)                             
workprecision_adaptive!(wp_l2, prob, algs2, names2, sol_ref, abstols, reltols;
                               compute_error, isoutofdomain=isnegative)

plot(wp_l2, [names1; names2]; title = "NPZD benchmark (l2)", legend = :topright,
     color = permutedims([2, 3, repeat([4], 3)..., repeat([5], 4)..., repeat([6], 4)...]),
     xlims = (10^-11, 10^-1), xticks = 10.0 .^ (-11:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)

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
