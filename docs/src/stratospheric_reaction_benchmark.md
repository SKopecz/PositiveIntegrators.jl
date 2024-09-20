# [Benchmark: Solution of a stratospheric reaction problem](@id benchmark-stratos)

We use the stiff stratospheric reacation problem [`prob_pds_stratreac`](@ref) to assess the efficiency of different solvers.


```@example stratreac
using OrdinaryDiffEq, PositiveIntegrators
using Plots

# select problem
prob = prob_pds_stratreac

# compute reference solution 
ref_sol = solve(prob, Rodas4P(); abstol = 1e-12, reltol = 1e-11);

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
    widen = true)
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

```@example stratreac
# compute reference solution
tspan = prob.tspan
dt_ref = (last(tspan) - first(tspan)) ./ 1e5
sol_ref = solve(prob, Rodas4P(); dt = dt_ref, adaptive = false, save_everystep = false);
sol_ref = sol_ref.u[end]

alg_ref = Rodas4P()

# define error functions
l2_error(sol, sol_ref) = sqrt(sum(((sol .- sol_ref) ./ sol_ref) .^ 2) / length(sol_ref))
l∞_error(sol, sol_ref) = maximum(abs.((sol .- sol_ref) ./ sol_ref))
nothing #hide output
```
### Adaptive time stepping

```@example stratreac
abstols = 1.0 ./ 10.0 .^ (2:1:5)
reltols = 10.0 .* abstols
nothing # hide output
```

Remark: Stricter tolerances will require more than a million steps!

#### L∞ errors

First we compare different (adaptive) MPRK schemes described in the literature. 

```@example stratreac

# choose methods to compare
algs = [#MPRK22(0.5) # fail
        #MPRK22(0.5, small_constant = 1e-6) #fail
        #MPRK22(2.0 / 3.0) #fail
        #MPRK22(2.0 / 3.0, small_constant = 1e-6) #fail
         MPRK22(1.0)
         MPRK22(1.0, small_constant = 1e-6)
        #SSPMPRK22(0.5, 1.0) # takes too long
         MPRK43I(1.0, 0.5)
         MPRK43I(1.0, 0.5, small_constant = 1e-6)
         MPRK43I(0.5, 0.75)
         MPRK43I(0.5, 0.75, small_constant = 1e-6)
         MPRK43II(0.5)
         MPRK43II(0.5, small_constant = 1e-6)
         MPRK43II(2.0 / 3.0)
         MPRK43II(2.0 / 3.0, small_constant = 1e-6)]

names = [#"MPRK22(0.5)"
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

# compute work-precision
wp_l∞ = workprecision_adaptive(prob, algs, names, abstols, reltols, alg_ref;
                               compute_error = PositiveIntegrators.rel_l∞_error_at_end)

plot(wp_l∞, names; title = "Stratospheric reaction benchmark (l∞)", legend = :bottomleft,     
     color = permutedims([repeat([1],2)...,repeat([3],4)...,repeat([4],4)...]),
     xlims = (10^-5, 10^0), xticks = 10.0 .^ (-8:1:0),
     ylims = (10^-5, 10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```


All methods using `small_constant = 1e-6` behave similar, irrespective of the method's order.
For comparisons with other second and third order schemes from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) we choose the third order scheme `MPRK43I(1.0, 0.5)`. To guarantee positive solutions of the [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) methods, we must select the solver option `isoutofdomain = isnegative`.

```@example stratreac
# select methods
algs2 = [MPRK43I(1.0, 0.5)
    TRBDF2()
    Kvaerno3()
    KenCarp3()
    Rodas3()
    ROS2()
    ROS3()
    Rosenbrock23()]

names2 = ["MPRK43I(1.0,0.5)"
          "TRBDF2"
          "Kvearno3"
          "KenCarp3"
          "Rodas3"
          "ROS2"
          "ROS3"
          "Rosenbrock23"]

# compute work-precision
wp_l∞ = workprecision_adaptive(prob, algs2, names2, abstols, reltols, alg_ref;
                               compute_error = PositiveIntegrators.rel_l∞_error_at_end)

#
plot(wp_l∞, names2; title = "Stratospheric reaction benchmark (l∞)", legend = :topright,     
     color = permutedims([3, repeat([4], 3)..., repeat([5], 4)...]),
     xlims = (10^-7, 10^-1), xticks = 10.0 .^ (-8:1:0),
     ylims = (2*10^-4, 5*10^0), yticks = 10.0 .^ (-5:1:0), minorticks = 10)
```

### Fixed time steps sizes

```@example stratreac
# set time step sizes
dt0 = 48 * 60 #48 minutes
dts = dt0 ./ 2.0 .^ (0:1:6)
nothing #hide output
```

#### L∞

```@example stratreac
# choose methods to compare
algs = [MPE()
        MPE(small_constant=1e-6)
        #MPRK22(0.5) # fail
        #MPRK22(0.5, small_constant = 1e-6) #fail
        #MPRK22(2.0 / 3.0) #fail
        #MPRK22(2.0 / 3.0, small_constant = 1e-6) #fail
        MPRK22(1.0)
        MPRK22(1.0, small_constant = 1e-6)
        #SSPMPRK22(0.5, 1.0) # takes too long
        MPRK43I(1.0, 0.5)
        MPRK43I(1.0, 0.5, small_constant = 1e-6)
        MPRK43I(0.5, 0.75)
        MPRK43I(0.5, 0.75, small_constant = 1e-6)
        MPRK43II(0.5)
        MPRK43II(0.5, small_constant = 1e-6)
        MPRK43II(2.0 / 3.0)
        MPRK43II(2.0 / 3.0, small_constant = 1e-6)
        SSPMPRK43()
        SSPMPRK43(small_constant = 1e-6)]

names = ["MPE()"
         "MPE(sc=1e-6)"
         #"MPRK22(0.5)"
         #"MPRK22(0.5, sc=1e-6)"
         #"MPPRK22(2/3)"
         #"MPPRK22(2/3, sc=1e-6)"
         "MPRK22(1.0)"
         "MPRK22(1.0, sc=1e-6)"
         #"SSPMPRK22(0.5,1.0)"
         "MPRK43I(1.0,0.5, sc=1e-6)"
         "MPRK43I(0.5,0.75)"
         "MPRK43I(0.5,0.75, sc=1e-6)"
         "MPRK43II(0.5)"
         "MPRK43II(0.5, sc=1e-6)"
         "MPRK43II(2.0/3.0)"
         "MPRK43II(2.0/3.0, sc=1e-6)"
         "SSPMPRK43()"
         "SSPMPRK43(sc=1e-6)"]



# compute work-precision
wp_l∞ = workprecision_fixed(prob, algs, names, dts, alg_ref;
                               compute_error = PositiveIntegrators.rel_l∞_error_at_end)

plot(wp_l∞, names; title = "Stratospheric reaction benchmark (l∞)", legend = :topright,     
     color = permutedims([repeat([5],2)...,repeat([1], 2)..., repeat([3], 4)..., repeat([4], 4)...,repeat([6],2)...]),
     xlims = (10^-5, 5*10^1), xticks = 10.0 .^ (-10:1:1),
     ylims = (1*10^-5, 10^-1), yticks = 10.0 .^ (-6:1:0), minorticks = 10
     )
```

```@example stratreac
algs2 = [MPE(small_constant = 1e-6)
         MPRK22(1.0, small_constant = 1e-6)
         ROS2()]

names2 = ["MPE(sc=1e-6)"
          "MPRK22(1.0, sc=1e-6)"
          "ROS2"]

algs3 = [Rodas3()         
         ROS3()
         Rosenbrock23()]         

names3 = ["Rodas3"
          "ROS3"
          "Rosenbrock23" ]

algs4 = [TRBDF2()
    Kvaerno3()
    KenCarp3()]

names4 = ["TRBDF2"
          "Kvearno3"
          "KenCarp3"]

compute_error = PositiveIntegrators.rel_l∞_error_at_end
wp_l∞ = workprecision_fixed(prob, algs2, names2, dts, alg_ref;
                               compute_error)
workprecision_fixed!(wp_l∞, prob, algs3, names3, dts[3:end], alg_ref;
                               compute_error)
workprecision_fixed!(wp_l∞, prob, algs4, names4, dts[5:end], alg_ref;
                               compute_error)

plot(wp_l∞, [names2; names3; names4]; title = "Stratospheric reaction benchmark (l∞)", legend = :bottomleft,
     color = permutedims([1, 3, repeat([4], 3)...]),
     xlims = (10^-11, 10^2), xticks = 10.0 .^ (-12:1:2),
     ylims = (2*10^-5, 10^-1), yticks = 10.0 .^ (-5:1:0), minorticks = 10)         
```

```@example stratreac
sol_MPRK = solve(prob, MPRK22(1.0, small_constant = 1e-6); dt = dts[3], adaptive = false);

plot(ref_sol, layout = (3,2),
    xguide = "t [h]", xguidefontsize = 8,
    xticks = (range(first(tspan), last(tspan), 4), range(12.0, 84.0, 4)), tickfontsize = 7,
    yguide=["O¹ᴰ" "O" "O₃" "O₂" "NO" "NO₂"],    
    linestyle = :dash, label = "",    
    legend = :outertop, legend_column = -1,
    widen = true)
plot!(sol_MPRK, label = "MPRK22(1.0)", denseplot = false, markers = :circle)
```

## Literature
- Kopecz, Meister 2nd order
- Kopecz, Meister 3rd order
- Huang, Shu 2nd order


## Package versions

These results were obtained using the following versions.
```@example STRATREAC
using InteractiveUtils
versioninfo()
println()

using Pkg
Pkg.status(["PositiveIntegrators", "StaticArrays", "LinearSolve", "OrdinaryDiffEq"],
           mode=PKGMODE_MANIFEST)
nothing # hide
```
