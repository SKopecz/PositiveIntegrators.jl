# Install packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()

# load packages
using PositiveIntegrators
using OrdinaryDiffEq
using Plots
using DiffEqDevTools, PrettyTables

# load utility function for the assessment of the order of numerical schemes
include("utilities.jl")

# functions to check if invariants are preserved
# quick implementation. definitively need something better
f2(t, x, y) = (t, x + y)
f3(t, x, y, z) = (t, x + y + z)
f4(t, u1, u2, u3, u4) = (t, u1 + u2 + u3 + u4)
f_npzd(t, u1, u2, u3, u4) = (t, 0.66 * (u1 + u2 + u3 + u4))
f6(t, u1, u2, u3, u4, u5, u6) = (t, u1 + u2 + u3 + u4 + u5 + u6)
f_brusselator(t, u1, u2, u3, u4, u5, u6) = (t, 0.55 * (u1 + u2 + u3 + u4 + u5 + u6))

## linear model ##########################################################
sol_linmod = solve(prob_pds_linmod, Tsit5());
sol_linmod_MPE = solve(prob_pds_linmod, MPE(), dt = 0.2);
sol_linmod_MPRK22 = solve(prob_pds_linmod, MPRK22(0.5), dt = 0.2);
sol_linmod_MPRK43I = solve(prob_pds_linmod, MPRK43I(1.0, 0.5), dt = 0.2);
sol_linmod_MPRK43II = solve(prob_pds_linmod, MPRK43II(2.0 / 3.0), dt = 0.2);

# plot
p1 = plot(sol_linmod)
myplot!(sol_linmod_MPE, "MPE")
plot!(sol_linmod_MPE, idxs = (f2, 0, 1, 2))
p2 = plot(sol_linmod)
myplot!(sol_linmod_MPRK22, "MPRK22")
plot!(sol_linmod_MPRK22, idxs = (f2, 0, 1, 2))
p3 = plot(sol_linmod)
myplot!(sol_linmod_MPRK43I, "MPRK43I")
plot!(sol_linmod_MPRK43I, idxs = (f2, 0, 1, 2))
p4 = plot(sol_linmod)
myplot!(sol_linmod_MPRK43II, "MPRK43II")
plot!(sol_linmod_MPRK43II, idxs = (f2, 0, 1, 2))
plot(p1, p2, p3, p4)
plot!(legend = :none)

# convergence order
# error based on analytic solution
sims = convergence_tab_plot(prob_pds_linmod, [MPE(), Euler()]; dts = 0.5 .^ (3:18),
                            analytic = true, order_plot = true);
@assert sims[1].𝒪est[:l∞] > 0.9
#savefig("figs/error_linmod_analytic.svg")
# error based on reference solution
test_setup = Dict(:alg => Vern9(), :reltol => 1e-14, :abstol => 1e-14);
sims = convergence_tab_plot(prob_pds_linmod, [MPE(), Euler()], test_setup;
                            dts = 0.5 .^ (3:18), order_plot = true);
@assert sims[1].𝒪est[:l∞] > 0.9
#savefig("figs/error_linmod_reference.svg")

sims = convergence_tab_plot(prob_pds_linmod,
                            [MPRK22(1.0), MPRK22(2 / 3), MPRK22(0.5), Heun()], test_setup;
                            dts = 0.5 .^ (5:18), order_plot = true);
for i in 1:4
    @assert sims[i].𝒪est[:l∞] > 1.9
end

sims = convergence_tab_plot(prob_pds_linmod,
                            [MPRK43I(1.0, 0.5), MPRK43II(2.0 / 3.0), Rodas3()], test_setup;
                            dts = 0.5 .^ (5:12), order_plot = true);
for i in 1:3
    @assert sims[i].𝒪est[:l∞] > 2.9
end
## nonlinear model ########################################################
sol_nonlinmod = solve(prob_pds_nonlinmod, Tsit5());
sol_nonlinmod_MPE = solve(prob_pds_nonlinmod, MPE(), dt = 1.0);
sol_nonlinmod_MPRK22 = solve(prob_pds_nonlinmod, MPRK22(1.0), dt = 1.0, adaptive = false);
sol_nonlinmod_MPRK43I = solve(prob_pds_nonlinmod, MPRK43I(1.0, 0.5), dt = 1.0,
                              adaptive = false);
sol_nonlinmod_MPRK43II = solve(prob_pds_nonlinmod, MPRK43II(2.0 / 3.0), dt = 1.0,
                               adaptive = false);

# plot
p1 = plot(sol_nonlinmod)
myplot!(sol_nonlinmod_MPE, "MPE")
plot!(sol_nonlinmod_MPE, idxs = (f3, 0, 1, 2, 3))
p2 = plot(sol_nonlinmod)
myplot!(sol_nonlinmod_MPRK22, "MPRK22")
plot!(sol_nonlinmod_MPRK22, idxs = (f3, 0, 1, 2, 3))
p3 = plot(sol_nonlinmod)
myplot!(sol_nonlinmod_MPRK43I, "MPRK43I")
plot!(sol_nonlinmod_MPRK43I, idxs = (f3, 0, 1, 2, 3))
p4 = plot(sol_nonlinmod)
myplot!(sol_nonlinmod_MPRK43II, "MPRK43II")
plot!(sol_nonlinmod_MPRK43II, idxs = (f3, 0, 1, 2, 3))
plot(p1, p2, p3, p4, layout = (2, 2))
plot!(legend = :none)

# convergence order
test_setup = Dict(:alg => Vern9(), :reltol => 1e-14, :abstol => 1e-14)
sims = convergence_tab_plot(prob_pds_nonlinmod, [MPE(), Euler()], test_setup;
                            dts = 0.5 .^ (3:12), order_plot = true);
@assert sims[1].𝒪est[:l∞] > 0.9

sims = convergence_tab_plot(prob_pds_nonlinmod,
                            [MPRK22(1.0), MPRK22(2 / 3), MPRK22(0.5), Heun()], test_setup;
                            dts = 0.5 .^ (3:12), order_plot = true);
for i in 1:4
    @assert sims[i].𝒪est[:l∞] > 1.9
end

sims = convergence_tab_plot(prob_pds_nonlinmod,
                            [MPRK43I(1.0, 0.5), MPRK43II(2.0 / 3.0), Rodas3()], test_setup;
                            dts = 0.5 .^ (3:12), order_plot = true);
for i in 1:3
    @assert sims[i].𝒪est[:l∞] > 1.9
end
## robertson problem ######################################################
sol_robertson = solve(prob_pds_robertson, Rosenbrock23());
# Cannot use MPE() since adaptive time stepping is not implemented
sol_robertson_MPRK22 = solve(prob_pds_robertson, MPRK22(1.0), reltol = 1e-3, abstol = 1e-2);
sol_robertson_MPRK43I = solve(prob_pds_robertson, MPRK43I(1.0, 0.5), reltol = 1e-3,
                              abstol = 1e-2);
sol_robertson_MPRK43II = solve(prob_pds_robertson, MPRK43II(2.0 / 3.0), reltol = 1e-3,
                               abstol = 1e-2);

# plot
tspan = (1e-6, 1e11)
p1 = plot(sol_robertson, tspan = tspan,
          idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)],
          color = palette(:default)[1:3]', xaxis = :log)
plot!(sol_robertson, tspan = tspan, idxs = (f3, 0, 1, 2, 3), xaxis = :log)
myplot!(sol_robertson_MPRK22[2:end], tspan = tspan,
        idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], xaxis = :log)
myplot!(sol_robertson_MPRK22[2:end], "MPRK22"; tspan = tspan, idxs = (f3, 0, 1, 2, 3),
        xaxis = :log)
p2 = plot(sol_robertson, tspan = tspan,
          idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)],
          color = palette(:default)[1:3]', xaxis = :log)
plot!(sol_robertson[2:end], tspan = tspan, idxs = (f3, 0, 1, 2, 3), xaxis = :log)
myplot!(sol_robertson_MPRK43I[2:end], "MPRK43I"; tspan = tspan,
        idxs = [(0, 1), ((x, y) -> (x, 1e4 .* y), 0, 2), (0, 3)], xaxis = :log)
myplot!(sol_robertson_MPRK43I[2:end], "MPRK43I"; tspan = tspan, idxs = (f3, 0, 1, 2, 3),
        markershape = :circle,
        xaxis = :log)
plot(p1, p2, layout = (1, 2))
plot!(legend = :none)

## brusselator problem ####################################################
sol_brusselator = solve(prob_pds_brusselator, Tsit5());
sol_brusselator_MPE = solve(prob_pds_brusselator, MPE(), dt = 0.25);
sol_brusselator_MPRK22 = solve(prob_pds_brusselator, MPRK22(1.0), dt = 0.25,
                               adaptive = false);
sol_brusselator_MPRK43I = solve(prob_pds_brusselator, MPRK43I(1.0, 0.5), dt = 0.25,
                                adaptive = false);
sol_brusselator_MPRK43II = solve(prob_pds_brusselator, MPRK43II(2.0 / 3.0), dt = 0.25,
                                 adaptive = false);

# plot
p1 = plot(sol_brusselator, tspan = (0.0, 5.0), legend = :outerright)
myplot!(sol_brusselator_MPE, "MPE"; tspan = (0.0, 5.0))
plot!(sol_brusselator_MPE, tspan = (0.0, 5.0), idxs = (f_brusselator, 0, 1, 2, 3, 4, 5, 6),
      label = "f_brusselator")
p2 = plot(sol_brusselator, tspan = (0.0, 5.0), legend = :outerright)
myplot!(sol_brusselator_MPRK22, "MPRK22"; tspan = (0.0, 5.0))
plot!(sol_brusselator_MPRK22, tspan = (0.0, 5.0),
      idxs = (f_brusselator, 0, 1, 2, 3, 4, 5, 6),
      label = "f_brusselator")
p3 = plot(sol_brusselator, tspan = (0.0, 5.0), legend = :outerright)
myplot!(sol_brusselator_MPRK43I, "MPRK43I"; tspan = (0.0, 5.0))
plot!(sol_brusselator_MPRK43I, tspan = (0.0, 5.0),
      idxs = (f_brusselator, 0, 1, 2, 3, 4, 5, 6),
      label = "f_brusselator")
plot(p1, p2, p3, layout = (2, 2))
plot!(legend = :none)

# convergence order
test_setup = Dict(:alg => Vern9(), :reltol => 1e-14, :abstol => 1e-14)
sims = convergence_tab_plot(prob_pds_brusselator, [MPE(), Euler()], test_setup;
                            dts = 0.5 .^ (3:15),
                            order_plot = true);
@assert sims[1].𝒪est[:l∞] > 0.9
sims = convergence_tab_plot(prob_pds_brusselator,
                            [MPRK22(1.0), MPRK22(2 / 3), MPRK22(0.5), Heun()],
                            test_setup; dts = 0.5 .^ (5:15),
                            order_plot = true);
for i in 1:4
    @assert sims[i].𝒪est[:l∞] > 1.9
end
## SIR model ##############################################################
sol_sir = solve(prob_pds_sir, Tsit5());
sol_sir_Euler = solve(prob_pds_sir, Euler(), dt = 1.0);
sol_sir_MPE = solve(prob_pds_sir, MPE(), dt = 1.0);
sol_sir_MPRK22 = solve(prob_pds_sir, MPRK22(1.0), dt = 1.0, adaptive = false);
sol_sir_MPRK43I = solve(prob_pds_sir, MPRK43I(1.0, 0.5), dt = 1.0, adaptive = false);
sol_sir_MPRK43II = solve(prob_pds_sir, MPRK43II(0.5), dt = 1.0, adaptive = false);

# plot
p1 = plot(sol_sir)
myplot!(sol_sir_Euler, "Euler")
plot!(sol_sir_Euler, idxs = (f3, 0, 1, 2, 3), label = "f3")
p2 = plot(sol_sir)
myplot!(sol_sir_MPE, "MPE")
plot!(sol_sir_MPE, idxs = (f3, 0, 1, 2, 3), label = "f3")
p3 = plot(sol_sir)
myplot!(sol_sir_MPRK22, "MPRK22")
plot!(sol_sir_MPRK22, idxs = (f3, 0, 1, 2, 3), label = "f3")
p4 = plot(sol_sir)
myplot!(sol_sir_MPRK43I, "MPRK43I")
plot!(sol_sir_MPRK43I, idxs = (f3, 0, 1, 2, 3), label = "f3")
p5 = plot(sol_sir)
myplot!(sol_sir_MPRK43II, "MPRK43II")
plot!(sol_sir_MPRK43II, idxs = (f3, 0, 1, 2, 3), label = "f3")
plot(p1, p2, p3, p4, p5, layout = (3, 2))
plot!(legend = :none)
# convergence order
test_setup = Dict(:alg => Vern9(), :reltol => 1e-14, :abstol => 1e-14)
sims = convergence_tab_plot(prob_pds_sir, [MPE(), Euler()], test_setup; dts = 0.5 .^ (1:15),
                            order_plot = true);
@assert sims[1].𝒪est[:l∞] > 0.9
sims = convergence_tab_plot(prob_pds_sir, [MPRK22(1.0), MPRK22(2 / 3), MPRK22(0.5), Heun()],
                            test_setup; dts = 0.5 .^ (5:15),
                            order_plot = true);
for i in 1:4
    @assert sims[i].𝒪est[:l∞] > 1.9
end
sims = convergence_tab_plot(prob_pds_sir, [MPRK43I(1.0, 0.5), MPRK43II(0.5), Rodas3()],
                            test_setup; dts = 0.5 .^ (5:11),
                            order_plot = true);
for i in 1:3
    @assert sims[i].𝒪est[:l∞] > 2.9
end
## bertolazzi problem #####################################################
sol_bertolazzi = solve(prob_pds_bertolazzi, TRBDF2());
sol_bertolazzi_MPE = solve(prob_pds_bertolazzi, MPE(), dt = 0.01);
sol_bertolazzi_MPRK22 = solve(prob_pds_bertolazzi, MPRK22(1.0), dt = 0.01, abstol = 1e-3,
                              reltol = 1e-3);
sol_bertolazzi_MPRK43I = solve(prob_pds_bertolazzi, MPRK43I(1.0, 0.5), dt = 0.01,
                               abstol = 1e-3, reltol = 1e-3);
sol_bertolazzi_MPRK43II = solve(prob_pds_bertolazzi, MPRK43II(0.5), dt = 0.01,
                                abstol = 1e-3, reltol = 1e-3);

# plot
p1 = plot(sol_bertolazzi, legend = :right)
myplot!(sol_bertolazzi_MPE, "MPE")
ylims!((-0.5, 3.5))
plot!(sol_bertolazzi_MPE, idxs = (f3, 0, 1, 2, 3))
p2 = plot(sol_bertolazzi, legend = :right)
myplot!(sol_bertolazzi_MPRK22, "MPRK22")
ylims!((-0.5, 3.5))
plot!(sol_bertolazzi_MPRK22, idxs = (f3, 0, 1, 2, 3))
p3 = plot(sol_bertolazzi, legend = :right)
myplot!(sol_bertolazzi_MPRK43I, "MPRK43I")
ylims!((-0.5, 3.5))
plot!(sol_bertolazzi_MPRK43I, idxs = (f3, 0, 1, 2, 3))
p4 = plot(sol_bertolazzi, legend = :right)
myplot!(sol_bertolazzi_MPRK43II, "MPRK43II")
ylims!((-0.5, 3.5))
plot!(sol_bertolazzi_MPRK43II, idxs = (f3, 0, 1, 2, 3))
plot(p1, p2, p3, p4, layout = (2, 2))
plot!(legend = :none)

# convergence order
test_setup = Dict(:alg => Rosenbrock23(), :reltol => 1e-8, :abstol => 1e-8)
convergence_tab_plot(prob_pds_bertolazzi, [MPE(), ImplicitEuler()], test_setup;
                     dts = 0.5 .^ (10:15), order_plot = true)

### npzd problem ##########################################################
sol_npzd = solve(prob_pds_npzd, Rosenbrock23());
sol_npzd_MPE = solve(prob_pds_npzd, MPE(), dt = 0.4);
sol_npzd_MPRK22 = solve(prob_pds_npzd, MPRK22(1.0), dt = 0.4, adaptive = false);
sol_npzd_MPRK43I = solve(prob_pds_npzd, MPRK43I(1.0, 0.5), dt = 0.4, adaptive = false);
sol_npzd_MPRK43II = solve(prob_pds_npzd, MPRK43II(0.5), dt = 0.4, adaptive = false);

# plot
p1 = plot(sol_npzd)
myplot!(sol_npzd_MPE, "MPE")
plot!(sol_npzd_MPE, idxs = (f_npzd, 0, 1, 2, 3, 4), label = "f_npzd")
p2 = plot(sol_npzd)
myplot!(sol_npzd_MPRK22, "MPRK22")
plot!(sol_npzd_MPRK22, idxs = (f_npzd, 0, 1, 2, 3, 4), label = "f_npzd")
p3 = plot(sol_npzd)
myplot!(sol_npzd_MPRK43I, "MPRK43I")
plot!(sol_npzd_MPRK43I, idxs = (f_npzd, 0, 1, 2, 3, 4), label = "f_npzd")
p4 = plot(sol_npzd)
myplot!(sol_npzd_MPRK43II, "MPRK43II")
plot!(sol_npzd_MPRK43II, idxs = (f_npzd, 0, 1, 2, 3, 4), label = "f_npzd")
plot(p1, p2, p3, p4, layout = (2, 2))
plot!(legend = :none)

# convergence order
# error should take all time steps into account, not only the final time!
test_setup = Dict(:alg => Rodas3(), :reltol => 1e-14, :abstol => 1e-14)
sims = convergence_tab_plot(prob_pds_npzd, [MPE(), ImplicitEuler()], test_setup;
                            dts = 0.5 .^ (5:17), order_plot = true);
@assert sims[1].𝒪est[:l∞] > 0.9
sims = convergence_tab_plot(prob_pds_npzd,
                            [MPRK22(1.0), MPRK22(2 / 3), MPRK22(0.5), Heun()], test_setup;
                            dts = 0.5 .^ (10:17), order_plot = true);
for i in 1:4
    @assert sims[i].𝒪est[:l∞] > 1.9
end

sims = convergence_tab_plot(prob_pds_npzd,
                            [MPRK43I(1.0, 0.5), MPRK43II(0.5), Rodas3()], test_setup;
                            dts = 0.5 .^ (12:17), order_plot = true);
for i in 1:2
    @assert sims[i].𝒪est[:l∞] > 2.9
end
### stratospheric reaction problem ####################################################

sol_stratreac = solve(prob_pds_stratreac, TRBDF2(autodiff = false));
# currently no solver for non-conservative PDS implemented

tspan = prob_pds_stratreac.tspan
u0 = prob_pds_stratreac.u0
linear_invariant_1 = u0[1] + u0[2] + 3 * u0[3] + 2 * u0[4] + u0[5] + 2 * u0[6]
linear_invariant_2 = u0[5] + u0[6]

function g1(t, u1, u2, u3, u4, u5, u6)
    (t,
     abs(u1 + u2 + 3 * u3 + 2 * u4 + u5 + 2 * u6 - linear_invariant_1) / linear_invariant_1)
end
g2(t, u1, u2, u3, u4, u5, u6) = (t, abs(u5 + u6 - linear_invariant_2) / linear_invariant_2)

p1 = plot(sol_stratreac, idxs = (0, 1), xticks = [tspan[1], tspan[2]], legend = :outertop)
#plot!(sol_stratreac_MPE.t, tmp[1, :])
ylims!((-10, 110))

p2 = plot(sol_stratreac, idxs = (0, 2), xticks = [tspan[1], tspan[2]], legend = :outertop)
#plot!(sol_stratreac_MPE.t, tmp[2, :])
ylims!((-1e8, 8e8))

p3 = plot(sol_stratreac, idxs = (0, 3), xticks = [tspan[1], tspan[2]], legend = :outertop)
#plot!(sol_stratreac_MPE.t, tmp[3, :])
ylims!((2e11, 6e11))

p4 = plot(sol_stratreac, idxs = (0, 4), xticks = [tspan[1], tspan[2]], legend = :outertop)
#plot!(sol_stratreac_MPE.t, tmp[4, :])
ylims!((1.69698e16, 1.69705e16))

p5 = plot(sol_stratreac, idxs = (0, 5), xticks = [tspan[1], tspan[2]], legend = :outertop)
#plot!(sol_stratreac_MPE.t, tmp[5, :])
ylims!((-5e6, 15e6))

p6 = plot(sol_stratreac, idxs = (0, 6), xticks = [tspan[1], tspan[2]], legend = :outertop)
#plot!(sol_stratreac_MPE.t, tmp[6, :])
ylims!((1.08e9, 1.1e9))

p7 = plot(sol_stratreac, idxs = (g1, 0, 1, 2, 3, 4, 5, 6), xticks = [tspan[1], tspan[2]],
          legend = :outertop)
p8 = plot(sol_stratreac, idxs = (g2, 0, 1, 2, 3, 4, 5, 6),
          xticks = [tspan[1], tspan[2]], legend = :outertop)

plot(p1, p2, p3, p4, p5, p6, p7, p8)
