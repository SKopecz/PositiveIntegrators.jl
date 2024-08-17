using Plots
using DiffEqDevTools: test_convergence, analyticless_test_convergence
using PrettyTables: pretty_table

function convergence_tab_plot(prob, algs, test_setup = nothing; dts = 0.5 .^ (1:10),
                              order_plot = false, analytic = false)
    sims = Array{Any}(undef, length(algs))
    for i in eachindex(algs)
        #convergence order
        if analytic
            sim = test_convergence(dts, prob, algs[i])
        else
            sim = analyticless_test_convergence(dts, prob, algs[i], test_setup)
        end
        sims[i] = sim
        err = sim.errors[:l∞]
        p = -log2.(err[2:end] ./ err[1:(end - 1)])
        #table
        algname = string(Base.typename(typeof(algs[i])).wrapper)
        if algname == "MPRK22"
            my_matches = match(r"(?<= alpha = )([0-9][.][0-9]*)", string(algs[i]))
            algname = algname .* "_" .* my_matches.match
        end
        #algname = string(algs[i])
        pretty_table([dts err [NaN; p]]; header = (["dt", "err", "p"]),
                     title = string("\n\n", string(algs[i])), title_alignment = :c,
                     title_autowrap = true, title_same_width_as_table = true)
        #plot
        if order_plot
            pop!(sim.errors, :final)
            # analyticless_test_convergence computes additional errors L2 and L∞.
            if !analytic
                pop!(sim.errors, :L2)
                pop!(sim.errors, :L∞)
            end
            label = algname .* " " .* ["l∞" "l2"]
            if i == 1
                plot(sim; label = label)
            else
                plot!(sim; label = label)
            end
            plot!(legend = :outertop)
        end
    end
    if order_plot
        display(current())
    end
    return sims
end

function myplot(sol, name = "", analytic = false; kwargs...)
    _myplot(plot, sol, name, analytic; kwargs...)
end
function myplot!(sol, name = "", analytic = false; kwargs...)
    _myplot(plot!, sol, name, analytic; kwargs...)
end

function _myplot(plotf, sol, name = "", analytic = false; kwargs...)
    N = length(sol.u[1])
    if analytic == true
        plotf(sol, color = palette(:default)[1:(2 * N)]',
              plot_analytic = true; kwargs...)
    else
        plotf(sol, color = palette(:default)[1:N]', plot_analytic = false; kwargs...)
    end
    p = plot!(sol, color = palette(:default)[1:N]', denseplot = false,
              markershape = :circle,
              markersize = 3, #markerstrokecolor = palette(:default)[1:N]',
              linecolor = invisible(), label = ""; kwargs...)
    title!(name)
    return p
end
