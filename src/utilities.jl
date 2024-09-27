
function isnegative(u::AbstractVector, args...)
    return any(<(0), u)
end

function isnegative(u::AbstractVector{<:AbstractVector}, args...)
    anyisnegative = any(isnegative, u)
    if anyisnegative
        components = Int64[]
        minima = eltype(first(u))[]
        # look for components with negative elements
        for i in eachindex(first(u))
            v = getindex.(u, i)
            if isnegative(v)
                push!(components, i)
                push!(minima, minimum(v))
            end
        end
        if length(components) == 1
            println("Component ", components[1],
                    " contains negative elements, the minimum is ", minima[1]), "."
        else
            println("Components ", components,
                    " contain negative elements, the respective minima are ", minima, ".")
        end
    end
    return anyisnegative
end

"""
    isnegative(sol::ODESolution)

Returns `true` if `sol` contains negative elements.

See also [`isnonnegative`](@ref).
"""
function isnegative(sol::ODESolution, args...)
    return isnegative(sol.u, args...)
end

"""
    isnonnegative(u)

Negation of [`isnegative`](@ref).    
"""
isnonnegative(args...) = !isnegative(args...)

### Work-precision #########################################################################

# The following function take sol.u and sol_ref.u as inputs
# relative errors
function rel_l2_error_at_end(sol, sol_ref)
    sqrt(sum(((sol[end] .- sol_ref[end]) ./ sol_ref[end]) .^ 2) / length(sol_ref[end]))
end

function rel_l1_error_at_end(sol, sol_ref)
    sum(abs.((sol[end] .- sol_ref[end]) ./ sol_ref[end])) / length(sol_ref[end])
end

function rel_l∞_error_at_end(sol, sol_ref)
    maximum(abs.((sol[end] .- sol_ref[end]) ./ sol_ref[end]))
end

function rel_l∞_error_all(sol, sol_ref)
    err = zero(eltype(eltype(sol)))
    for i in eachindex(sol)
        max_err_i = maximum(abs.((abs.(sol[i]) .- abs.(sol_ref[i])) ./ sol_ref[i]))
        if max_err_i > err
            err = max_err_i
        end
    end
    return err
end

function compute_time_fixed(dt, prob, alg, seconds, numruns)
    # using bechmarktools is too slow 
    # time = @belapsed solve($prob, $alg, dt = $dt, adaptive = false, save_everystep = false)

    ### adapted from DiffEqDevTools.jl/src/benchmark.jl#L84 ##################
    benchmark_f = let dt = dt, prob = prob, alg = alg
        () -> @elapsed solve(prob, alg; dt, adaptive = false,
                             save_everystep = false)
    end

    benchmark_f() # pre-compile

    b_t = benchmark_f()
    if b_t > seconds
        time = b_t
    else
        time = mapreduce(i -> benchmark_f(), min, 2:numruns; init = b_t)
    end
    ##########################################################################
    return time
end

function compute_time_adaptive(abstol, reltol, prob, alg, seconds, numruns, kwargs...)
    # using bechmarktools is too slow 
    # time = @belapsed solve($prob, $alg, dt = $dt, adaptive = false, save_everystep = false)
    ### adapted from DiffEqDevTools.jl/src/benchmark.jl#L84 ##################
    benchmark_f = let abstol = abstol, reltol = reltol, prob = prob, alg = alg,
        kwargs = kwargs

        () -> @elapsed solve(prob, alg; abstol, reltol, adaptive = true,
                             save_everystep = false, kwargs...)
    end

    benchmark_f() # pre-compile

    b_t = benchmark_f()
    if b_t > seconds
        time = b_t
    else
        time = mapreduce(i -> benchmark_f(), min, 2:numruns; init = b_t)
    end
    ##########################################################################
    return time
end

# functions to compute data for workprecision diagrams
function workprecision_fixed!(dict, prob, algs, names, dts, alg_ref;
                              compute_error = rel_l∞_error_at_end, seconds = 2,
                              numruns = 20)
    tspan = prob.tspan
    dt_ref = (last(tspan) - first(tspan)) ./ 1e5
    sol_ref = solve(prob, alg_ref; dt = dt_ref, adaptive = false, save_everystep = true)

    let sol_ref = sol_ref
        for (alg, name) in zip(algs, names)
            println(name)
            error_time = Vector{Tuple{Float64, Float64}}(undef, length(dts))

            for (i, dt) in enumerate(dts)
                error_time[i] = (Inf, Inf)
                try
                    sol = solve(prob, alg; dt, adaptive = false, save_everystep = true)
                    if Int(sol.retcode) == 1 && isnonnegative(sol)
                        error = compute_error(sol.u, sol_ref(sol.t))
                        time = compute_time_fixed(dt, prob, alg, seconds, numruns)

                        error_time[i] = (error, time)
                    else
                        error_time[i] = (Inf, Inf)
                    end
                catch e
                end
            end
            dict[name] = error_time
        end
    end
end

function workprecision_fixed(prob, algs, names, dts, alg_ref;
                             compute_error = rel_l∞_error_at_end,
                             seconds = 2, numruns = 20)
    dict = Dict(name => [] for name in names)
    workprecision_fixed!(dict, prob, algs, names, dts, alg_ref; compute_error, seconds,
                         numruns)
    return dict
end

function workprecision_adaptive!(dict, prob, algs, names, abstols, reltols, alg_ref;
                                 adaptive_ref = false,
                                 abstol_ref = 1e-14, reltol_ref = 1e-13,
                                 compute_error = rel_l∞_error_at_end,
                                 seconds = 2, numruns = 20, kwargs...)
    if adaptive_ref
        sol_ref = solve(prob, alg_ref; adaptive = true, save_everystep = true,
                        abstol = abstol_ref, reltol = reltol_ref)
    else
        tspan = prob.tspan
        dt_ref = (last(tspan) - first(tspan)) ./ 1e5
        sol_ref = solve(prob, alg_ref; dt = dt_ref, adaptive = false, save_everystep = true)
    end

    for (alg, name) in zip(algs, names)
        println(name)
        error_time = Vector{Tuple{Float64, Float64}}(undef, length(abstols))

        for (i, dt) in enumerate(abstols)
            abstol = abstols[i]
            reltol = reltols[i]
            sol = solve(prob, alg; abstol, reltol, save_everystep = true,
                        kwargs...)

            if Int(sol.retcode) == 1 && isnonnegative(sol)
                error = compute_error(sol.u, sol_ref(sol.t))
                time = compute_time_adaptive(abstol, reltol, prob, alg, seconds, numruns,
                                             kwargs...)

                error_time[i] = (error, time)
            else
                error_time[i] = (Inf, Inf)
            end
        end
        dict[name] = error_time
    end

    return nothing
end

function workprecision_adaptive(prob, algs, names, abstols, reltols, alg_ref;
                                adaptive_ref = false,
                                abstol_ref = 1e-14, reltol_ref = 1e-13,
                                compute_error = rel_l∞_error_at_end, seconds = 2,
                                numruns = 20, kwargs...)
    dict = Dict(name => [] for name in names)
    workprecision_adaptive!(dict, prob, algs, names, abstols, reltols, alg_ref;
                            adaptive_ref, abstol_ref, reltol_ref,
                            compute_error, seconds,
                            numruns, kwargs...)
    return dict
end

@recipe function f(wp::Dict, names; color = nothing)
    seriestype --> :path
    linewidth --> 3
    xscale --> :log10
    yscale --> :log10
    markershape --> :auto
    xs = [first.(wp[name]) for name in names]
    ys = [last.(wp[name]) for name in names]
    xguide --> "error"
    yguide --> "times (s)"
    label --> reshape(names, 1, length(wp))
    return xs, ys
end
