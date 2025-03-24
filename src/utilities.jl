
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

Returns `true` if `sol.u` contains negative elements.

Please note that negative values may occur when plotting the solution, depending on the interpolation used.

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

### Errors #########################################################################
"""
    rel_max_error_tend(sol, ref_sol)

Returns the relative maximum error between `sol` and `ref_sol` at time `sol.t[end]`.
"""
function rel_max_error_tend(sol::AbstractVector, ref_sol::AbstractVector)
    return maximum(abs.((sol[end] .- ref_sol[end]) ./ ref_sol[end]))
end

rel_max_error_tend(sol, ref_sol) = rel_max_error_tend(sol.u, ref_sol.u)

"""
    rel_max_error_overall(sol, ref_sol)

Returns the maximum of the relative maximum errors between `sol` and `ref_sol` over all time steps.
"""
function rel_max_error_overall(sol::AbstractVector, ref_sol::AbstractVector)
    err = zero(eltype(eltype(sol)))
    for i in eachindex(sol)
        max_err_i = maximum(abs.((abs.(sol[i]) .- abs.(ref_sol[i])) ./ ref_sol[i]))
        if max_err_i > err
            err = max_err_i
        end
    end
    return err
end

rel_max_error_overall(sol, ref_sol) = rel_max_error_overall(sol.u, ref_sol.u)

"""
    rel_l1_error_tend(sol, ref_sol)

Returns the relative l1 error between `sol` and `ref_sol` at time `sol.t[end]`.
"""
function rel_l1_error_tend(sol::AbstractVector, ref_sol::AbstractVector)
    return sum(abs.((sol[end] .- ref_sol[end]) ./ ref_sol[end])) / length(ref_sol[end])
end

rel_l1_error_tend(sol, ref_sol) = rel_l1_error_tend(sol.u, ref_sol.u)

"""
    rel_l2_error_tend(sol, ref_sol)

Returns the relative l2 error between `sol` and `ref_sol` at time `sol.t[end]`.
"""
function rel_l2_error_tend(sol::AbstractVector, ref_sol::AbstractVector)
    return sqrt(sum(abs2.((sol[end] .- ref_sol[end]) ./ ref_sol[end])) /
                length(ref_sol[end]))
end

rel_l2_error_tend(sol, ref_sol) = rel_l2_error_tend(sol.u, ref_sol.u)

### Functions to compute work-precision diagrams ##########################################
function _compute_time(benchmark_f, seconds, numruns)
    benchmark_f() # pre-compile

    time = benchmark_f()

    if time ≤ seconds
        time = median([time; [benchmark_f() for i in 2:numruns]])
    end

    return time
end

function compute_time_fixed(dt, prob, alg, seconds, numruns)
    benchmark_f = let dt = dt, prob = prob, alg = alg
        () -> @elapsed solve(prob, alg; dt, adaptive = false,
                             save_everystep = false)
    end

    return _compute_time(benchmark_f, seconds, numruns)
end

function compute_time_adaptive(abstol, reltol, prob, alg, seconds, numruns, kwargs...)
    benchmark_f = let abstol = abstol, reltol = reltol, prob = prob, alg = alg,
        kwargs = kwargs

        () -> @elapsed solve(prob, alg; abstol, reltol, adaptive = true,
                             save_everystep = false, kwargs...)
    end

    return _compute_time(benchmark_f, seconds, numruns)
end

"""
    work_precision_fixed!(dict, prob, algs, labels, dts, alg_ref;
                          compute_error = rel_max_error_tend,
                          seconds = 2,
                          numruns = 20)
    )

Adds work-precision data to the dictionary `dict`, which was created with `work_precion_fixed`.
See [`work_precision_fixed`](@ref) for the meaning of the inputs.
"""
function work_precision_fixed!(dict, prob, algs, labels, dts, alg_ref;
                               compute_error = rel_max_error_tend, seconds = 2,
                               numruns = 20)
    tspan = prob.tspan
    dt_ref = (last(tspan) - first(tspan)) ./ 1e5
    ref_sol = solve(prob, alg_ref; dt = dt_ref, adaptive = false, save_everystep = true)

    let ref_sol = ref_sol
        for (alg, label) in zip(algs, labels)
            println(label)
            error_time = Vector{Tuple{Float64, Float64}}(undef, length(dts))

            for (i, dt) in enumerate(dts)
                error_time[i] = (Inf, Inf)
                try
                    sol = solve(prob, alg; dt, adaptive = false, save_everystep = true)
                    if Int(sol.retcode) == 1 && isnonnegative(sol)
                        error = compute_error(sol, ref_sol(sol.t))
                        time = compute_time_fixed(dt, prob, alg, seconds, numruns)

                        error_time[i] = (error, time)
                    else
                        error_time[i] = (Inf, Inf)
                    end
                catch e
                end
            end
            dict[label] = error_time
        end
    end
end

"""
    work_precision_fixed(prob, algs, labels, dts, alg_ref;
                         compute_error = rel_max_error_tend,
                         seconds = 2,
                         numruns = 20)

Returns a dictionary to create work-precision diagrams.
The problem `prob` is solved by each algorithm in `algs` for all the step sizes defined in `dts`.
For each step size the error and computing time are stored in the dictionary.
If the solve is not successful for a given step size, then `(Inf, Inf)` is stored in the dictionary.
The strings in the array `labels` are used as keys of the dictionary.
The reference solution used for error computations is computed with the algorithm `alg_ref`.

### Keyword arguments: ###

- `compute_error(sol::ODESolution, ref_sol::ODESolution)`: Function to compute the error between `sol` and `ref_sol`.
- `seconds`: If the measured computing time of a single solve is larger than `seconds`, then this computing time is stored in the dictionary.
- `numruns`: If the measured computing time of a single solve is less than or equal to `seconds`, then `numruns` solves are performed and the median of the respective computing times is stored in the dictionary.
"""
function work_precision_fixed(prob, algs, labels, dts, alg_ref;
                              compute_error = rel_max_error_tend,
                              seconds = 2, numruns = 20)
    dict = Dict(label => [] for label in labels)
    work_precision_fixed!(dict, prob, algs, labels, dts, alg_ref; compute_error, seconds,
                          numruns)
    return dict
end

"""
    work_precision_adaptive(prob, algs, labels, abstols, reltols, alg_ref;
                            adaptive_ref = false,
                            abstol_ref = 1e-14,
                            reltol_ref = 1e-13,
                            compute_error = rel_max_error_tend,
                            seconds = 2,
                            numruns = 20,
                            kwargs...)

Adds work-precision data to the dictionary `dict`, which was created with `work_precion_fixed_adaptive`.
See [`work_precision_adaptive`](@ref) for the meaning of the inputs.
"""
function work_precision_adaptive!(dict, prob, algs, labels, abstols, reltols, alg_ref;
                                  adaptive_ref = false,
                                  abstol_ref = 1e-14, reltol_ref = 1e-13,
                                  compute_error = rel_max_error_tend,
                                  seconds = 2, numruns = 20, kwargs...)
    if adaptive_ref
        ref_sol = solve(prob, alg_ref; adaptive = true, save_everystep = true,
                        abstol = abstol_ref, reltol = reltol_ref)
    else
        tspan = prob.tspan
        dt_ref = (last(tspan) - first(tspan)) ./ 1e5
        ref_sol = solve(prob, alg_ref; dt = dt_ref, adaptive = false, save_everystep = true)
    end

    for (alg, label) in zip(algs, labels)
        println(label)
        error_time = Vector{Tuple{Float64, Float64}}(undef, length(abstols))

        for (i, dt) in enumerate(abstols)
            abstol = abstols[i]
            reltol = reltols[i]
            sol = solve(prob, alg; abstol, reltol, save_everystep = true,
                        kwargs...)

            if Int(sol.retcode) == 1 && isnonnegative(sol)
                error = compute_error(sol, ref_sol(sol.t))
                time = compute_time_adaptive(abstol, reltol, prob, alg, seconds, numruns,
                                             kwargs...)

                error_time[i] = (error, time)
            else
                error_time[i] = (Inf, Inf)
            end
        end
        dict[label] = error_time
    end

    return nothing
end

"""
    work_precision_adaptive(prob, algs, labels, abstols, reltols, alg_ref;
                            adaptive_ref = false,
                            abstol_ref = 1e-14,
                            reltol_ref = 1e-13,
                            compute_error = rel_max_error_tend,
                            seconds = 2,
                            numruns = 20,
                            kwargs...)

Returns a dictionary to create work-precision diagrams.
The problem `prob` is solved by each algorithm in `algs` for all tolerances defined in `abstols` and `reltols`.
For the respective tolerances the error and computing time are stored in the dictionary.
If the solve is not successful for the given tolerances, then `(Inf, Inf)` is stored in the dictionary.
The strings in the array `labels` are used as keys of the dictionary.
The reference solution used for error computations is computed with the algorithm `alg_ref`.
Additional keyword arguments are passed on to `solve`.

### Keyword arguments: ###

- `adaptive_ref`: If `true` the refenerce solution is computed adaptively with tolerances `abstol_ref` and `reltol_ref`. Otherwise ``10^5`` steps are used.
- `abstol_ref`: See `adaptive_ref`.
- `reltol_ref`: See `adaptive_ref`.
- `compute_error(sol::ODESolution, ref_sol::ODESolution)`: A function to compute the error between `sol` and `ref_sol`.
- `seconds`: If the measured computing time of a single solve is larger than `seconds`, then this computing time is stored in the dictionary.
- `numruns`: If the measured computing time of a single solve is less than or equal to `seconds`, then `numruns` solves are performed and the median of the respective computing times is stored in the dictionary.
"""
function work_precision_adaptive(prob, algs, labels, abstols, reltols, alg_ref;
                                 adaptive_ref = false,
                                 abstol_ref = 1e-14, reltol_ref = 1e-13,
                                 compute_error = rel_max_error_tend, seconds = 2,
                                 numruns = 20, kwargs...)
    dict = Dict(label => [] for label in labels)
    work_precision_adaptive!(dict, prob, algs, labels, abstols, reltols, alg_ref;
                             adaptive_ref, abstol_ref, reltol_ref,
                             compute_error, seconds,
                             numruns, kwargs...)
    return dict
end

# plot recipe to plot work-precision dictionaries
@recipe function f(wp::Dict, labels; color = nothing)
    seriestype --> :path
    linewidth --> 3
    xscale --> :log10
    yscale --> :log10
    markershape --> :auto
    xs = [first.(wp[label]) for label in labels]
    ys = [last.(wp[label]) for label in labels]
    xguide --> "error"
    yguide --> "times (s)"
    label --> reshape(labels, 1, length(wp))
    return xs, ys
end
