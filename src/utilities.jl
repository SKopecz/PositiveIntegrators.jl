
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

# Taken from Sandu (2001)
function l2_error(sol, sol_ref)
    sqrt(sum(((sol .- sol_ref) ./ sol_ref) .^ 2) / length(sol_ref))
end

function l1_error(sol, sol_ref)
    sqrt(sum(((sol .- sol_ref) ./ sol_ref) .^ 2) / length(sol_ref))
end

function l∞_error(sol, sol_ref)
    maximum(abs.((sol .- sol_ref) ./ sol_ref))
end

function workprecision_fixed!(dict, prob, algs, names, sol_ref, dts;
                              compute_error = l∞_error, seconds = 2,
                              numruns = 20)
    for (alg, name) in zip(algs, names)
        println(name)
        error_time = Vector{Tuple{Float64, Float64}}(undef, length(dts))

        for (i, dt) in enumerate(dts)
            sol = solve(prob, alg; dt, adaptive = false, save_everystep = false)
            error = compute_error(sol.u[end], sol_ref)

            # using bechmarktools is too slow to 
            #time = @belapsed solve($prob, $alg, dt = $dt, adaptive = false, save_everystep = false)

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
            error_time[i] = (error, time)
        end
        dict[name] = error_time
    end
end

function workprecision_fixed(prob, algs, names, sol_ref, dts; compute_error = l∞_error,
                             seconds = 2, numruns = 20)
    dict = Dict(name => [] for name in names)
    workprecision_fixed!(dict, prob, algs, names, sol_ref, dts; compute_error, seconds,
                         numruns)
    return dict
end

function workprecision_adaptive!(dict, prob, algs, names, sol_ref, abstols, reltols;
                                 compute_error = l∞_error,
                                 seconds = 2, numruns = 20, kwargs...)
    for (alg, name) in zip(algs, names)
        println(name)
        error_time = Vector{Tuple{Float64, Float64}}(undef, length(abstols))

        for (i, dt) in enumerate(abstols)
            abstol = abstols[i]
            reltol = reltols[i]
            sol = solve(prob, alg; dt, abstol, reltol, save_everystep = false, kwargs...)
            error = compute_error(sol.u[end], sol_ref)

            #time = @belapsed solve($prob, $alg, dt = $dt, adaptive = false, save_everystep = false)
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

            error_time[i] = (error, time)
        end
        dict[name] = error_time
    end
    return nothing
end

function workprecision_adaptive(prob, algs, names, sol_ref, abstols, reltols;
                                compute_error = l∞_error, seconds = 2,
                                numruns = 20, kwargs...)
    dict = Dict(name => [] for name in names)
    workprecision_adaptive!(dict, prob, algs, names, sol_ref, abstols, reltols;
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
