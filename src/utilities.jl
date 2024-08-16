
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
