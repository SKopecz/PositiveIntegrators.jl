# New problem type PDSProblem
abstract type AbstractPDSProblem end

"""
    PDSProblem(P, D, u0, tspan, p = NullParameters();
               p_prototype = nothing,
               analytic = nothing,
               std_rhs = nothing)

A structure describing a system of ordinary differential equations in form of a production-destruction system (PDS).
`P` denotes the function defining the production matrix ``P``.
The diagonal of ``P`` contains production terms without destruction counterparts.
`D` is the function defining the vector of destruction terms ``D`` without production counterparts.
`u0` is the vector of initial conditions and `tspan` the time span
`(t_initial, t_final)` of the problem. The optional argument `p` can be used
to pass additional parameters to the functions `P` and `D`.

The functions `P` and `D` can be used either in the out-of-place form with signature
`production_terms = P(u, p, t)` or the in-place form `P(production_terms, u, p, t)`.

### Keyword arguments: ###

- `p_prototype`: If `P` is given in in-place form, `p_prototype` or copies thereof are used to store evaluations of `P`.
  If `p_prototype` is not specified explicitly and `P` is in-place, then `p_prototype` will be internally
  set to `zeros(eltype(u0), (length(u0), length(u0)))`.
- `analytic`: The analytic solution of a PDS must be given in the form `f(u0,p,t)`.
  Specifying the analytic solution can be useful for plotting and convergence tests.
- `std_rhs`: The standard ODE right-hand side evaluation function callable
  as `du = std_rhs(u, p, t)` for the out-of-place form and
  as `std_rhs(du, u, p, t)` for the in-place form. Solvers that do not rely on
  the production-destruction representation of the ODE, will use this function
  instead to compute the solution. If not specified,
  a default implementation calling `P` and `D` is used.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
"""
struct PDSProblem{iip} <: AbstractPDSProblem end

# New ODE function PDSFunction
struct PDSFunction{iip, specialize, P, D, PrototypeP, PrototypeD, StdRHS, Ta} <:
       AbstractODEFunction{iip}
    p::P
    d::D
    p_prototype::PrototypeP
    d_prototype::PrototypeD
    std_rhs::StdRHS
    analytic::Ta
end

# define behavior of PDSFunctions for non-existing fields
function Base.getproperty(obj::PDSFunction, sym::Symbol)
    if sym === :mass_matrix
        return I
    elseif sym === :jac_prototype
        return nothing
    elseif sym === :colorvec
        return nothing
    elseif sym === :sparsity
        return nothing
    elseif sym === :sys
        return SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing}(nothing,
                                                                                nothing,
                                                                                nothing)
    else # fallback to getfield
        return getfield(obj, sym)
    end
end

# Most general constructor for PDSProblems
function PDSProblem(P, D, u0, tspan, p = NullParameters();
                    kwargs...)
    Piip = isinplace(P, 4)
    Diip = isinplace(D, 4)
    if Piip == Diip
        iip = Piip
    else
        error("Conflict due to the joint use of in-place and out-of-place functions.")
    end
    return PDSProblem{iip}(P, D, u0, tspan, p; kwargs...)
end

# Specialized constructor for PDSProblems setting `iip` manually
# (arbitrary functions)
function PDSProblem{iip}(P, D, u0, tspan, p = NullParameters();
                         p_prototype = nothing,
                         analytic = nothing,
                         std_rhs = nothing,
                         kwargs...) where {iip}

    # p_prototype is used to store evaluations of P, if P is in-place.
    if isnothing(p_prototype) && iip
        p_prototype = zeros(eltype(u0), (length(u0), length(u0)))
    end
    # If a PDSFunction is to be evaluated and D is in-place, then d_prototype is used to store
    # evaluations of D.
    d_prototype = similar(u0)

    PD = PDSFunction{iip}(P, D; p_prototype, d_prototype,
                          analytic, std_rhs)
    PDSProblem{iip}(PD, u0, tspan, p; kwargs...)
end

# Specialized constructor for PDSProblems setting `iip` manually
# (PDSFunction)
function PDSProblem{iip}(PD::PDSFunction, u0, tspan, p = NullParameters();
                         kwargs...) where {iip}
    ODEProblem(PD, u0, tspan, p, PDSProblem{iip}(); kwargs...)
end

# Specialized constructor for PDSFunction setting `iip` manually
function PDSFunction{iip}(P, D; kwargs...) where {iip}
    PDSFunction{iip, FullSpecialize}(P, D; kwargs...)
end

# Most specific constructor for PDSFunction
function PDSFunction{iip, FullSpecialize}(P, D;
                                          p_prototype = nothing,
                                          d_prototype = nothing,
                                          analytic = nothing,
                                          std_rhs = nothing) where {iip}
    if std_rhs === nothing
        std_rhs = PDSStdRHS(P, D, p_prototype, d_prototype)
    end
    PDSFunction{iip, FullSpecialize, typeof(P), typeof(D), typeof(p_prototype),
                typeof(d_prototype),
                typeof(std_rhs), typeof(analytic)}(P, D, p_prototype, d_prototype, std_rhs,
                                                   analytic)
end

# Evaluation of a PDSFunction
function (PD::PDSFunction)(u, p, t)
    return PD.std_rhs(u, p, t)
end

function (PD::PDSFunction)(du, u, p, t)
    return PD.std_rhs(du, u, p, t)
end

# Default implementation of the standard right-hand side evaluation function
struct PDSStdRHS{P, D, PrototypeP, PrototypeD} <: Function
    p::P
    d::D
    p_prototype::PrototypeP
    d_prototype::PrototypeD
end

# Evaluation of a PDSStdRHS (out-of-place)
function (PD::PDSStdRHS)(u, p, t)
    P = PD.p(u, p, t)
    D = PD.d(u, p, t)
    diag(P) + vec(sum(P, dims = 2)) -
    vec(sum(P, dims = 1)) - vec(D)
end

# Evaluation of a PDSStdRHS (in-place)
function (PD::PDSStdRHS)(du, u, p, t)
    PD.p(PD.p_prototype, u, p, t)

    if PD.p_prototype isa AbstractSparseMatrix
        # row sum coded as matrix-vector product 
        fill!(PD.d_prototype, one(eltype(PD.tmp)))
        mul!(vec(du), PD.p_prototype, PD.tmp)

        for i in 1:length(u)  #vec(du) .+= diag(PD.p_prototype)
            du[i] += PD.p_prototype[i, i]
        end
        sum!(PD.d_prototype', PD.p_prototype)
        vec(du) .-= PD.d_prototype
        PD.d(PD.d_prototype, u, p, t)
        vec(du) .-= PD.d_prototype
    else
        PD.d(PD.d_prototype, u, p, t)
        # This implementation does not need any auxiliary vectors
        for i in 1:length(u)
            du[i] = PD.p_prototype[i, i] - PD.d_prototype[i]
            for j in 1:length(u)
                du[i] += PD.p_prototype[i, j] - PD.p_prototype[j, i]
            end
        end
    end
    return nothing
end

# New problem type ConservativePDSProblem
"""
    ConservativePDSProblem(P, u0, tspan, p = NullParameters();
                           p_prototype = nothing,
                           analytic = nothing,
                           std_rhs = nothing)

A structure describing a conservative system of ordinary differential equation in form of a production-destruction system (PDS).
`P` denotes the function defining the production matrix ``P``.
The diagonal of ``P`` contains production terms without destruction counterparts.
`u0` is the vector of initial conditions and `tspan` the time span
`(t_initial, t_final)` of the problem. The optional argument `p` can be used
to pass additional parameters to the function `P`.

The function `P` can be given either in the out-of-place form with signature
`production_terms = P(u, p, t)` or the in-place form `P(production_terms, u, p, t)`.

### Keyword arguments: ###

- `p_prototype`: If `P` is given in in-place form, `p_prototype` or copies thereof are used to store evaluations of `P`.
  If `p_prototype` is not specified explicitly and `P` is in-place, then `p_prototype` will be internally
  set to `zeros(eltype(u0), (length(u0), length(u0)))`.
- `analytic`: The analytic solution of a PDS must be given in the form `f(u0,p,t)`.
  Specifying the analytic solution can be useful for plotting and convergence tests.
- `std_rhs`: The standard ODE right-hand side evaluation function callable
  as `du = std_rhs(u, p, t)` for the out-of-place form and
  as `std_rhs(du, u, p, t)` for the in-place form. Solvers that do not rely on
  the production-destruction representation of the ODE, will use this function
  instead to compute the solution. If not specified,
  a default implementation calling `P` is used.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
"""
struct ConservativePDSProblem{iip} <: AbstractPDSProblem end

# New ODE function ConservativePDSFunction
struct ConservativePDSFunction{iip, specialize, P, PrototypeP, StdRHS, Ta} <:
       AbstractODEFunction{iip}
    p::P # production terms
    p_prototype::PrototypeP # prototype for production terms
    std_rhs::StdRHS # standard right-hand side evaluation function
    analytic::Ta # analytic solution (or nothing)
end

# define behavior of ConservativePDSFunction for non-existing fields
function Base.getproperty(obj::ConservativePDSFunction, sym::Symbol)
    if sym === :mass_matrix
        return I
    elseif sym === :jac_prototype
        return nothing
    elseif sym === :colorvec
        return nothing
    elseif sym === :sparsity
        return nothing
    elseif sym === :sys
        return SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing}(nothing,
                                                                                nothing,
                                                                                nothing)
    else # fallback to getfield
        return getfield(obj, sym)
    end
end

# Most general constructor for ConservativePDSProblems
function ConservativePDSProblem(P, u0, tspan, p = NullParameters();
                                kwargs...)
    iip = isinplace(P, 4)
    return ConservativePDSProblem{iip}(P, u0, tspan, p; kwargs...)
end

# Specialized constructor for ConservativePDSProblems setting `iip` manually
# (arbitrary function)
function ConservativePDSProblem{iip}(P, u0, tspan, p = NullParameters();
                                     p_prototype = nothing,
                                     analytic = nothing,
                                     std_rhs = nothing,
                                     kwargs...) where {iip}

    # p_prototype is used to store evaluations of P, if P is in-place.
    if isnothing(p_prototype) && iip
        p_prototype = zeros(eltype(u0), (length(u0), length(u0)))
    end

    PD = ConservativePDSFunction{iip}(P; p_prototype, analytic, std_rhs)
    ConservativePDSProblem{iip}(PD, u0, tspan, p; kwargs...)
end

# Specialized constructor for ConservativePDSProblems setting `iip` manually
# (ConservativePDSFunction)
function ConservativePDSProblem{iip}(PD::ConservativePDSFunction, u0, tspan,
                                     p = NullParameters(); kwargs...) where {iip}
    ODEProblem(PD, u0, tspan, p, ConservativePDSProblem{iip}(); kwargs...)
end

# Specialized constructor for ConservativePDSFunction setting `iip` manually
function ConservativePDSFunction{iip}(P; kwargs...) where {iip}
    ConservativePDSFunction{iip, FullSpecialize}(P; kwargs...)
end

# Most specific constructor for ConservativePDSFunction
function ConservativePDSFunction{iip, FullSpecialize}(P;
                                                      p_prototype = nothing,
                                                      analytic = nothing,
                                                      std_rhs = nothing) where {iip}
    if std_rhs === nothing
        std_rhs = ConservativePDSStdRHS(P, p_prototype)
    end
    ConservativePDSFunction{iip, FullSpecialize, typeof(P), typeof(p_prototype),
                            typeof(std_rhs), typeof(analytic)}(P, p_prototype, std_rhs,
                                                               analytic)
end

# Evaluation of a ConservativePDSFunction
function (PD::ConservativePDSFunction)(u, p, t)
    return PD.std_rhs(u, p, t)
end

function (PD::ConservativePDSFunction)(du, u, p, t)
    return PD.std_rhs(du, u, p, t)
end

# Default implementation of the standard right-hand side evaluation function
struct ConservativePDSStdRHS{P, PrototypeP, TMP} <: Function
    p::P
    p_prototype::PrototypeP
    tmp::TMP
end

function ConservativePDSStdRHS(P, p_prototype)
    if p_prototype isa AbstractSparseMatrix
        tmp = zeros(eltype(p_prototype), (size(p_prototype, 1),))
    else
        tmp = nothing
    end
    ConservativePDSStdRHS(P, p_prototype, tmp)
end

# Evaluation of a ConservativePDSStdRHS (out-of-place)
function (PD::ConservativePDSStdRHS)(u, p, t)
    #vec(sum(PD.p(u, p, t), dims = 2)) - vec(sum(PD.p(u, p, t), dims = 1))
    P = PD.p(u, p, t)

    f = zeros(eltype(P), size(u))
    @fastmath @inbounds @simd for I in CartesianIndices(P)
        if !iszero(P[I])
            f[I[1]] += P[I]
            f[I[2]] -= P[I]
        end
    end
    return f
end

function (PD::ConservativePDSStdRHS)(u::SVector, p, t)
    P = PD.p(u, p, t)

    f = similar(P[:, 1]) #constructs MVector
    zeroT = zero(eltype(P))
    for i in eachindex(f)
        f[i] = zeroT
    end

    @fastmath @inbounds @simd for I in CartesianIndices(P)
        if !iszero(P[I])
            f[I[1]] += P[I]
            f[I[2]] -= P[I]
        end
    end

    return SVector(f)
end

# Evaluation of a ConservativePDSStdRHS (in-place)
function (PD::ConservativePDSStdRHS)(du, u, p, t)
    PD.p(PD.p_prototype, u, p, t)
    sum_terms!(du, PD.tmp, PD.p_prototype)
    return nothing
end

# Generic fallback (for dense arrays)
# This implementation does not need any auxiliary vectors
@inline function sum_terms!(du, tmp, P)
    for i in eachindex(du)
        du[i] = zero(eltype(du))
        for j in eachindex(du)
            du[i] += P[i, j] - P[j, i]
        end
    end
    return nothing
end

# Same result but more efficient - at least currently for SparseMatrixCSC
@inline function sum_terms!(du, tmp, P::AbstractSparseMatrix)
    # row sum coded as matrix vector product
    fill!(tmp, one(eltype(tmp)))
    mul!(vec(du), P, tmp)
    #sum!(vec(du), P)

    #column sum
    sum!(tmp', P)

    vec(du) .-= tmp
    return nothing
end

@inline function sum_terms!(du, tmp, P::Tridiagonal)
    Base.require_one_based_indexing(du, P.dl, P.du)
    @assert length(du) == length(P.dl) + 1 == length(P.du) + 1

    let i = 1
        Pij = P.du[i]
        Pji = P.dl[i]
        du[i] = Pij - Pji
    end
    for i in 2:(length(du) - 1)
        Pij = P.dl[i - 1] + P.du[i]
        Pji = P.du[i - 1] + P.dl[i]
        du[i] = Pij - Pji
    end
    let i = lastindex(du)
        Pij = P.dl[i - 1]
        Pji = P.du[i - 1]
        du[i] = Pij - Pji
    end
    return nothing
end
