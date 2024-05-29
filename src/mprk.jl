# Helper functions
add_small_constant(v, small_constant) = v .+ small_constant

function add_small_constant(v::SVector{N, T}, small_constant::T) where {N, T}
    v + SVector{N, T}(ntuple(i -> small_constant, N))
end

#####################################################################
p_prototype(u, f) = zeros(eltype(u), length(u), length(u))
p_prototype(u, f::ConservativePDSFunction) = zero(f.p_prototype)

#####################################################################
# out-of-place for dense and static arrays
function build_mprk_matrix(P, sigma, dt; d = nothing)
    # re-use the in-place version implemented below
    M = similar(P)
    build_mprk_matrix!(M, P, sigma, dt; d = d)

    if P isa StaticArray
        return SMatrix(M)
    else
        return M
    end
end

# in-place for dense arrays
function build_mprk_matrix!(M, P, sigma, dt; d = nothing)
    # M[i,i] = (sigma[i] + dt * sum_j P[j,i]) / sigma[i]
    # M[i,j] = -dt * P[i,j] / sigma[j]
    # TODO: the performance of this can likely be improved
    Base.require_one_based_indexing(M, P, sigma)
    @assert size(M, 1) == size(M, 2) == size(P, 1) == size(P, 2) == length(sigma)
    if !isnothing(d)
        Base.require_one_based_indexing(d)
        @assert length(sigma) == length(d)
    end

    zeroM = zero(eltype(P))

    # Set sigma on diagonal
    @inbounds for i in eachindex(sigma)
        M[i, i] = sigma[i]
    end
    # Add nonconservative destruction terms to diagonal (PDSFunctions only!)
    if !isnothing(d)
        @inbounds for i in eachindex(d)
            M[i, i] += dt * d[i]
        end
    end

    # Run through P and fill M accordingly.
    # If P[i,j] ≠ 0 set M[i,j] = -dt*P[i,j] and add dt*P[i,j] to M[j,j].
    @fastmath @inbounds @simd for I in CartesianIndices(P)
        if I[1] != I[2]
            if !iszero(P[I])
                dtP = dt * P[I]
                M[I] = -dtP / sigma[I[2]]
                M[I[2], I[2]] += dtP
            else
                M[I] = zeroM
            end
        end
    end

    # Divide diagonal elements by Patankar weights denominators
    @fastmath @inbounds @simd for i in eachindex(sigma)
        M[i, i] /= sigma[i]
    end

    return nothing
end

# optimized versions for Tridiagonal matrices
function build_mprk_matrix!(M::Tridiagonal, P::Tridiagonal, σ, dt)
    # M[i,i] = (sigma[i] + dt * sum_j P[j,i]) / sigma[i]
    # M[i,j] = -dt * P[i,j] / sigma[j]
    Base.require_one_based_indexing(M.dl, M.d, M.du,
                                    P.dl, P.d, P.du, σ)
    @assert length(M.dl) + 1 == length(M.d) == length(M.du) + 1 ==
            length(P.dl) + 1 == length(P.d) == length(P.du) + 1 == length(σ)

    for i in eachindex(M.d, σ)
        M.d[i] = σ[i]
    end

    for i in eachindex(M.dl, P.dl)
        dtP = dt * P.dl[i]
        M.dl[i] = -dtP / σ[i]
        M.d[i] += dtP
    end

    for i in eachindex(M.du, P.du)
        dtP = dt * P.du[i]
        M.du[i] = -dtP / σ[i + 1]
        M.d[i + 1] += dtP
    end

    for i in eachindex(M.d, σ)
        M.d[i] /= σ[i]
    end

    return nothing
end

#####################################################################
# Linear interpolations
@muladd @inline function linear_interpolant(Θ, dt, u0, u1, idxs::Nothing, T::Type{Val{0}})
    Θm1 = (1 - Θ)
    @.. broadcast=false Θm1 * u0+Θ * u1
end

@muladd @inline function linear_interpolant(Θ, dt, u0, u1, idxs, T::Type{Val{0}})
    Θm1 = (1 - Θ)
    @.. broadcast=false Θm1 * u0[idxs]+Θ * u1[idxs]
end

@muladd @inline function linear_interpolant!(out, Θ, dt, u0, u1, idxs::Nothing,
                                             T::Type{Val{0}})
    Θm1 = (1 - Θ)
    @.. broadcast=false out=Θm1 * u0 + Θ * u1
    out
end

@muladd @inline function linear_interpolant!(out, Θ, dt, u0, u1, idxs, T::Type{Val{0}})
    Θm1 = (1 - Θ)
    @views @.. broadcast=false out=Θm1 * u0[idxs] + Θ * u1[idxs]
    out
end

@inline function linear_interpolant(Θ, dt, u0, u1, idxs::Nothing, T::Type{Val{1}})
    @.. broadcast=false (u1 - u0)/dt
end

@inline function linear_interpolant(Θ, dt, u0, u1, idxs, T::Type{Val{1}})
    @.. broadcast=false (u1[idxs] - u0[idxs])/dt
end

@inline function linear_interpolant!(out, Θ, dt, u0, u1, idxs::Nothing, T::Type{Val{1}})
    @.. broadcast=false out=(u1 - u0) / dt
    out
end

@inline function linear_interpolant!(out, Θ, dt, u0, u1, idxs, T::Type{Val{1}})
    @views @.. broadcast=false out=(u1[idxs] - u0[idxs]) / dt
    out
end

### MPE #####################################################################################
"""
    MPE([linsolve = ...])

The first-order modified Patankar-Euler algorithm for (conservative)
production-destruction systems. This one-step, one-stage method is
first-order accurate, unconditionally positivity-preserving, and
linearly implicit.

The modified Patankar-Euler method requires the special structure of a
[`PDSProblem`](@ref) or a [`ConservativePDSProblem`](@ref).

You can optionally choose the linear solver to be used by passing an
algorithm from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
as keyword argument `linsolve`.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
- TODO: Add literature for nonconservative part - Meister ???
"""
struct MPE{F} <: OrdinaryDiffEqAlgorithm
    linsolve::F
end

function MPE(; linsolve = LUFactorization())
    MPE(linsolve)
end

# TODO: Consider switching to the interface of LinearSolve.jl directly,
#       avoiding `dolinesolve` from OrdinaryDiffEq.jl.
# TODO: Think about adding preconditioners to the MPE algorithm
# This hack is currently required to make OrdinaryDiffEq.jl happy...
function Base.getproperty(alg::MPE, f::Symbol)
    # preconditioners
    if f === :precs
        return Returns((nothing, nothing))
    else
        return getfield(alg, f)
    end
end

alg_order(::MPE) = 1
isfsal(::MPE) = false

struct MPEConstantCache{T} <: OrdinaryDiffEqConstantCache
    small_constant::T
end

function alg_cache(alg::MPE, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    MPEConstantCache(floatmin(uEltypeNoUnits))
end

function initialize!(integrator, cache::MPEConstantCache)
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsalfirst = zero(integrator.u)
    integrator.fsallast = integrator.fsalfirst
    integrator.k[1] = integrator.fsallast

    # TODO: Do we need to set fsalfirst here? The other non-FSAL caches
    #       in OrdinaryDiffEq.jl use something like
    #         integrator.fsalfirst = integrator.f(integrator.uprev, integrator,
    #                                             integrator.t) # Pre-start fsal
    #         integrator.stats.nf += 1
    #         integrator.fsallast = zero(integrator.fsalfirst)
    #         integrator.k[1] = integrator.fsalfirst
    #       Do we need something similar here to get a cache for k values
    #       with the correct units?
end

function perform_step!(integrator, cache::MPEConstantCache, repeat_step = false)
    f = integrator.f

    if f isa ConservativePDSFunction
        perform_step_conservative!(integrator, cache, repeat_step)
    elseif f isa PDSFunction
        perform_step_nonconservative!(integrator, cache, repeat_step)
    else
        throw(ArgumentError("MPE can only be applied to production-destruction systems"))
    end
end

function perform_step_conservative!(integrator, cache::MPEConstantCache,
                                    repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack small_constant = cache

    # Attention: Implementation assumes that the pds is conservative,
    # i.e., P[i, i] == 0 for all i  

    # evaluate production matrix
    P = f.p(uprev, p, t)
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)

    # build linear system matrix
    M = build_mprk_matrix(P, σ, dt)

    # solve linear system
    linprob = LinearProblem(M, uprev)
    sol = solve(linprob, alg.linsolve,
                alias_A = false, alias_b = false,
                assumptions = LinearSolve.OperatorAssumptions(true))
    u = sol.u
    integrator.stats.nsolve += 1

    integrator.u = u
end

function perform_step_nonconservative!(integrator, cache::MPEConstantCache,
                                       repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack small_constant = cache

    # Attention: Implementation assumes that the pds is conservative,
    # i.e., P[i, i] == 0 for all i  

    # evaluate PDS
    P = f.p(uprev, p, t)
    d = f.d(uprev, p, t)
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)

    # build linear system matrix and right hand side
    rhs = uprev + dt * diag(P)
    M = build_mprk_matrix(P, σ, dt; d = d)

    # solve linear system
    linprob = LinearProblem(M, rhs)
    sol = solve(linprob, alg.linsolve,
                alias_A = false, alias_b = false,
                assumptions = LinearSolve.OperatorAssumptions(true))
    u = sol.u
    integrator.stats.nsolve += 1

    integrator.u = u
end

struct MPECache{uType, rateType, PType, F, uNoUnitsType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    k::rateType
    fsalfirst::rateType
    P::PType
    D::uType
    linsolve_tmp::uType  # stores rhs of linear system
    linsolve::F
    weight::uNoUnitsType
end

function alg_cache(alg::MPE, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    tmp = zero(u)

    P = p_prototype(u, f)
    linsolve_tmp = zero(u)
    weight = similar(u, uEltypeNoUnits)
    recursivefill!(weight, false)

    # We use P to store the evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    linprob = LinearProblem(P, _vec(linsolve_tmp); u0 = _vec(tmp))
    linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                    assumptions = LinearSolve.OperatorAssumptions(true))

    MPECache(u, uprev, tmp,
             zero(rate_prototype), # k
             zero(rate_prototype), # fsalfirst
             P,
             zero(u), # D
             linsolve_tmp, linsolve, weight)
end

function initialize!(integrator, cache::MPECache)
    @unpack k, fsalfirst = cache
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    integrator.kshortsize = 1
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
end

function perform_step!(integrator, cache::MPECache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack P, D, weight = cache

    # We use P to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system  

    # TODO: Shall we require the users to set unused entries to zero?
    fill!(P, zero(eltype(P)))

    f.p(P, uprev, p, t) # evaluate production terms
    integrator.stats.nf += 1

    build_mprk_matrix!(P, P, uprev, dt)

    # Same as linres = P \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = P, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    u .= linres
    integrator.stats.nsolve += 1
end

# interpolation specializations
function interp_summary(::Type{cacheType},
                        dense::Bool) where {
                                            cacheType <: Union{MPEConstantCache, MPECache}}
    "1st order linear"
end

function _ode_interpolant(Θ, dt, u0, u1, k,
                          cache::Union{MPEConstantCache, MPECache},
                          idxs, # Optionally specialize for ::Nothing and others
                          T::Type{Val{0}},
                          differential_vars::Nothing)
    linear_interpolant(Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant!(out, Θ, dt, u0, u1, k,
                           cache::Union{MPEConstantCache, MPECache},
                           idxs, # Optionally specialize for ::Nothing and others
                           T::Type{Val{0}},
                           differential_vars::Nothing)
    linear_interpolant!(out, Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant(Θ, dt, u0, u1, k,
                          cache::Union{MPEConstantCache, MPECache},
                          idxs, # Optionally specialize for ::Nothing and others
                          T::Type{Val{1}},
                          differential_vars::Nothing)
    linear_interpolant(Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant!(out, Θ, dt, u0, u1, k,
                           cache::Union{MPEConstantCache, MPECache},
                           idxs, # Optionally specialize for ::Nothing and others
                           T::Type{Val{1}},
                           differential_vars::Nothing)
    linear_interpolant!(out, Θ, dt, u0, u1, idxs, T)
end

### MPRK22 #####################################################################################
"""
    MPRK22(α; [linsolve = ...])

The second-order modified Patankar-Runge-Kutta algorithm for (conservative)
production-destruction systems. This one-step, two-stage method is
second-order accurate, unconditionally positivity-preserving, and linearly
implicit. The parameter `α` is described by Kopecz and Meister (2018) and
studied by Izgin, Kopecz and Meister (2022) as well as
Torlo, Öffner and Ranocha (2022).

This modified Patankar-Runge-Kutta method requires the special structure of a
[`PDSProblem`](@ref) or a [`ConservativePDSProblem`](@ref).

You can optionally choose the linear solver to be used by passing an
algorithm from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
as keyword argument `linsolve`.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
- Stefan Kopecz and Andreas Meister.
  "On order conditions for modified Patankar-Runge-Kutta schemes."
  Applied Numerical Mathematics 123 (2018): 159-179.
  [DOI: 10.1016/j.apnum.2017.09.004](https://doi.org/10.1016/j.apnum.2017.09.004)
- Thomas Izgin, Stefan Kopecz, and Andreas Meister.
  "On Lyapunov stability of positive and conservative time integrators and application
  to second order modified Patankar-Runge-Kutta schemes."
  ESAIM: Mathematical Modelling and Numerical Analysis 56.3 (2022): 1053-1080.
  [DOI: 10.1051/m2an/2022031](https://doi.org/10.1051/m2an/2022031)
- Davide Torlo, Philipp Öffner, and Hendrik Ranocha.
  "Issues with positivity-preserving Patankar-type schemes."
  Applied Numerical Mathematics 182 (2022): 117-147.
  [DOI: 10.1016/j.apnum.2022.07.014](https://doi.org/10.1016/j.apnum.2022.07.014)
"""
struct MPRK22{T, Thread, F} <: OrdinaryDiffEqAdaptiveAlgorithm
    alpha::T
    thread::Thread
    linsolve::F
end

function MPRK22(alpha; thread = False(), linsolve = LUFactorization())
    MPRK22{typeof(alpha), typeof(thread), typeof(linsolve)}(alpha, thread, linsolve)
end

# TODO: Consider switching to the interface of LinearSolve.jl directly,
#       avoiding `dolinesolve` from OrdinaryDiffEq.jl.
# TODO: Think about adding preconditioners to the MPRK22 algorithm
# This hack is currently required to make OrdinaryDiffEq.jl happy...
function Base.getproperty(alg::MPRK22, f::Symbol)
    # preconditioners
    if f === :precs
        return Returns((nothing, nothing))
    else
        return getfield(alg, f)
    end
end

alg_order(::MPRK22) = 2
isfsal(::MPRK22) = false

struct MPRK22Cache{uType, rateType, PType, tabType, Thread, F, uNoUnitsType} <:
       OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    atmp::uType
    k::rateType
    fsalfirst::rateType
    P::PType
    P2::PType
    D::uType
    D2::uType
    σ::uType
    tab::tabType
    thread::Thread
    linsolve_tmp::uType  # stores rhs of linear system
    linsolve::F
    weight::uNoUnitsType
end

function alg_cache(alg::MPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    tab = MPRK22ConstantCache(alg.alpha, 1 - 1 / (2 * alg.alpha), 1 / (2 * alg.alpha),
                              alg.alpha, floatmin(uEltypeNoUnits))

    tmp = zero(u)

    P2 = p_prototype(u, f)
    linsolve_tmp = zero(u)
    weight = similar(u, uEltypeNoUnits)
    recursivefill!(weight, false)

    # We use P2 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    linprob = LinearProblem(P2, _vec(linsolve_tmp); u0 = _vec(tmp))
    linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                    assumptions = LinearSolve.OperatorAssumptions(true))

    MPRK22Cache(u, uprev, tmp,
                zero(u), # atmp
                zero(rate_prototype), # k
                zero(rate_prototype), #fsalfirst
                p_prototype(u, f), # P
                P2, # P2
                zero(u), # D
                zero(u), # D2
                zero(u), # σ
                tab, alg.thread,
                linsolve_tmp, linsolve, weight)
end

struct MPRK22ConstantCache{T} <: OrdinaryDiffEqConstantCache
    a21::T
    b1::T
    b2::T
    c2::T
    small_constant::T
end

function alg_cache(alg::MPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}

    #TODO: Should assert alg.alpha >= 0.5

    MPRK22ConstantCache(alg.alpha, 1 - 1 / (2 * alg.alpha), 1 / (2 * alg.alpha), alg.alpha,
                        floatmin(uEltypeNoUnits))
end

function initialize!(integrator, cache::MPRK22ConstantCache)
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsalfirst = zero(integrator.u)
    integrator.fsallast = integrator.fsalfirst
    integrator.k[1] = integrator.fsallast

    # TODO: Do we need to set fsalfirst here? The other non-FSAL caches
    #       in OrdinaryDiffEq.jl use something like
    #         integrator.fsalfirst = integrator.f(integrator.uprev, integrator,
    #                                             integrator.t) # Pre-start fsal
    #         integrator.stats.nf += 1
    #         integrator.fsallast = zero(integrator.fsalfirst)
    #         integrator.k[1] = integrator.fsalfirst
    #       Do we need something similar here to get a cache for k values
    #       with the correct units?
end

function perform_step!(integrator, cache::MPRK22ConstantCache, repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack a21, b1, b2, small_constant = cache

    # Attention: Implementation assumes that the pds is conservative,
    # i.e. , P[i, i] == 0 for all i

    # evaluate production matrix
    P = f.p(uprev, p, t)
    Ptmp = a21 * P
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)

    # build linear system matrix
    M = build_mprk_matrix(Ptmp, σ, dt)

    # solve linear system
    linprob = LinearProblem(M, uprev)
    sol = solve(linprob, alg.linsolve,
                alias_A = false, alias_b = false,
                assumptions = LinearSolve.OperatorAssumptions(true))
    u = sol.u
    integrator.stats.nsolve += 1

    # compute Patankar weight denominator
    if isone(a21)
        σ = u
    else
        # σ = σ .* (u ./ σ) .^ (1 / a21) # generated Infs when solving brusselator
        σ = σ .^ (1 - 1 / a21) .* u .^ (1 / a21)
    end
    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(σ, small_constant)

    P2 = f.p(u, p, t + a21 * dt)
    Ptmp = b1 * P + b2 * P2
    integrator.stats.nf += 1

    # build linear system matrix
    M = build_mprk_matrix(Ptmp, σ, dt)

    # solve linear system
    linprob = LinearProblem(M, uprev)
    sol = solve(linprob, alg.linsolve,
                alias_A = false, alias_b = false,
                assumptions = LinearSolve.OperatorAssumptions(true))
    u = sol.u
    integrator.stats.nsolve += 1

    # copied from perform_step for HeunConstantCache
    # If a21 = 1.0, then σ is the MPE approximation and thus suited for stiff problems.
    # If a21 ≠ 1.0, σ might be a bad choice to estimate errors.
    tmp = u - σ
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)

    integrator.u = u
end

function initialize!(integrator, cache::MPRK22Cache)
    @unpack k, fsalfirst = cache
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    integrator.kshortsize = 1
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
end

function perform_step!(integrator, cache::MPRK22Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, atmp, P, P2, D, D2, σ, thread, weight = cache
    @unpack a21, b1, b2, c2, small_constant = cache.tab

    # We use P2 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system

    f.p(P, uprev, p, t) # evaluate production terms
    integrator.stats.nf += 1
    @.. broadcast=false P2=a21 * P

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    build_mprk_matrix!(P2, P2, σ, dt)

    # Same as linres = P2 \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = P2, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    u .= linres
    integrator.stats.nsolve += 1

    if isone(a21)
        σ .= u
    else
        @.. broadcast=false σ=σ^(1 - 1 / a21) * u^(1 / a21)
    end
    @.. broadcast=false σ=σ + small_constant

    f.p(P2, u, p, t + a21 * dt) # evaluate production terms
    integrator.stats.nf += 1

    @.. broadcast=false P2=b1 * P + b2 * P2

    build_mprk_matrix!(P2, P2, σ, dt)

    # Same as linres = P2 \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = P2, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    u .= linres
    integrator.stats.nsolve += 1

    @.. broadcast=false tmp=u - σ
    calculate_residuals!(atmp, tmp, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         thread)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)
end

### MPRK43 #####################################################################################
"""
    MPRK43I(α, β; [linsolve = ...])

A family of third-order modified Patankar-Runge-Kutta schemes for (conservative)
production-destruction systems, which is based on the two-parameter family of third order explicit Runge--Kutta schemes.
Each member of this family is a one-step method with four-stages which is
third-order accurate, unconditionally positivity-preserving, conservative and linearly
implicit. In this implementation the stage-values are conservative as well.
The parameters `α` and `β` must be chosen such that the Runge--Kutta coefficients are nonnegative, 
see Kopecz and Meister (2018) for details. 

These modified Patankar-Runge-Kutta methods require the special structure of a
[`PDSProblem`](@ref) or a [`ConservativePDSProblem`](@ref).

You can optionally choose the linear solver to be used by passing an
algorithm from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
as keyword argument `linsolve`.

## References

- Stefan Kopecz and Andreas Meister.
  "Unconditionally positive and conservative third order modified Patankar–Runge–Kutta 
   discretizations of production–destruction systems."
   BIT Numerical Mathematics 58 (2018): 691–728.
  [DOI: 10.1007/s10543-018-0705-1](https://doi.org/10.1007/s10543-018-0705-1)
"""
struct MPRK43I{T, Thread, F} <: OrdinaryDiffEqAdaptiveAlgorithm
    alpha::T
    beta::T
    thread::Thread
    linsolve::F
end

function MPRK43I(alpha, beta; thread = False(), linsolve = LUFactorization())
    MPRK43I{typeof(alpha), typeof(thread), typeof(linsolve)}(alpha, beta,
                                                             thread, linsolve)
end

"""
    MPRK43II(γ; [linsolve = ...])

A family of third-order modified Patankar-Runge-Kutta schemes for (conservative)
production-destruction systems, which is based on the one-parameter family of third order explicit Runge--Kutta schemes with 
non-negative Runge--Kutta coefficients.
Each member of this family is a one-step method with four-stages which is
third-order accurate, unconditionally positivity-preserving, conservative and linearly
implicit. In this implementation the stage-values are conservative as well. The parameter `γ` must satisfy
`3/8 ≤ γ ≤ 3/4`. 
Further details are given in Kopecz and Meister (2018).  

These modified Patankar-Runge-Kutta methods require the special structure of a
[`PDSProblem`](@ref) or a [`ConservativePDSProblem`](@ref).

You can optionally choose the linear solver to be used by passing an
algorithm from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
as keyword argument `linsolve`.

## References

- Stefan Kopecz and Andreas Meister.
  "Unconditionally positive and conservative third order modified Patankar–Runge–Kutta 
   discretizations of production–destruction systems."
   BIT Numerical Mathematics 58 (2018): 691–728.
  [DOI: 10.1007/s10543-018-0705-1](https://doi.org/10.1007/s10543-018-0705-1)
"""
struct MPRK43II{T, Thread, F} <: OrdinaryDiffEqAdaptiveAlgorithm
    gamma::T
    thread::Thread
    linsolve::F
end

function MPRK43II(gamma; thread = False(), linsolve = LUFactorization())
    MPRK43II{typeof(gamma), typeof(thread), typeof(linsolve)}(gamma, thread, linsolve)
end

# TODO: Consider switching to the interface of LinearSolve.jl directly,
#       avoiding `dolinesolve` from OrdinaryDiffEq.jl.
# TODO: Think about adding preconditioners to the algorithm
# This hack is currently required to make OrdinaryDiffEq.jl happy...
function Base.getproperty(alg::Union{MPRK43I, MPRK43II}, f::Symbol)
    # preconditioners
    if f === :precs
        return Returns((nothing, nothing))
    else
        return getfield(alg, f)
    end
end

alg_order(::MPRK43I) = 3
alg_order(::MPRK43II) = 3
isfsal(::MPRK43I) = false
isfsal(::MPRK43II) = false

struct MPRK43Cache{uType, rateType, PType, tabType, Thread, F, uNoUnitsType} <:
       OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    atmp::uType
    k::rateType
    fsalfirst::rateType
    P::PType
    P2::PType
    P3::PType
    D::uType
    D2::uType
    D3::uType
    σ::uType
    tab::tabType
    thread::Thread
    linsolve_tmp::uType  # stores rhs of linear system
    linsolve::F
    weight::uNoUnitsType
end

function get_constant_parameters(alg::MPRK43I)
    @assert alg.alpha ≥ 1 / 3&&alg.alpha ≠ 2 / 3 "MPRK43I requires α ≥ 1/3 and α ≠ 2/3."
    α0 = 1 / 6 * (3 + (3 - 2 * sqrt(2))^(1 / 3) + (3 + 2 * sqrt(2))^(1 / 3))
    if 1 / 3 ≤ alg.alpha < 2 / 3
        @assert 2/3≤alg.beta≤3*alg.alpha*(1-alg.alpha) "For this choice of α MPRK43I requires 2/3 ≤ β ≤ 3α(1-α)."
    elseif 2 / 3 < alg.alpha ≤ α0
        @assert 3*alg.alpha*(1-alg.alpha)≤alg.beta≤2/3 "For this choice of α MPRK43I requires 3α(1-α) ≤ β ≤ 2/3."
    else
        @assert (3 * alg.alpha - 2)/(6 * alg.alpha - 3)≤alg.beta≤2/3 "For this choice of α MPRK43I requires (3α-2)/(6α-3) ≤ β ≤ 2/3."
    end

    a21 = alg.alpha
    a31 = (3 * alg.alpha * alg.beta * (1 - alg.alpha) - alg.beta^2) /
          (alg.alpha * (2 - 3 * alg.alpha))
    a32 = (alg.beta * (alg.beta - alg.alpha)) / (alg.alpha * (2 - 3 * alg.alpha))
    b1 = 1 + (2 - 3 * (alg.alpha + alg.beta)) / (6 * alg.alpha * alg.beta)
    b2 = (3 * alg.beta - 2) / (6 * alg.alpha * (alg.beta - alg.alpha))
    b3 = (2 - 3 * alg.alpha) / (6 * alg.beta * (alg.beta - alg.alpha))
    c2 = alg.alpha
    c3 = alg.beta

    beta2 = 1 / (2 * a21)
    beta1 = 1 - beta2

    q1 = 1 / (3 * a21 * (a31 + a32) * b3)
    q2 = 1 / a21

    @assert all((a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2) .≥ 0) "MPRK43I requires nonnegative RK coefficients."
    return a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2
end

function get_constant_parameters(alg::MPRK43II)
    @assert 3/8≤alg.gamma≤3/4 "MPRK43II requires 3/8 ≤ γ ≤ 3/4."

    a21 = 2 * one(alg.gamma) / 3
    a31 = a21 - 1 / (4 * alg.gamma)
    a32 = 1 / (4 * alg.gamma)
    b1 = one(alg.gamma) / 4
    b2 = 3 * one(alg.gamma) / 4 - alg.gamma
    b3 = alg.gamma
    c2 = a21
    c3 = a21

    beta2 = 1 / (2 * a21)
    beta1 = 1 - beta2

    q1 = 1 / (3 * a21 * (a31 + a32) * b3)
    q2 = 1 / a21

    @assert all((a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2) .≥ 0) "MPRK43II requires nonnegative RK coefficients."
    return a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2
end

struct MPRK43ConstantCache{T} <: OrdinaryDiffEqConstantCache
    a21::T
    a31::T
    a32::T
    b1::T
    b2::T
    b3::T
    c2::T
    c3::T
    beta1::T
    beta2::T
    q1::T
    q2::T
    small_constant::T
end

function alg_cache(alg::Union{MPRK43I, MPRK43II}, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2 = get_constant_parameters(alg)
    tab = MPRK43ConstantCache(a21, a31, a32, b1, b2, b3, c2, c3,
                              beta1, beta2, q1, q2, floatmin(uEltypeNoUnits))
end

function initialize!(integrator, cache::MPRK43ConstantCache)
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsalfirst = zero(integrator.u)
    integrator.fsallast = integrator.fsalfirst
    integrator.k[1] = integrator.fsallast

    # TODO: Do we need to set fsalfirst here? The other non-FSAL caches
    #       in OrdinaryDiffEq.jl use something like
    #         integrator.fsalfirst = integrator.f(integrator.uprev, integrator,
    #                                             integrator.t) # Pre-start fsal
    #         integrator.stats.nf += 1
    #         integrator.fsallast = zero(integrator.fsalfirst)
    #         integrator.k[1] = integrator.fsalfirst
    #       Do we need something similar here to get a cache for k values
    #       with the correct units?
end

function perform_step!(integrator, cache::MPRK43ConstantCache, repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2, small_constant = cache

    # Attention: Implementation assumes that the pds is conservative,
    # i.e. , P[i, i] == 0 for all i

    # evaluate production matrix
    P = f.p(uprev, p, t)
    Ptmp = a21 * P
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)
    if !(q1 ≈ q2)
        σ0 = σ
    end

    # build linear system matrix
    M = build_mprk_matrix(Ptmp, σ, dt)

    # solve linear system
    linprob = LinearProblem(M, uprev)
    sol = solve(linprob, alg.linsolve,
                alias_A = false, alias_b = false,
                assumptions = LinearSolve.OperatorAssumptions(true))
    u2 = sol.u
    u = u2
    integrator.stats.nsolve += 1

    # compute Patankar weight denominator
    σ = σ .^ (1 - q1) .* u .^ q1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(σ, small_constant)

    P2 = f.p(u, p, t + c2 * dt)
    Ptmp = a31 * P + a32 * P2
    integrator.stats.nf += 1

    # build linear system matrix
    M = build_mprk_matrix(Ptmp, σ, dt)

    # solve linear system
    linprob = LinearProblem(M, uprev)
    sol = solve(linprob, alg.linsolve,
                alias_A = false, alias_b = false,
                assumptions = LinearSolve.OperatorAssumptions(true))
    u = sol.u
    integrator.stats.nsolve += 1

    # compute Patankar weight denominator
    if !(q1 ≈ q2)
        σ = σ0 .^ (1 - q2) .* u2 .^ q2

        # avoid division by zero due to zero Patankar weights
        σ = add_small_constant(σ, small_constant)
    end

    Ptmp = beta1 * P + beta2 * P2

    # build linear system matrix
    M = build_mprk_matrix(Ptmp, σ, dt)

    # solve linear system
    linprob = LinearProblem(M, uprev)
    sol = solve(linprob, alg.linsolve,
                alias_A = false, alias_b = false,
                assumptions = LinearSolve.OperatorAssumptions(true))
    σ = sol.u
    integrator.stats.nsolve += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(σ, small_constant)

    P3 = f.p(u, p, t + c2 * dt)
    Ptmp = b1 * P + b2 * P2 + b3 * P3
    integrator.stats.nf += 1

    # build linear system matrix
    M = build_mprk_matrix(Ptmp, σ, dt)

    # solve linear system
    linprob = LinearProblem(M, uprev)
    sol = solve(linprob, alg.linsolve,
                alias_A = false, alias_b = false,
                assumptions = LinearSolve.OperatorAssumptions(true))
    u = sol.u
    integrator.stats.nsolve += 1

    tmp = u - σ
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)

    integrator.u = u
end

function alg_cache(alg::Union{MPRK43I, MPRK43II}, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2 = get_constant_parameters(alg)
    tab = MPRK43ConstantCache(a21, a31, a32, b1, b2, b3, c2, c3,
                              beta1, beta2, q1, q2, floatmin(uEltypeNoUnits))

    tmp = zero(u)

    P3 = p_prototype(u, f)
    linsolve_tmp = zero(u)
    weight = similar(u, uEltypeNoUnits)
    recursivefill!(weight, false)

    # We use P3 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    linprob = LinearProblem(P3, _vec(linsolve_tmp); u0 = _vec(tmp))
    linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                    assumptions = LinearSolve.OperatorAssumptions(true))

    MPRK43Cache(u, uprev, tmp,
                zero(u), # atmp
                zero(rate_prototype), # k
                zero(rate_prototype), #fsalfirst
                p_prototype(u, f), # P
                p_prototype(u, f), # P2
                P3,
                zero(u), # D
                zero(u), # D2
                zero(u), # D3
                zero(u), # σ
                tab, alg.thread,
                linsolve_tmp, linsolve, weight)
end

function initialize!(integrator, cache::MPRK43Cache)
    @unpack k, fsalfirst = cache
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    integrator.kshortsize = 1
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
end

function perform_step!(integrator, cache::MPRK43Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, atmp, P, P2, P3, D, D2, D3, σ, thread, weight = cache
    @unpack a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2, small_constant = cache.tab

    # We use P3 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system

    f.p(P, uprev, p, t) # evaluate production terms
    @.. broadcast=false P3=a21 * P
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = P3, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    u .= linres
    if !(q1 ≈ q2)
        atmp .= u #u2 in out-of-place version
    end
    integrator.stats.nsolve += 1

    @.. broadcast=false σ=σ^(1 - q1) * u^q1
    @.. broadcast=false σ=σ + small_constant

    f.p(P2, u, p, t + c2 * dt) # evaluate production terms
    @.. broadcast=false P3=a31 * P + a32 * P2
    integrator.stats.nf += 1

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = P3, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    u .= linres
    integrator.stats.nsolve += 1

    if !(q1 ≈ q2)
        @.. broadcast=false σ=(uprev + small_constant)^(1 - q2) * atmp^q2
        @.. broadcast=false σ=σ + small_constant
    end

    @.. broadcast=false P3=beta1 * P + beta2 * P2

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = P3, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    σ .= linres
    integrator.stats.nsolve += 1

    f.p(P3, u, p, t + c3 * dt) # evaluate production terms
    @.. broadcast=false P3=b1 * P + b2 * P2 + b3 * P3
    integrator.stats.nf += 1

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = P3, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    u .= linres
    integrator.stats.nsolve += 1

    @.. broadcast=false tmp=u - σ
    calculate_residuals!(atmp, tmp, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         thread)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)
end

########################################################################################

# interpolation specializations
function interp_summary(::Type{cacheType},
                        dense::Bool) where {
                                            cacheType <:
                                            Union{MPRK22ConstantCache, MPRK22Cache,
                                                  MPRK43ConstantCache, MPRK43Cache}}
    "1st order linear"
end

function _ode_interpolant(Θ, dt, u0, u1, k,
                          cache::Union{MPRK22ConstantCache, MPRK22Cache,
                                       MPRK43ConstantCache, MPRK43Cache},
                          idxs, # Optionally specialize for ::Nothing and others
                          T::Type{Val{0}},
                          differential_vars::Nothing)
    linear_interpolant(Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant!(out, Θ, dt, u0, u1, k,
                           cache::Union{MPRK22ConstantCache, MPRK22Cache,
                                        MPRK43ConstantCache, MPRK43Cache},
                           idxs, # Optionally specialize for ::Nothing and others
                           T::Type{Val{0}},
                           differential_vars::Nothing)
    linear_interpolant!(out, Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant(Θ, dt, u0, u1, k,
                          cache::Union{MPRK22ConstantCache, MPRK22Cache,
                                       MPRK43ConstantCache, MPRK43Cache},
                          idxs, # Optionally specialize for ::Nothing and others
                          T::Type{Val{1}},
                          differential_vars::Nothing)
    linear_interpolant(Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant!(out, Θ, dt, u0, u1, k,
                           cache::Union{MPRK22ConstantCache, MPRK22Cache,
                                        MPRK43ConstantCache, MPRK43Cache},
                           idxs, # Optionally specialize for ::Nothing and others
                           T::Type{Val{1}},
                           differential_vars::Nothing)
    linear_interpolant!(out, Θ, dt, u0, u1, idxs, T)
end
