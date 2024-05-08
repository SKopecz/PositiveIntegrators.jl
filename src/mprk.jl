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
function build_mprk_matrix(P, sigma, dt)
    # M[i,i] = (sigma[i] + dt*sum_j P[j,i])/sigma[i]
    # M[i,j] = -dt*P[i,j]/sigma[j]
    M = similar(P)
    zeroM = zero(eltype(M))

    # Set sigma on diagonal
    @inbounds for i in eachindex(sigma)
        M[i, i] = sigma[i]
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

    if P isa StaticArray
        return SMatrix(M)
    else
        return M
    end
end

# in-place for dense arrays
function build_mprk_matrix!(M, a, P, D, sigma, dt)
    # M[i,i] = (sigma[i] + dt * sum_j P[j,i]) / sigma[i]
    # M[i,j] = -dt * P[i,j] / sigma[j]
    # TODO: the performance of this can likely be improved
    Base.require_one_based_indexing(M, P, D, sigma)
    @assert size(M, 1) == size(M, 2) == size(P, 1) == size(P, 2) == length(D) ==
            length(sigma)

    for j in 1:length(sigma)
        for i in 1:length(sigma)
            if i == j
                M[i, i] = 1 + dt * a * D[i] / sigma[i]
            else
                M[i, j] = -dt * a * P[i, j] / sigma[j]
            end
        end
    end

    return M
end

function build_mprk_matrix!(M, b1, P1, D1, b2, P2, D2, sigma, dt)
    # M[i,i] = (sigma[i] + dt * sum_j P[j,i]) / sigma[i]
    # M[i,j] = -dt * P[i,j] / sigma[j]
    # TODO: the performance of this can likely be improved
    Base.require_one_based_indexing(M, P1, D1, P2, D2, sigma)
    @assert size(M, 1) == size(M, 2) == size(P1, 1) == size(P1, 2) == length(D1) ==
            size(P2, 1) == size(P2, 2) == length(D1) == length(sigma)

    for j in 1:length(sigma)
        for i in 1:length(sigma)
            if i == j
                M[i, i] = 1 + dt * (b1 * D1[i] + b2 * D2[i]) / sigma[i]
            else
                M[i, j] = -dt * (b1 * P1[i, j] + b2 * P2[i, j]) / sigma[j]
            end
        end
    end

    return M
end

# optimized versions for Tridiagonal matrices
function build_mprk_matrix!(M::Tridiagonal,
                            a, P::Tridiagonal, D,
                            sigma, dt)
    # M[i,i] = (sigma[i] + dt * sum_j P[j,i]) / sigma[i]
    # M[i,j] = -dt * P[i,j] / sigma[j]
    Base.require_one_based_indexing(M.dl, M.d, M.du,
                                    P.dl, P.d, P.du,
                                    D, sigma)
    @assert length(M.dl) + 1 == length(M.d) == length(M.du) + 1 ==
            length(P.dl) + 1 == length(P.d) == length(P.du) + 1 ==
            length(D) == length(sigma)

    factor = a * dt

    for i in eachindex(M.d, D, sigma)
        M.d[i] = 1 + factor * D[i] / sigma[i]
    end

    for i in eachindex(M.dl, P.dl)
        M.dl[i] = -factor * P.dl[i] / sigma[i]
    end

    for i in eachindex(M.dl, P.dl)
        M.du[i] = -factor * P.du[i] / sigma[i + 1]
    end

    return M
end

function build_mprk_matrix!(M::Tridiagonal,
                            b1, P1::Tridiagonal, D1,
                            b2, P2::Tridiagonal, D2,
                            sigma, dt)
    # M[i,i] = (sigma[i] + dt * sum_j P[j,i]) / sigma[i]
    # M[i,j] = -dt * P[i,j] / sigma[j]
    Base.require_one_based_indexing(M.dl, M.d, M.du,
                                    P1.dl, P1.d, P1.du, D1,
                                    P2.dl, P2.d, P2.du, D2,
                                    sigma)
    @assert length(M.dl) + 1 == length(M.d) == length(M.du) + 1 ==
            length(P1.dl) + 1 == length(P1.d) == length(P1.du) + 1 ==
            length(D1) ==
            length(P2.dl) + 1 == length(P2.d) == length(P2.du) + 1 ==
            length(D2) == length(sigma)

    factor1 = b1 * dt
    factor2 = b2 * dt

    for i in eachindex(M.d, D1, D2, sigma)
        M.d[i] = 1 + (factor1 * D1[i] + factor2 * D2[i]) / sigma[i]
    end

    for i in eachindex(M.dl, P1.dl, P2.dl)
        M.dl[i] = -(factor1 * P1.dl[i] + factor2 * P2.dl[i]) / sigma[i]
    end

    for i in eachindex(M.dl, P1.du, P2.du)
        M.du[i] = -(factor1 * P1.du[i] + factor2 * P2.du[i]) / sigma[i + 1]
    end

    return M
end

#####################################################################
# Generic fallback (for dense arrays)
sum_destruction_terms!(D, P) = sum!(D', P)

function sum_destruction_terms!(D, P::Tridiagonal)
    Base.require_one_based_indexing(D, P.dl, P.d, P.du)
    @assert length(D) == length(P.dl) + 1 == length(P.d) == length(P.du) + 1

    let i = 1
        D[i] = P.d[i] + P.dl[i]
    end
    for i in 2:(length(D) - 1)
        D[i] = P.du[i - 1] + P.d[i] + P.dl[i]
    end
    let i = lastindex(D)
        D[i] = P.du[i - 1] + P.d[i]
    end

    return D
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

    # TODO: Shall we require the users to set unused entries to zero?
    fill!(P, zero(eltype(P)))

    f.p(P, uprev, p, t) # evaluate production terms
    sum_destruction_terms!(D, P) # store destruction terms in D
    integrator.stats.nf += 1

    build_mprk_matrix!(P, 1, P, D, uprev, dt)
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
    M::PType
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

    M = p_prototype(u, f)
    linsolve_tmp = zero(u)
    weight = similar(u, uEltypeNoUnits)
    recursivefill!(weight, false)

    linprob = LinearProblem(M, _vec(linsolve_tmp); u0 = _vec(tmp))
    linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                    assumptions = LinearSolve.OperatorAssumptions(true))

    MPRK22Cache(u, uprev, tmp,
                zero(u), # atmp
                zero(rate_prototype), # k
                zero(rate_prototype), #fsalfirst
                p_prototype(u, f), # P
                p_prototype(u, f), # P2
                zero(u), # D
                zero(u), # D2
                M,
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
    @unpack tmp, atmp, P, P2, D, D2, M, σ, thread, weight = cache
    @unpack a21, b1, b2, c2, small_constant = cache.tab

    uprev .= uprev .+ small_constant

    f.p(P, uprev, p, t) # evaluate production terms
    sum_destruction_terms!(D, P) # store destruction terms in D
    integrator.stats.nf += 1

    build_mprk_matrix!(M, a21, P, D, uprev, dt)
    # Same as linres = M \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = M, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    u .= linres
    integrator.stats.nsolve += 1

    u .= u .+ small_constant

    σ .= uprev .* (u ./ uprev) .^ (1 / a21) .+ small_constant

    f.p(P2, u, p, t + a21 * dt) # evaluate production terms
    sum_destruction_terms!(D, P) # store destruction terms in D
    sum_destruction_terms!(D2, P2) # store destruction terms in D2

    build_mprk_matrix!(M, b1, P, D, b2, P2, D2, σ, dt)
    # Same as linres = M \ uprev
    linres = dolinsolve(integrator, cache.linsolve;
                        A = M, b = _vec(uprev),
                        du = integrator.fsalfirst, u = u, p = p, t = t,
                        weight = weight)
    u .= linres
    integrator.stats.nsolve += 1

    tmp .= u .- σ
    calculate_residuals!(atmp, tmp, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         thread)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)
end

### MPRK43 #####################################################################################
"""
    MPRK43I(α, β; [linsolve = ...])

A third-order modified Patankar-Runge-Kutta algorithm for (conservative)
production-destruction systems. This one-step, four-stage method is
third-order accurate, unconditionally positivity-preserving, and linearly
implicit. The parameters `α,β` are described by Kopecz and Meister (2018).

This modified Patankar-Runge-Kutta method requires the special structure of a
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

A third-order modified Patankar-Runge-Kutta algorithm for (conservative)
production-destruction systems. This one-step, four-stage method is
third-order accurate, unconditionally positivity-preserving, and linearly
implicit. The parameters `γ` is described by Kopecz and Meister (2018).

This modified Patankar-Runge-Kutta method requires the special structure of a
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
    D::uType
    D2::uType
    M::PType
    σ::uType
    tab::tabType
    thread::Thread
    linsolve_tmp::uType  # stores rhs of linear system
    linsolve::F
    weight::uNoUnitsType
end

function get_constant_parameters(alg::MPRK43I)
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

    return a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2
end

function set_constant_parameters(alg::MPRK43II)
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

    #TODO: Should assert alg.alpha ≠ alg.beta, alg.alpha ≠ 0, alg.beta ≠ 0, alg.alpha ≠ 2/3 
    a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2 = set_constant_parameters(alg)
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
    σ0 = σ

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
    integrator.stats.nf += 1

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

    # copied from perform_step for HeunConstantCache
    # If a21 = 1.0, then σ is the MPE approximation and thus suited for stiff problems.
    # If a21 ≠ 1.0, σ might be a bad choice to estimate errors.
    tmp = u - σ
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)

    integrator.u = u
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
