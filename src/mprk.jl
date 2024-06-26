# Helper functions
add_small_constant(v, small_constant) = v .+ small_constant

function add_small_constant(v::SVector{N, T}, small_constant::T) where {N, T}
    v + SVector{N, T}(ntuple(i -> small_constant, N))
end

#####################################################################
p_prototype(u, f) = zeros(eltype(u), length(u), length(u))
p_prototype(u, f::ConservativePDSFunction) = zero(f.p_prototype)
p_prototype(u, f::PDSFunction) = zero(f.p_prototype)

#####################################################################
# out-of-place for dense and static arrays
function build_mprk_matrix(P, sigma, dt, d = nothing)
    # re-use the in-place version implemented below
    M = similar(P)
    build_mprk_matrix!(M, P, sigma, dt, d)

    if P isa StaticArray
        return SMatrix(M)
    else
        return M
    end
end

# in-place for dense arrays
function build_mprk_matrix!(M, P, sigma, dt, d = nothing)
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
function build_mprk_matrix!(M::Tridiagonal, P::Tridiagonal, σ, dt, d = nothing)
    # M[i,i] = (sigma[i] + dt * sum_j P[j,i]) / sigma[i]
    # M[i,j] = -dt * P[i,j] / sigma[j]
    Base.require_one_based_indexing(M.dl, M.d, M.du,
                                    P.dl, P.d, P.du, σ)
    @assert length(M.dl) + 1 == length(M.d) == length(M.du) + 1 ==
            length(P.dl) + 1 == length(P.d) == length(P.du) + 1 == length(σ)

    if !isnothing(d)
        Base.require_one_based_indexing(d)
        @assert length(σ) == length(d)
    end

    if isnothing(d)
        for i in eachindex(M.d, σ)
            M.d[i] = σ[i]
        end
    else
        for i in eachindex(M.d, σ, d)
            M.d[i] = σ[i] + dt * d[i]
        end
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

The first-order modified Patankar-Euler algorithm for production-destruction systems. This one-step, one-stage method is
first-order accurate, unconditionally positivity-preserving, and
linearly implicit.

The scheme was introduced by Burchard et al for conservative production-destruction systems. 
For nonconservative production–destruction systems we use the straight forward extension

``u_i^{n+1} = u_i^n + Δt \\sum_{j, j≠i} \\biggl(p_{ij}^n \\frac{u_j^{n+1}}{u_j^n}-d_{ij}^n \\frac{u_i^{n+1}}{u_i^n}\\biggr) + {\\Delta}t p_{ii}^n - Δt d_{ii}^n\\frac{u_i^{n+1}}{u_i^n}``.

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

alg_order(::MPE) = 1
isfsal(::MPE) = false

struct MPEConstantCache{T} <: OrdinaryDiffEqConstantCache
    small_constant::T
end

# Out-of-place
function alg_cache(alg::MPE, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    if !(f isa PDSFunction || f isa ConservativePDSFunction)
        throw(ArgumentError("MPE can only be applied to production-destruction systems"))
    end
    MPEConstantCache(floatmin(uEltypeNoUnits))
end

function initialize!(integrator, cache::MPEConstantCache)
end

function perform_step!(integrator, cache::MPEConstantCache, repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack small_constant = cache

    f = integrator.f

    # evaluate production matrix
    P = f.p(uprev, p, t)
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)

    # build linear system matrix and rhs
    if f isa PDSFunction
        d = f.d(uprev, p, t)  # evaluate nonconservative destruction terms
        rhs = uprev + dt * diag(P)
        M = build_mprk_matrix(P, σ, dt, d)
        linprob = LinearProblem(M, rhs)
    else
        # f isa ConservativePDSFunction
        M = build_mprk_matrix(P, σ, dt)
        linprob = LinearProblem(M, uprev)
    end

    # solve linear system
    sol = solve(linprob, alg.linsolve)
    u = sol.u
    integrator.stats.nsolve += 1

    integrator.u = u
end

struct MPECache{PType, uType, tabType, F} <: OrdinaryDiffEqMutableCache
    P::PType
    D::uType
    σ::uType
    tab::tabType
    linsolve_rhs::uType  # stores rhs of linear system
    linsolve::F
end

struct MPEConservativeCache{PType, uType, tabType, F} <: OrdinaryDiffEqMutableCache
    P::PType
    σ::uType
    tab::tabType
    linsolve::F
end

# In-place
function alg_cache(alg::MPE, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    P = p_prototype(u, f)
    σ = zero(u)
    tab = MPEConstantCache(floatmin(uEltypeNoUnits))

    if f isa ConservativePDSFunction
        # We use P to store the evaluation of the PDS 
        # as well as to store the system matrix of the linear system
        # Right hand side of linear system is always uprev
        linprob = LinearProblem(P, _vec(uprev))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        MPEConservativeCache(P, σ, tab, linsolve)
    elseif f isa PDSFunction
        linsolve_rhs = zero(u)
        # We use P to store the evaluation of the PDS 
        # as well as to store the system matrix of the linear system
        linprob = LinearProblem(P, _vec(linsolve_rhs))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        MPECache(P, zero(u), σ, tab, linsolve_rhs, linsolve)
    else
        throw(ArgumentError("MPE can only be applied to production-destruction systems"))
    end
end

function initialize!(integrator, cache::Union{MPECache, MPEConservativeCache})
end

function perform_step!(integrator, cache::MPECache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack P, D, σ, linsolve, linsolve_rhs = cache
    @unpack small_constant = cache.tab

    # We use P to store the evaluation of the PDS 
    # as well as to store the system matrix of the linear system  

    # We require the users to set unused entries to zero!

    f.p(P, uprev, p, t) # evaluate production terms
    f.d(D, uprev, p, t) # evaluate nonconservative destruction terms
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    linsolve_rhs .= uprev
    @inbounds for i in eachindex(linsolve_rhs)
        linsolve_rhs[i] += dt * P[i, i]
    end

    build_mprk_matrix!(P, P, σ, dt, D)

    # Same as linres = P \ linsolve_rhs
    linsolve.A = P
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1
end

function perform_step!(integrator, cache::MPEConservativeCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack P, σ, linsolve = cache
    @unpack small_constant = cache.tab

    # We use P to store the evaluation of the PDS 
    # as well as to store the system matrix of the linear system  

    # We require the users to set unused entries to zero!

    f.p(P, uprev, p, t) # evaluate production terms
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    build_mprk_matrix!(P, P, σ, dt)

    # Same as linres = P \ uprev
    linsolve.A = P
    linres = solve!(linsolve)

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

The second-order modified Patankar-Runge-Kutta algorithm for 
production-destruction systems. This one-step, two-stage method is
second-order accurate, unconditionally positivity-preserving, and linearly
implicit. The parameter `α` is described by Kopecz and Meister (2018) and
studied by Izgin, Kopecz and Meister (2022) as well as
Torlo, Öffner and Ranocha (2022).

The scheme was introduced by Kopecz and Meister for conservative production-destruction systems. 
For nonconservative production–destruction systems we use the straight forward extension
analogous to [`MPE`](@ref).

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
struct MPRK22{T, F} <: OrdinaryDiffEqAdaptiveAlgorithm
    alpha::T
    linsolve::F
end

function MPRK22(alpha; linsolve = LUFactorization())
    MPRK22{typeof(alpha), typeof(linsolve)}(alpha, linsolve)
end

alg_order(::MPRK22) = 2
isfsal(::MPRK22) = false

function get_constant_parameters(alg::MPRK22)
    if !(alg.alpha ≥ 1 / 2)
        throw(ArgumentError("MPRK22 requires α ≥ 1/2."))
    end

    a21 = alg.alpha
    b2 = 1 / (2 * a21)
    b1 = 1 - b2

    # This should never happen
    if !all((a21, b1, b2) .≥ 0)
        throw(ArgumentError("MPRK22 requires nonnegative RK coefficients."))
    end
    return a21, b1, b2
end

struct MPRK22ConstantCache{T} <: OrdinaryDiffEqConstantCache
    a21::T
    b1::T
    b2::T
    small_constant::T
end

# Out-of-place
function alg_cache(alg::MPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    if !(f isa PDSFunction || f isa ConservativePDSFunction)
        throw(ArgumentError("MPRK22 can only be applied to production-destruction systems"))
    end

    a21, b1, b2 = get_constant_parameters(alg)
    MPRK22ConstantCache(a21, b1, b2, floatmin(uEltypeNoUnits))
end

function initialize!(integrator, cache::MPRK22ConstantCache)
end

function perform_step!(integrator, cache::MPRK22ConstantCache, repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack a21, b1, b2, small_constant = cache

    f = integrator.f

    # evaluate production matrix
    P = f.p(uprev, p, t)
    Ptmp = a21 * P
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)

    # build linear system matrix and rhs
    if f isa PDSFunction
        d = f.d(uprev, p, t)  # evaluate nonconservative destruction terms
        dtmp = a21 * d
        rhs = uprev + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
        linprob = LinearProblem(M, rhs)
    else
        # f isa ConservativePDSFunction
        M = build_mprk_matrix(Ptmp, σ, dt)
        linprob = LinearProblem(M, uprev)
    end

    # solve linear system
    sol = solve(linprob, alg.linsolve)
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

    # build linear system matrix and rhs
    if f isa PDSFunction
        d2 = f.d(u, p, t + a21 * dt)  # evaluate nonconservative destruction terms
        dtmp = b1 * d + b2 * d2
        rhs = uprev + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
        linprob = LinearProblem(M, rhs)
    else
        # f isa ConservativePDSFunction
        M = build_mprk_matrix(Ptmp, σ, dt)
        linprob = LinearProblem(M, uprev)
    end

    # solve linear system
    sol = solve(linprob, alg.linsolve)
    u = sol.u
    integrator.stats.nsolve += 1

    # If a21 = 1.0, then σ is the MPE approximation, i.e. a first order approximation
    # of the solution, and can be used for error estimation. Moreover, MPE is suited for stiff problems.
    # TODO: Find first order approximation if a21≠ 1.0.
    tmp = u - σ
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)

    integrator.u = u
end

struct MPRK22Cache{uType, PType, tabType, F} <:
       OrdinaryDiffEqMutableCache
    tmp::uType
    P::PType
    P2::PType
    D::uType
    D2::uType
    σ::uType
    tab::tabType
    linsolve::F
end

struct MPRK22ConservativeCache{uType, PType, tabType, F} <:
       OrdinaryDiffEqMutableCache
    tmp::uType
    P::PType
    P2::PType
    σ::uType
    tab::tabType
    linsolve::F
end

# In-place
function alg_cache(alg::MPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    a21, b1, b2 = get_constant_parameters(alg)
    tab = MPRK22ConstantCache(a21, b1, b2, floatmin(uEltypeNoUnits))
    tmp = zero(u)
    P = p_prototype(u, f)
    # We use P2 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    P2 = p_prototype(u, f)
    σ = zero(u)

    if f isa ConservativePDSFunction
        # The right hand side of the linear system is always uprev. But using
        # tmp instead of uprev for the rhs we allow `alias_b=true`. uprev must
        # not be altered, since it is needed to compute the adaptive time step
        # size. 
        linprob = LinearProblem(P2, _vec(tmp))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        MPRK22ConservativeCache(tmp, P, P2, σ,
                                tab, #MPRK22ConstantCache
                                linsolve)
    elseif f isa PDSFunction
        linprob = LinearProblem(P2, _vec(tmp))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        MPRK22Cache(tmp, P, P2,
                    zero(u), # D
                    zero(u), # D2
                    σ,
                    tab, #MPRK22ConstantCache 
                    linsolve)
    else
        throw(ArgumentError("MPRK22 can only be applied to production-destruction systems"))
    end
end

function initialize!(integrator, cache::Union{MPRK22Cache, MPRK22ConservativeCache})
end

function perform_step!(integrator, cache::MPRK22Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, P, P2, D, D2, σ, linsolve = cache
    @unpack a21, b1, b2, small_constant = cache.tab

    # We use P2 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system

    f.p(P, uprev, p, t) # evaluate production terms
    f.d(D, uprev, p, t) # evaluate nonconservative destruction terms
    integrator.stats.nf += 1
    @.. broadcast=false P2=a21 * P
    @.. broadcast=false D2=a21 * D

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    # tmp holds the right hand side of the linear system
    tmp .= uprev
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P2[i, i]
    end

    build_mprk_matrix!(P2, P2, σ, dt, D2)

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    if isone(a21)
        σ .= u
    else
        @.. broadcast=false σ=σ^(1 - 1 / a21) * u^(1 / a21)
    end
    @.. broadcast=false σ=σ + small_constant

    f.p(P2, u, p, t + a21 * dt) # evaluate production terms
    f.d(D2, u, p, t + a21 * dt) # evaluate nonconservative destruction terms
    integrator.stats.nf += 1

    @.. broadcast=false P2=b1 * P + b2 * P2
    @.. broadcast=false D2=b1 * D + b2 * D2

    # tmp holds the right hand side of the linear system
    tmp .= uprev
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P2[i, i]
    end

    build_mprk_matrix!(P2, P2, σ, dt, D2)

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    # Now σ stores the error estimate
    # If a21 = 1.0, then σ is the MPE approximation, i.e. a first order approximation
    # of the solution, and can be used for error estimation. Moreover, MPE is suited for stiff problems.
    # TODO: Find first order approximation if a21≠ 1.0.
    @.. broadcast=false σ=u - σ

    # Now tmp stores error residuals
    calculate_residuals!(tmp, σ, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp, t)
end

function perform_step!(integrator, cache::MPRK22ConservativeCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, P, P2, σ, linsolve = cache
    @unpack a21, b1, b2, small_constant = cache.tab

    # Set right hand side of linear system
    tmp .= uprev

    # We use P2 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    f.p(P, uprev, p, t) # evaluate production terms
    integrator.stats.nf += 1
    @.. broadcast=false P2=a21 * P

    # Avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    build_mprk_matrix!(P2, P2, σ, dt)

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

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

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    # Now σ stores the error estimate
    # If a21 = 1.0, then σ is the MPE approximation, i.e. a first order approximation    
    # of the solution, and can be used for error estimation. Moreover, MPE is suited for stiff problems.
    # TODO: Find first order approximation if a21≠ 1.0.
    @.. broadcast=false σ=u - σ

    # Now tmp stores error residuals
    calculate_residuals!(tmp, σ, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp, t)
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

The scheme was introduced by Kopecz and Meister for conservative production-destruction systems. 
For nonconservative production–destruction systems we use the straight forward extension
analogous to [`MPE`](@ref).

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
struct MPRK43I{T, F} <: OrdinaryDiffEqAdaptiveAlgorithm
    alpha::T
    beta::T
    linsolve::F
end

function MPRK43I(alpha, beta; linsolve = LUFactorization())
    MPRK43I{typeof(alpha), typeof(linsolve)}(alpha, beta, linsolve)
end

alg_order(::MPRK43I) = 3
isfsal(::MPRK43I) = false

function get_constant_parameters(alg::MPRK43I)
    if !(alg.alpha ≥ 1 / 3 && alg.alpha ≠ 2 / 3)
        throw(ArgumentError("MPRK43I requires α ≥ 1/3 and α ≠ 2/3."))
    end
    α0 = 1 / 6 * (3 + (3 - 2 * sqrt(2))^(1 / 3) + (3 + 2 * sqrt(2))^(1 / 3))
    if 1 / 3 ≤ alg.alpha < 2 / 3
        if !(2 / 3 ≤ alg.beta ≤ 3 * alg.alpha * (1 - alg.alpha))
            throw(ArgumentError("For this choice of α MPRK43I requires 2/3 ≤ β ≤ 3α(1-α)."))
        end
    elseif 2 / 3 < alg.alpha ≤ α0
        if !(3 * alg.alpha * (1 - alg.alpha) ≤ alg.beta ≤ 2 / 3)
            throw(ArgumentError("For this choice of α MPRK43I requires 3α(1-α) ≤ β ≤ 2/3."))
        end
    else
        if !((3 * alg.alpha - 2) / (6 * alg.alpha - 3) ≤ alg.beta ≤ 2 / 3)
            throw(ArgumentError("For this choice of α MPRK43I requires (3α-2)/(6α-3) ≤ β ≤ 2/3."))
        end
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

    #This should never happen
    if !all((a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2) .≥ 0)
        throw(ArgumentError("MPRK43I requires nonnegative RK coefficients."))
    end
    return a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2
end

"""
    MPRK43II(γ; [linsolve = ...])

A family of third-order modified Patankar-Runge-Kutta schemes for (conservative)
production-destruction systems, which is based on the one-parameter family of third order explicit Runge--Kutta schemes with 
non-negative Runge--Kutta coefficients.
Each member of this family is a one-step method with four stages which is
third-order accurate, unconditionally positivity-preserving, conservative and linearly
implicit. In this implementation the stage-values are conservative as well. The parameter `γ` must satisfy
`3/8 ≤ γ ≤ 3/4`. 
Further details are given in Kopecz and Meister (2018).  

The scheme was introduced by Kopecz and Meister for conservative production-destruction systems. 
For nonconservative production–destruction systems we use the straight forward extension
analogous to [`MPE`](@ref).

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
struct MPRK43II{T, F} <: OrdinaryDiffEqAdaptiveAlgorithm
    gamma::T
    linsolve::F
end

function MPRK43II(gamma; linsolve = LUFactorization())
    MPRK43II{typeof(gamma), typeof(linsolve)}(gamma, linsolve)
end

alg_order(::MPRK43II) = 3
isfsal(::MPRK43II) = false

function get_constant_parameters(alg::MPRK43II)
    if !(3 / 8 ≤ alg.gamma ≤ 3 / 4)
        throw(ArgumentError("MPRK43II requires 3/8 ≤ γ ≤ 3/4."))
    end

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

    #This should never happen
    if !all((a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2) .≥ 0)
        throw(ArgumentError("MPRK43II requires nonnegative RK coefficients."))
    end
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

# Out-of-place
function alg_cache(alg::Union{MPRK43I, MPRK43II}, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    if !(f isa PDSFunction || f isa ConservativePDSFunction)
        throw(ArgumentError("MPRK43 can only be applied to production-destruction systems"))
    end
    a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2 = get_constant_parameters(alg)
    MPRK43ConstantCache(a21, a31, a32, b1, b2, b3, c2, c3,
                        beta1, beta2, q1, q2, floatmin(uEltypeNoUnits))
end

function initialize!(integrator, cache::MPRK43ConstantCache)
end

function perform_step!(integrator, cache::MPRK43ConstantCache, repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2, small_constant = cache

    f = integrator.f

    # evaluate production matrix
    P = f.p(uprev, p, t)
    Ptmp = a21 * P
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)
    if !(q1 ≈ q2)
        σ0 = σ
    end

    # build linear system matrix and rhs
    if f isa PDSFunction
        d = f.d(uprev, p, t)
        dtmp = a21 * d
        rhs = uprev + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
        linprob = LinearProblem(M, rhs)
    else
        M = build_mprk_matrix(Ptmp, σ, dt)
        linprob = LinearProblem(M, uprev)
    end

    # solve linear system
    sol = solve(linprob, alg.linsolve)
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

    # build linear system matrix and rhs
    if f isa PDSFunction
        d2 = f.d(u, p, t + c2 * dt)  # evaluate nonconservative destruction terms
        dtmp = a31 * d + a32 * d2
        rhs = uprev + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
        linprob = LinearProblem(M, rhs)
    else
        M = build_mprk_matrix(Ptmp, σ, dt)
        linprob = LinearProblem(M, uprev)
    end

    # solve linear system
    sol = solve(linprob, alg.linsolve)
    u = sol.u
    integrator.stats.nsolve += 1

    # compute Patankar weight denominator
    if !(q1 ≈ q2)
        σ = σ0 .^ (1 - q2) .* u2 .^ q2

        # avoid division by zero due to zero Patankar weights
        σ = add_small_constant(σ, small_constant)
    end

    Ptmp = beta1 * P + beta2 * P2

    # build linear system matrix and rhs
    if f isa PDSFunction
        dtmp = beta1 * d + beta2 * d2
        rhs = uprev + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
        linprob = LinearProblem(M, rhs)
    else
        M = build_mprk_matrix(Ptmp, σ, dt)
        linprob = LinearProblem(M, uprev)
    end

    # solve linear system
    sol = solve(linprob, alg.linsolve)
    σ = sol.u
    integrator.stats.nsolve += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(σ, small_constant)

    P3 = f.p(u, p, t + c2 * dt)
    Ptmp = b1 * P + b2 * P2 + b3 * P3
    integrator.stats.nf += 1

    # build linear system matrix
    if f isa PDSFunction
        d3 = f.d(u, p, t + c2 * dt)  # evaluate nonconservative destruction terms
        dtmp = b1 * d + b2 * d2 + b3 * d3
        rhs = uprev + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
        linprob = LinearProblem(M, rhs)
    else
        M = build_mprk_matrix(Ptmp, σ, dt)
        linprob = LinearProblem(M, uprev)
    end

    # solve linear system
    sol = solve(linprob, alg.linsolve)
    u = sol.u
    integrator.stats.nsolve += 1

    tmp = u - σ
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)

    integrator.u = u
end

struct MPRK43Cache{uType, PType, tabType, F} <: OrdinaryDiffEqMutableCache
    tmp::uType
    tmp2::uType
    P::PType
    P2::PType
    P3::PType
    D::uType
    D2::uType
    D3::uType
    σ::uType
    tab::tabType
    linsolve::F
end

struct MPRK43ConservativeCache{uType, PType, tabType, F} <: OrdinaryDiffEqMutableCache
    tmp::uType
    tmp2::uType
    P::PType
    P2::PType
    P3::PType
    σ::uType
    tab::tabType
    linsolve::F
end

# In-place
function alg_cache(alg::Union{MPRK43I, MPRK43II}, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2 = get_constant_parameters(alg)
    tab = MPRK43ConstantCache(a21, a31, a32, b1, b2, b3, c2, c3,
                              beta1, beta2, q1, q2, floatmin(uEltypeNoUnits))
    tmp = zero(u)
    tmp2 = zero(u)
    P = p_prototype(u, f)
    P2 = p_prototype(u, f)
    # We use P3 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    P3 = p_prototype(u, f)
    σ = zero(u)

    if f isa ConservativePDSFunction
        # The right hand side of the linear system is always uprev. But using
        # tmp instead of uprev for the rhs we allow `alias_b=true`. uprev must
        # not be altered, since it is needed to compute the adaptive time step
        # size. 
        linprob = LinearProblem(P3, _vec(tmp))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))
        MPRK43ConservativeCache(tmp, tmp2, P, P2, P3, σ, tab, linsolve)
    elseif f isa PDSFunction
        D = zero(u)
        D2 = zero(u)
        D3 = zero(u)

        linprob = LinearProblem(P3, _vec(tmp))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        MPRK43Cache(tmp, tmp2, P, P2, P3, D, D2, D3, σ, tab, linsolve)
    else
        throw(ArgumentError("MPRK43 can only be applied to production-destruction systems"))
    end
end

function initialize!(integrator, cache::Union{MPRK43Cache, MPRK43ConservativeCache})
end

function perform_step!(integrator, cache::MPRK43Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, tmp2, P, P2, P3, D, D2, D3, σ, linsolve = cache
    @unpack a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2, small_constant = cache.tab

    # We use P3 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system

    f.p(P, uprev, p, t) # evaluate production terms
    f.d(D, uprev, p, t) # evaluate nonconservative destruction terms
    @.. broadcast=false P3=a21 * P
    @.. broadcast=false D3=a21 * D
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    # tmp holds the right hand side of the linear system
    tmp .= uprev
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P3[i, i]
    end

    build_mprk_matrix!(P3, P3, σ, dt, D3)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    if !(q1 ≈ q2)
        tmp2 .= u #u2 in out-of-place version
    end
    integrator.stats.nsolve += 1

    @.. broadcast=false σ=σ^(1 - q1) * u^q1
    @.. broadcast=false σ=σ + small_constant

    f.p(P2, u, p, t + c2 * dt) # evaluate production terms
    f.d(D2, u, p, t + c2 * dt) # evaluate nonconservative destruction terms
    @.. broadcast=false P3=a31 * P + a32 * P2
    @.. broadcast=false D3=a31 * D + a32 * D2
    integrator.stats.nf += 1

    # tmp holds the right hand side of the linear system
    tmp .= uprev
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P3[i, i]
    end

    build_mprk_matrix!(P3, P3, σ, dt, D3)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    if !(q1 ≈ q2)
        @.. broadcast=false σ=(uprev + small_constant)^(1 - q2) * tmp2^q2
        @.. broadcast=false σ=σ + small_constant
    end

    @.. broadcast=false P3=beta1 * P + beta2 * P2
    @.. broadcast=false D3=beta1 * D + beta2 * D2

    # tmp holds the right hand side of the linear system
    tmp .= uprev
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P3[i, i]
    end

    build_mprk_matrix!(P3, P3, σ, dt, D3)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)
    integrator.stats.nsolve += 1

    σ .= linres
    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=σ + small_constant

    f.p(P3, u, p, t + c3 * dt) # evaluate production terms
    f.d(D3, u, p, t + c3 * dt) # evaluate nonconservative destruction terms
    @.. broadcast=false P3=b1 * P + b2 * P2 + b3 * P3
    @.. broadcast=false D3=b1 * D + b2 * D2 + b3 * D3
    integrator.stats.nf += 1

    # tmp holds the right hand side of the linear system
    tmp .= uprev
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P3[i, i]
    end

    build_mprk_matrix!(P3, P3, σ, dt, D3)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    # Now tmp stores the error estimate
    @.. broadcast=false tmp=u - σ

    # Now tmp2 stores error residuals
    calculate_residuals!(tmp2, tmp, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp2, t)
end

function perform_step!(integrator, cache::MPRK43ConservativeCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, tmp2, P, P2, P3, σ, linsolve = cache
    @unpack a21, a31, a32, b1, b2, b3, c2, c3, beta1, beta2, q1, q2, small_constant = cache.tab

    # Set right hand side of linear system
    tmp .= uprev

    # We use P3 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    f.p(P, uprev, p, t) # evaluate production terms
    @.. broadcast=false P3=a21 * P
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    if !(q1 ≈ q2)
        tmp2 .= u #u2 in out-of-place version
    end
    integrator.stats.nsolve += 1

    @.. broadcast=false σ=σ^(1 - q1) * u^q1
    @.. broadcast=false σ=σ + small_constant

    f.p(P2, u, p, t + c2 * dt) # evaluate production terms
    @.. broadcast=false P3=a31 * P + a32 * P2
    integrator.stats.nf += 1

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    if !(q1 ≈ q2)
        @.. broadcast=false σ=(uprev + small_constant)^(1 - q2) * tmp2^q2
        @.. broadcast=false σ=σ + small_constant
    end

    @.. broadcast=false P3=beta1 * P + beta2 * P2

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)
    integrator.stats.nsolve += 1

    σ .= linres
    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=σ + small_constant

    f.p(P3, u, p, t + c3 * dt) # evaluate production terms
    @.. broadcast=false P3=b1 * P + b2 * P2 + b3 * P3
    integrator.stats.nf += 1

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    # Now tmp stores the error estimate
    @.. broadcast=false tmp=u - σ

    # Now tmp2 stores error residuals
    calculate_residuals!(tmp2, tmp, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp2, t)
end

########################################################################################


#######################################################################################
# interpolation specializations
function interp_summary(::Type{cacheType},
                        dense::Bool) where {
                                            cacheType <:
                                            Union{MPRK22ConstantCache, MPRK22Cache,
                                                  SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                                  MPRK43ConstantCache, MPRK43Cache}}
    "1st order linear"
end

function _ode_interpolant(Θ, dt, u0, u1, k,
                          cache::Union{MPRK22ConstantCache, MPRK22Cache,
                                       SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                       MPRK43ConstantCache, MPRK43Cache},
                          idxs, # Optionally specialize for ::Nothing and others
                          T::Type{Val{0}},
                          differential_vars::Nothing)
    linear_interpolant(Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant!(out, Θ, dt, u0, u1, k,
                           cache::Union{MPRK22ConstantCache, MPRK22Cache,
                                        SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                        MPRK43ConstantCache, MPRK43Cache},
                           idxs, # Optionally specialize for ::Nothing and others
                           T::Type{Val{0}},
                           differential_vars::Nothing)
    linear_interpolant!(out, Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant(Θ, dt, u0, u1, k,
                          cache::Union{MPRK22ConstantCache, MPRK22Cache,
                                       SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                       MPRK43ConstantCache, MPRK43Cache},
                          idxs, # Optionally specialize for ::Nothing and others
                          T::Type{Val{1}},
                          differential_vars::Nothing)
    linear_interpolant(Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant!(out, Θ, dt, u0, u1, k,
                           cache::Union{MPRK22ConstantCache, MPRK22Cache,
                                        SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                        MPRK43ConstantCache, MPRK43Cache},
                           idxs, # Optionally specialize for ::Nothing and others
                           T::Type{Val{1}},
                           differential_vars::Nothing)
    linear_interpolant!(out, Θ, dt, u0, u1, idxs, T)
end
