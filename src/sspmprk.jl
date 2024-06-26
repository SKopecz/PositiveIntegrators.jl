### SSPMPRK #####################################################################################
"""
    SSPMPRK22(α, β; [linsolve = ...])

A family of second-order modified Patankar-Runge-Kutta algorithms for 
production-destruction systems. Each member of this family is an one-step, two-stage method which is
second-order accurate, unconditionally positivity-preserving, and linearly
implicit. The parameters `α` and `β` are described by Huang and Shu (2019) and
studied by Huang, Izgin, Kopecz, Meister and Shu (2023). 
The difference to [`MPRK22`](@ref) is that this method is based on the SSP formulation of 
an explicit second-order Runge-Kutta method. This family of schemes contains the [`MPRK22`](@ref) family,
where `MPRK22(α) = SSMPRK22(0, α)` applies.

The scheme was introduced by Huang and Shu for conservative production-destruction systems. 
For nonconservative production–destruction systems we use the straight forward extension
analogous to [`MPE`](@ref).

This modified Patankar-Runge-Kutta method requires the special structure of a
[`PDSProblem`](@ref) or a [`ConservativePDSProblem`](@ref).

You can optionally choose the linear solver to be used by passing an
algorithm from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
as keyword argument `linsolve`.

## References

- Juntao Huang and Chi-Wang Shu.
  "Positivity-Preserving Time Discretizations for Production–Destruction Equations 
  with Applications to Non-equilibrium Flows."
  Journal of Scientific Computing 78 (2019): 1811–1839
  [DOI: 10.1007/s10915-018-0852-1](https://doi.org/10.1007/s10915-018-0852-1)
- Juntao Huang, Thomas Izgin, Stefan Kopecz, Andreas Meister and Chi-Wang Shu.
  "On the stability of strong-stability-preserving modified Patankar-Runge-Kutta schemes."
  ESAIM: Mathematical Modelling and Numerical Analysis 57 (2023):1063–1086
  [DOI: 10.1051/m2an/2023005](https://doi.org/10.1051/m2an/2023005)
"""
struct SSPMPRK22{T, F} <: OrdinaryDiffEqAdaptiveAlgorithm
    alpha::T
    beta::T
    linsolve::F
end

function SSPMPRK22(alpha, beta; linsolve = LUFactorization())
    SSPMPRK22{typeof(alpha), typeof(linsolve)}(alpha, beta, linsolve)
end

alg_order(::SSPMPRK22) = 2
isfsal(::SSPMPRK22) = false

function get_constant_parameters(alg::SSPMPRK22)
    if !((0 ≤ alg.alpha ≤ 1) && (alg.beta ≥ 0) &&
         (alg.alpha * alg.beta + 1 / (2 * alg.beta) ≤ 1))
        throw(ArgumentError("SSPMPRK22 requires 0 ≤ α ≤ 1, β ≥ 0 and αβ + 1/(2β) ≤ 1."))
    end

    a21 = alg.alpha
    a10 = one(alg.alpha)
    a20 = 1 - a21

    b10 = alg.beta
    b20 = 1 - 1 / (2 * b10) - a21 * b10
    b21 = 1 / (2 * b10)

    s = (b20 + b21 + a21 * b10^2) / (b10 * (b20 + b21))

    # This should never happen
    if !all((a21, a10, a20, b10, b20, b21) .≥ 0)
        throw(ArgumentError("SSPMPRK22 requires nonnegative SSP coefficients."))
    end
    return a21, a10, a20, b10, b20, b21, s
end

struct SSPMPRK22ConstantCache{T} <: OrdinaryDiffEqConstantCache
    a21::T
    a10::T
    a20::T
    b10::T
    b20::T
    b21::T
    s::T
    small_constant::T
end

# Out-of-place
function alg_cache(alg::SSPMPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    if !(f isa PDSFunction || f isa ConservativePDSFunction)
        throw(ArgumentError("SSPMPRK22 can only be applied to production-destruction systems"))
    end

    a21, a10, a20, b10, b20, b21, s = get_constant_parameters(alg)
    SSPMPRK22ConstantCache(a21, a10, a20, b10, b20, b21, s, floatmin(uEltypeNoUnits))
end

function initialize!(integrator, cache::SSPMPRK22ConstantCache)
end

function perform_step!(integrator, cache::SSPMPRK22ConstantCache, repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack a21, a10, a20, b10, b20, b21, s, small_constant = cache

    f = integrator.f

    # evaluate production matrix
    P = f.p(uprev, p, t)
    Ptmp = b10 * P
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)

    # build linear system matrix and rhs
    if f isa PDSFunction
        d = f.d(uprev, p, t)  # evaluate nonconservative destruction terms
        dtmp = b10 * d
        rhs = a10 * uprev + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
    else
        # f isa ConservativePDSFunction
        M = build_mprk_matrix(Ptmp, σ, dt)
        rhs = a10 * uprev
    end

    # solve linear system
    linprob = LinearProblem(M, rhs)
    sol = solve(linprob, alg.linsolve)
    u = sol.u
    integrator.stats.nsolve += 1

    # compute Patankar weight denominator
    if isone(s)
        σ = u
    else
        σ = σ .^ (1 - s) .* u .^ s
    end
    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(σ, small_constant)

    P2 = f.p(u, p, t + a21 * dt)
    Ptmp = b20 * P + b21 * P2
    integrator.stats.nf += 1

    # build linear system matrix and rhs
    if f isa PDSFunction
        d2 = f.d(u, p, t + a21 * dt)  # evaluate nonconservative destruction terms
        dtmp = b20 * d + b21 * d2
        rhs = a20 * uprev + a21 * u + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
    else
        # f isa ConservativePDSFunction
        M = build_mprk_matrix(Ptmp, σ, dt)
        rhs = a20 * uprev + a21 * u
    end

    # solve linear system
    linprob = LinearProblem(M, rhs)
    sol = solve(linprob, alg.linsolve)
    u = sol.u
    integrator.stats.nsolve += 1

    # If a21 = 0 or b10 = 0, then σ is a first order approximation of the solution and
    # can be used for error estimation. 
    # TODO: Find first order approximation, if a21*b10 ≠ 0.
    tmp = u - σ
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)

    integrator.u = u
end

struct SSPMPRK22Cache{uType, PType, tabType, F} <:
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

struct SSPMPRK22ConservativeCache{uType, PType, tabType, F} <:
       OrdinaryDiffEqMutableCache
    tmp::uType
    P::PType
    P2::PType
    σ::uType
    tab::tabType
    linsolve::F
end

# In-place
function alg_cache(alg::SSPMPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    a21, a10, a20, b10, b20, b21, s = get_constant_parameters(alg)
    tab = SSPMPRK22ConstantCache(a21, a10, a20, b10, b20, b21, s, floatmin(uEltypeNoUnits))
    tmp = zero(u)
    P = p_prototype(u, f)
    # We use P2 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    P2 = p_prototype(u, f)
    σ = zero(u)

    if f isa ConservativePDSFunction
        linprob = LinearProblem(P2, _vec(tmp))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        SSPMPRK22ConservativeCache(tmp, P, P2, σ,
                                   tab, #MPRK22ConstantCache
                                   linsolve)
    elseif f isa PDSFunction
        linprob = LinearProblem(P2, _vec(tmp))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        SSPMPRK22Cache(tmp, P, P2,
                       zero(u), # D
                       zero(u), # D2
                       σ,
                       tab, #MPRK22ConstantCache 
                       linsolve)
    else
        throw(ArgumentError("MPRK22 can only be applied to production-destruction systems"))
    end
end

function initialize!(integrator, cache::Union{SSPMPRK22Cache, SSPMPRK22ConservativeCache})
end

function perform_step!(integrator, cache::SSPMPRK22Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, P, P2, D, D2, σ, linsolve = cache
    @unpack a21, a10, a20, b10, b20, b21, s, small_constant = cache.tab

    # We use P2 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system

    f.p(P, uprev, p, t) # evaluate production terms
    f.d(D, uprev, p, t) # evaluate nonconservative destruction terms
    integrator.stats.nf += 1
    @.. broadcast=false P2=b10 * P
    @.. broadcast=false D2=b10 * D

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    # tmp holds the right hand side of the linear system
    @.. broadcast=false tmp=a10 * uprev
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P2[i, i]
    end

    build_mprk_matrix!(P2, P2, σ, dt, D2)

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    if isone(s)
        σ .= u
    else
        @.. broadcast=false σ=σ^(1 - s) * u^s
    end
    @.. broadcast=false σ=σ + small_constant

    f.p(P2, u, p, t + a21 * dt) # evaluate production terms
    f.d(D2, u, p, t + a21 * dt) # evaluate nonconservative destruction terms
    integrator.stats.nf += 1

    @.. broadcast=false P2=b20 * P + b21 * P2
    @.. broadcast=false D2=b20 * D + b21 * D2

    # tmp holds the right hand side of the linear system
    @.. broadcast=false tmp=a20 * uprev + a21 * u
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P2[i, i]
    end

    build_mprk_matrix!(P2, P2, σ, dt, D2)

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    # If a21 = 0 or b10 = 0, then σ is a first order approximation of the solution and
    # can be used for error estimation. 
    # TODO: Find first order approximation, if a21*b10 ≠ 0.

    # Now σ stores the error estimate
    @.. broadcast=false σ=u - σ

    # Now tmp stores error residuals
    calculate_residuals!(tmp, σ, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp, t)
end

function perform_step!(integrator, cache::SSPMPRK22ConservativeCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, P, P2, σ, linsolve = cache
    @unpack a21, a10, a20, b10, b20, b21, s, small_constant = cache.tab

    # We use P2 to store the last evaluation of the PDS 
    # as well as to store the system matrix of the linear system
    f.p(P, uprev, p, t) # evaluate production terms
    integrator.stats.nf += 1
    @.. broadcast=false P2=b10 * P

    # Avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    # tmp holds the right hand side of the linear system
    @.. broadcast=false tmp=a10 * uprev

    build_mprk_matrix!(P2, P2, σ, dt)

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    if isone(s)
        σ .= u
    else
        @.. broadcast=false σ=σ^(1 - s) * u^s
    end
    @.. broadcast=false σ=σ + small_constant

    f.p(P2, u, p, t + a21 * dt) # evaluate production terms
    integrator.stats.nf += 1

    @.. broadcast=false P2=b20 * P + b21 * P2

    @.. broadcast=false tmp=a20 * uprev + a21 * u

    build_mprk_matrix!(P2, P2, σ, dt)

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    # If a21 = 0 or b10 = 0, then σ is a first order approximation of the solution and
    # can be used for error estimation. 
    # TODO: Find first order approximation, if a21*b10 ≠ 0.
    # Now σ stores the error estimate
    @.. broadcast=false σ=u - σ

    # Now tmp stores error residuals
    calculate_residuals!(tmp, σ, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp, t)
end
