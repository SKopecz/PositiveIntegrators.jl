### SSPMPRK #####################################################################################
"""
    SSPMPRK22(α, β; [linsolve = ..., small_constant = ...])

A family of second-order modified Patankar-Runge-Kutta algorithms for
production-destruction systems. Each member of this family is an adaptive, one-step, two-stage method which is
second-order accurate, unconditionally positivity-preserving, and linearly
implicit. The parameters `α` and `β` are described by Huang and Shu (2019) and
studied by Huang, Izgin, Kopecz, Meister and Shu (2023).
The difference to [`MPRK22`](@ref) is that this method is based on the SSP formulation of
an explicit second-order Runge-Kutta method. This family of schemes contains the [`MPRK22`](@ref) family,
where `MPRK22(α) = SSMPRK22(0, α)` applies.

This method supports adaptive time stepping, using the first order approximations
``(σ_i - u_i^n) / τ + u_i^n`` with ``τ=1+(α_{21}β_{10}^2)/(β_{20}+β_{21})``,
see (2.7) in Huang and Shu (2019), to estimate the error.

The scheme was introduced by Huang and Shu for conservative production-destruction systems.
For nonconservative production–destruction systems we use the straight forward extension
analogous to [`MPE`](@ref).

This modified Patankar-Runge-Kutta method requires the special structure of a
[`PDSProblem`](@ref) or a [`ConservativePDSProblem`](@ref).

You can optionally choose the linear solver to be used by passing an
algorithm from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
as keyword argument `linsolve`.
You can also choose the parameter `small_constant` which is added to all Patankar-weight denominators
to avoid divisions by zero. You can pass a value explicitly, otherwise `small_constant` is set to
`floatmin` of the floating point type used.

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
struct SSPMPRK22{T, F, T2} <: OrdinaryDiffEqAdaptiveAlgorithm
    alpha::T
    beta::T
    linsolve::F
    small_constant_function::T2
end

function SSPMPRK22(alpha, beta; linsolve = LUFactorization(),
                   small_constant = nothing)
    if isnothing(small_constant)
        small_constant_function = floatmin
    elseif small_constant isa Number
        small_constant_function = Returns(small_constant)
    else # assume small_constant isa Function
        small_constant_function = small_constant
    end
    SSPMPRK22{typeof(alpha), typeof(linsolve), typeof(small_constant_function)}(alpha, beta,
                                                                                linsolve,
                                                                                small_constant_function)
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

    τ = 1 + (a21 * b10^2) / (b20 + b21)

    # This should never happen
    if any(<(0), (a21, a10, a20, b10, b20, b21))
        throw(ArgumentError("SSPMPRK22 requires nonnegative SSP coefficients."))
    end
    return a21, a10, a20, b10, b20, b21, s, τ
end

struct SSPMPRK22ConstantCache{T} <: OrdinaryDiffEqConstantCache
    a21::T
    a10::T
    a20::T
    b10::T
    b20::T
    b21::T
    s::T
    τ::T
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

    a21, a10, a20, b10, b20, b21, s, τ = get_constant_parameters(alg)
    SSPMPRK22ConstantCache(a21, a10, a20, b10, b20, b21, s, τ,
                           alg.small_constant_function(uEltypeNoUnits))
end

function initialize!(integrator, cache::SSPMPRK22ConstantCache)
end

@muladd function perform_step!(integrator, cache::SSPMPRK22ConstantCache,
                               repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack a21, a10, a20, b10, b20, b21, s, τ, small_constant = cache

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

    P2 = f.p(u, p, t + b10 * dt)
    Ptmp = b20 * P + b21 * P2
    integrator.stats.nf += 1

    # build linear system matrix and rhs
    if f isa PDSFunction
        d2 = f.d(u, p, t + b10 * dt)  # evaluate nonconservative destruction terms
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

    # Unless τ = 1, σ is not a first order approximation, since
    # σ = uprev + τ * dt *(P^n − D^n) + O(dt^2), see (2.7) in
    # https://doi.org/10.1007/s10915-018-0852-1.
    # But we can compute a 1st order approximation σ2, as follows.
    # σ2 may become negative, but still can be used for error estimation.
    σ2 = (σ - uprev) / τ + uprev

    # If a21 = 0 or b10 = 0, then σ is a first order approximation of the solution and
    # can be used for error estimation.
    # TODO: Find first order approximation, if a21*b10 ≠ 0.
    tmp = u - σ2
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)

    integrator.u = u
end

struct SSPMPRK22Cache{uType, PType, tabType, F} <: MPRKMutableCache
    tmp::uType
    P::PType
    P2::PType
    D::uType
    D2::uType
    σ::uType
    tab::tabType
    linsolve::F
end

struct SSPMPRK22ConservativeCache{uType, PType, tabType, F} <: MPRKMutableCache
    tmp::uType
    P::PType
    P2::PType
    σ::uType
    tab::tabType
    linsolve::F
end

get_tmp_cache(integrator, ::SSPMPRK22, cache::OrdinaryDiffEqMutableCache) = (cache.σ,)

# In-place
function alg_cache(alg::SSPMPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    a21, a10, a20, b10, b20, b21, s, τ = get_constant_parameters(alg)
    tab = SSPMPRK22ConstantCache(a21, a10, a20, b10, b20, b21, s, τ,
                                 alg.small_constant_function(uEltypeNoUnits))
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
                       similar(u), # D
                       similar(u), # D2
                       σ,
                       tab, #MPRK22ConstantCache
                       linsolve)
    else
        throw(ArgumentError("SSPMPRK22 can only be applied to production-destruction systems"))
    end
end

function initialize!(integrator, cache::Union{SSPMPRK22Cache, SSPMPRK22ConservativeCache})
end

@muladd function perform_step!(integrator, cache::SSPMPRK22Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, P, P2, D, D2, σ, linsolve = cache
    @unpack a21, a10, a20, b10, b20, b21, s, τ, small_constant = cache.tab

    # We use P2 to store the last evaluation of the PDS
    # as well as to store the system matrix of the linear system

    f.p(P, uprev, p, t) # evaluate production terms
    f.d(D, uprev, p, t) # evaluate nonconservative destruction terms
    integrator.stats.nf += 1
    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        @.. broadcast=false nz_P2=b10 * nz_P
    else
        @.. broadcast=false P2=b10 * P
    end
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

    f.p(P2, u, p, t + b10 * dt) # evaluate production terms
    f.d(D2, u, p, t + b10 * dt) # evaluate nonconservative destruction terms
    integrator.stats.nf += 1

    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        @.. broadcast=false nz_P2=b20 * nz_P + b21 * nz_P2
    else
        @.. broadcast=false P2=b20 * P + b21 * P2
    end
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

    # Unless τ = 1, σ is not a first order approximation, since
    # σ = uprev + τ * dt *(P^n − D^n) + O(dt^2), see (2.7) in
    # https://doi.org/10.1007/s10915-018-0852-1.
    # But we can compute a 1st order approximation as σ2 = (σ - uprev) / τ + uprev.
    # σ2 may become negative, but still can be used for error estimation.

    # Now σ stores the error estimate
    @.. broadcast=false σ=u - (σ - uprev) / τ - uprev

    # Now tmp stores error residuals
    calculate_residuals!(tmp, σ, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp, t)
end

@muladd function perform_step!(integrator, cache::SSPMPRK22ConservativeCache,
                               repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, P, P2, σ, linsolve = cache
    @unpack a21, a10, a20, b10, b20, b21, s, τ, small_constant = cache.tab

    # We use P2 to store the last evaluation of the PDS
    # as well as to store the system matrix of the linear system
    f.p(P, uprev, p, t) # evaluate production terms
    integrator.stats.nf += 1
    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        @.. broadcast=false nz_P2=b10 * nz_P
    else
        @.. broadcast=false P2=b10 * P
    end

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

    f.p(P2, u, p, t + b10 * dt) # evaluate production terms
    integrator.stats.nf += 1

    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        @.. broadcast=false nz_P2=b20 * nz_P + b21 * nz_P2
    else
        @.. broadcast=false P2=b20 * P + b21 * P2
    end

    @.. broadcast=false tmp=a20 * uprev + a21 * u

    build_mprk_matrix!(P2, P2, σ, dt)

    # Same as linres = P2 \ tmp
    linsolve.A = P2
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    # Unless τ = 1, σ is not a first order approximation, since
    # σ = uprev + τ * dt *(P^n − D^n) + O(dt^2), see (2.7) in
    # https://doi.org/10.1007/s10915-018-0852-1.
    # But we can compute a 1st order approximation as σ2 = (σ - uprev) / τ + uprev.
    # σ2 may become negative, but still can be used for error estimation.

    # Now σ stores the error estimate
    @.. broadcast=false σ=u - (σ - uprev) / τ - uprev

    # Now tmp stores error residuals
    calculate_residuals!(tmp, σ, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp, t)
end

"""
    SSPMPRK43([linsolve = ..., small_constant = ...])

A third-order modified Patankar-Runge-Kutta algorithm for
production-destruction systems. This scheme is a one-step, four-stage method which is
third-order accurate, unconditionally positivity-preserving, and linearly
implicit. The scheme is described by Huang, Zhao and Shu (2019) and
studied by Huang, Izgin, Kopecz, Meister and Shu (2023).
The difference to [`MPRK43I`](@ref) or [`MPRK43II`](@ref) is that this method is based on the SSP formulation of
an explicit third-order Runge-Kutta method.

The scheme was introduced by Huang, Zhao and Shu for conservative production-destruction systems.
For nonconservative production–destruction systems we use the straight forward extension
analogous to [`MPE`](@ref).

This modified Patankar-Runge-Kutta method requires the special structure of a
[`PDSProblem`](@ref) or a [`ConservativePDSProblem`](@ref).

You can optionally choose the linear solver to be used by passing an
algorithm from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
as keyword argument `linsolve`.
You can also choose the parameter `small_constant` which is added to all Patankar-weight denominators
to avoid divisions by zero. To display the default value for data type `type` evaluate 
`SSPMPRK43. small_constant_function(type)`, where `type` can be, e.g.,
`Float64`.

The current implementation only supports fixed time steps.

## References

- Juntao Huang, Weifeng Zhao and Chi-Wang Shu.
  "A Third-Order Unconditionally Positivity-Preserving Scheme for
  Production–Destruction Equations with Applications to Non-equilibrium Flows."
  Journal of Scientific Computing 79 (2019): 1015–1056
  [DOI: 10.1007/s10915-018-0881-9](https://doi.org/10.1007/s10915-018-0881-9)
- Juntao Huang, Thomas Izgin, Stefan Kopecz, Andreas Meister and Chi-Wang Shu.
  "On the stability of strong-stability-preserving modified Patankar-Runge-Kutta schemes."
  ESAIM: Mathematical Modelling and Numerical Analysis 57 (2023):1063–1086
  [DOI: 10.1051/m2an/2023005](https://doi.org/10.1051/m2an/2023005)
"""
struct SSPMPRK43{F, T} <: OrdinaryDiffEqAlgorithm
    linsolve::F
    small_constant_function::T
end

function small_constant_function_SSPMPRK43(type)
    if type == Float64
        small_constant = 1e-50
    elseif type == Float32
        # small_constant is chosen such that the problem below
        # (zero initial condition) can be solved
        # P_linmod(u, p, t) = [0 u[2]; 5*u[1] 0]
        # u0 = [1.0f0, 0.0f0]
        # prob = ConservativePDSProblem(P_linmod, u0, (0.0f0, 2.0f0))
        # sol = solve(prob, SSPMPRK43(); dt=0.1f0)
        small_constant = 1.0f-8
    else
        small_constant = floatmin(type)
    end
    return small_constant
end

function SSPMPRK43(; linsolve = LUFactorization(),
                   small_constant = small_constant_function_SSPMPRK43)
    if isnothing(small_constant)
        small_constant_function = floatmin
    elseif small_constant isa Number
        small_constant_function = Returns(small_constant)
    else # assume small_constant isa Function
        small_constant_function = small_constant
    end
    SSPMPRK43{typeof(linsolve), typeof(small_constant_function)}(linsolve,
                                                                 small_constant_function)
end

alg_order(::SSPMPRK43) = 3
isfsal(::SSPMPRK43) = false

function get_constant_parameters(alg::SSPMPRK43)
    # parameters from original paper

    n1 = 2.569046025732011E-01
    n2 = 7.430953974267989E-01
    z = 6.288938077828750E-01
    η1 = 3.777285888379173E-02
    η2 = 1.0 / 3.0
    η3 = 1.868649805549811E-01
    η4 = 2.224876040351123
    s = 5.721964308755304

    η5 = η3 * (η1 + η2)
    η6 = η4 * (η1 + η2)

    α10 = 1.0
    α20 = 9.2600312554031827E-01
    α21 = 7.3996874459681783E-02
    α30 = 7.0439040373427619E-01
    α31 = 2.0662904223744017E-10
    α32 = 2.9560959605909481E-01
    β10 = 4.7620819268131703E-01
    β20 = 7.7545442722396801E-02
    β21 = 5.9197500149679749E-01
    β30 = 2.0044747790361456E-01
    β31 = 6.8214380786704851E-10
    β32 = 5.9121918658514827E-01

    c3 = β20 + α21 * β10 + β21

    return n1, n2, z, η1, η2, η3, η4, η5, η6, s, α10, α20, α21, α30, α31, α32, β10, β20,
           β21, β30, β31, β32, c3
end

struct SSPMPRK43ConstantCache{T} <: OrdinaryDiffEqConstantCache
    n1::T
    n2::T
    z::T
    η1::T
    η2::T
    η3::T
    η4::T
    η5::T
    η6::T
    s::T
    α10::T
    α20::T
    α21::T
    α30::T
    α31::T
    α32::T
    β10::T
    β20::T
    β21::T
    β30::T
    β31::T
    β32::T
    c3::T
    small_constant::T
end

# Out-of-place
function alg_cache(alg::SSPMPRK43, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    if !(f isa PDSFunction || f isa ConservativePDSFunction)
        throw(ArgumentError("SSPMPRK43 can only be applied to production-destruction systems"))
    end
    const_param = get_constant_parameters(alg)
    const_param = convert.(uEltypeNoUnits, const_param)
    n1, n2, z, η1, η2, η3, η4, η5, η6, s, α10, α20, α21, α30, α31, α32, β10, β20, β21, β30, β31, β32, c3 = const_param
    small_constant = alg.small_constant_function(uEltypeNoUnits)
    SSPMPRK43ConstantCache(n1, n2, z, η1, η2, η3, η4, η5, η6, s, α10, α20, α21, α30, α31,
                           α32, β10, β20, β21, β30, β31, β32, c3, small_constant)
end

function initialize!(integrator, cache::SSPMPRK43ConstantCache)
end

@muladd function perform_step!(integrator, cache::SSPMPRK43ConstantCache,
                               repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack n1, n2, z, η1, η2, η3, η4, η5, η6, s, α10, α20, α21, α30, α31, α32, β10, β20, β21, β30, β31, β32, c3, small_constant = cache

    f = integrator.f

    # evaluate production matrix
    P = f.p(uprev, p, t)
    Ptmp = β10 * P
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(uprev, small_constant)

    # build linear system matrix and rhs
    if f isa PDSFunction
        d = f.d(uprev, p, t)
        dtmp = β10 * d
        rhs = α10 * uprev + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
    else
        rhs = α10 * uprev
        M = build_mprk_matrix(Ptmp, σ, dt)
    end

    # solve linear system
    linprob = LinearProblem(M, rhs)
    sol = solve(linprob, alg.linsolve)
    u2 = sol.u
    u = u2
    integrator.stats.nsolve += 1

    # compute Patankar weight denominator
    ρ = n1 * u + n2 * u .^ 2 ./ σ
    # avoid division by zero due to zero Patankar weights
    ρ = add_small_constant(ρ, small_constant)

    P2 = f.p(u, p, t + β10 * dt)
    Ptmp = β20 * P + β21 * P2
    integrator.stats.nf += 1

    # build linear system matrix and rhs
    if f isa PDSFunction
        d2 = f.d(u, p, t + β10 * dt)  # evaluate nonconservative destruction terms
        dtmp = β20 * d + β21 * d2
        rhs = α20 * uprev + α21 * u2 + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, ρ, dt, dtmp)

    else
        rhs = α20 * uprev + α21 * u2
        M = build_mprk_matrix(Ptmp, ρ, dt)
    end

    # solve linear system
    linprob = LinearProblem(M, rhs)
    sol = solve(linprob, alg.linsolve)
    u = sol.u
    integrator.stats.nsolve += 1

    # compute Patankar weight denominator
    σ = σ .^ (1 - s) .* u2 .^ s
    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(σ, small_constant)

    Ptmp = η3 * P + η4 * P2

    # build linear system matrix and rhs
    if f isa PDSFunction
        dtmp = η3 * d + η4 * d2

        # see (3.25 f) in original paper
        rhs = η1 * uprev + η2 * u2 + dt * (η5 * diag(P) + η6 * diag(P2))

        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
    else
        rhs = η1 * uprev + η2 * u2
        M = build_mprk_matrix(Ptmp, σ, dt)
    end

    # solve linear system
    linprob = LinearProblem(M, rhs)
    sol = solve(linprob, alg.linsolve)
    σ = sol.u
    integrator.stats.nsolve += 1

    # compute Patankar weight denominator
    σ = σ + z .* uprev .* u ./ ρ
    # avoid division by zero due to zero Patankar weights
    σ = add_small_constant(σ, small_constant)

    P3 = f.p(u, p, t + c3 * dt)
    Ptmp = β30 * P + β31 * P2 + β32 * P3
    integrator.stats.nf += 1

    # build linear system matrix
    if f isa PDSFunction
        d3 = f.d(u, p, t + c3 * dt)  # evaluate nonconservative destruction terms
        dtmp = β30 * d + β31 * d2 + β32 * d3
        rhs = α30 * uprev + α31 * u2 + α32 * u + dt * diag(Ptmp)
        M = build_mprk_matrix(Ptmp, σ, dt, dtmp)
    else
        rhs = α30 * uprev + α31 * u2 + α32 * u
        M = build_mprk_matrix(Ptmp, σ, dt)
    end

    # solve linear system
    linprob = LinearProblem(M, rhs)
    sol = solve(linprob, alg.linsolve)
    u = sol.u
    integrator.stats.nsolve += 1

    #TODO: Figure out if a second order approximation of the solution
    # is hidden somewhere.
    #=
    tmp = u - σ
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)
    =#

    integrator.u = u
end

struct SSPMPRK43Cache{uType, PType, tabType, F} <: MPRKMutableCache
    tmp::uType
    tmp2::uType
    P::PType
    P2::PType
    P3::PType
    D::uType
    D2::uType
    D3::uType
    σ::uType
    ρ::uType
    tab::tabType
    linsolve::F
end

struct SSPMPRK43ConservativeCache{uType, PType, tabType, F} <: MPRKMutableCache
    tmp::uType
    tmp2::uType
    P::PType
    P2::PType
    P3::PType
    σ::uType
    ρ::uType
    tab::tabType
    linsolve::F
end

get_tmp_cache(integrator, ::SSPMPRK43, cache::OrdinaryDiffEqMutableCache) = (cache.σ,)

# In-place
function alg_cache(alg::SSPMPRK43, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    n1, n2, z, η1, η2, η3, η4, η5, η6, s, α10, α20, α21, α30, α31, α32, β10, β20, β21, β30, β31, β32, c3 = get_constant_parameters(alg)
    tab = SSPMPRK43ConstantCache(n1, n2, z, η1, η2, η3, η4, η5, η6, s, α10, α20, α21, α30,
                                 α31, α32,
                                 β10, β20, β21, β30, β31, β32, c3,
                                 alg.small_constant_function(uEltypeNoUnits))
    tmp = zero(u)
    tmp2 = zero(u)
    P = p_prototype(u, f)
    P2 = p_prototype(u, f)
    # We use P3 to store the last evaluation of the PDS
    # as well as to store the system matrix of the linear system
    P3 = p_prototype(u, f)
    σ = zero(u)
    ρ = zero(u)

    if f isa ConservativePDSFunction
        linprob = LinearProblem(P3, _vec(tmp))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))
        SSPMPRK43ConservativeCache(tmp, tmp2, P, P2, P3, σ, ρ, tab, linsolve)
    elseif f isa PDSFunction
        D = similar(u)
        D2 = similar(u)
        D3 = similar(u)

        linprob = LinearProblem(P3, _vec(tmp))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        SSPMPRK43Cache(tmp, tmp2, P, P2, P3, D, D2, D3, σ, ρ, tab, linsolve)
    else
        throw(ArgumentError("SSPMPRK43 can only be applied to production-destruction systems"))
    end
end

function initialize!(integrator, cache::Union{SSPMPRK43ConservativeCache, SSPMPRK43Cache})
end

@muladd function perform_step!(integrator, cache::SSPMPRK43Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, tmp2, P, P2, P3, D, D2, D3, σ, ρ, linsolve = cache
    @unpack n1, n2, z, η1, η2, η3, η4, η5, η6, s, α10, α20, α21, α30, α31, α32, β10, β20, β21, β30, β31, β32, c3, small_constant = cache.tab

    # We use P3 to store the last evaluation of the PDS
    # as well as to store the system matrix of the linear system

    f.p(P, uprev, p, t) # evaluate production terms
    f.d(D, uprev, p, t) # evaluate nonconservative destruction terms
    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P3 = nonzeros(P3)
        @.. broadcast=false nz_P3=β10 * nz_P
    else
        @.. broadcast=false P3=β10 * P
    end
    @.. broadcast=false D3=β10 * D
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    # tmp holds the right hand side of the linear system
    @.. broadcast=false tmp=α10 * uprev
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P3[i, i]
    end

    build_mprk_matrix!(P3, P3, σ, dt, D3)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    tmp2 .= u
    integrator.stats.nsolve += 1

    @.. broadcast=false ρ=n1 * u + n2 * u^2 / σ
    @.. broadcast=false ρ=ρ + small_constant

    f.p(P2, u, p, t + β10 * dt) # evaluate production terms
    f.d(D2, u, p, t + β10 * dt) # evaluate nonconservative destruction terms
    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        nz_P3 = nonzeros(P3)
        @.. broadcast=false nz_P3=β20 * nz_P + β21 * nz_P2
    else
        @.. broadcast=false P3=β20 * P + β21 * P2
    end
    @.. broadcast=false D3=β20 * D + β21 * D2
    integrator.stats.nf += 1

    # tmp holds the right hand side of the linear system
    @.. broadcast=false tmp=α20 * uprev + α21 * tmp2
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P3[i, i]
    end

    build_mprk_matrix!(P3, P3, ρ, dt, D3)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    @.. broadcast=false σ=σ^(1 - s) * tmp2^s
    @.. broadcast=false σ=σ + small_constant

    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        nz_P3 = nonzeros(P3)
        @.. broadcast=false nz_P3=η3 * nz_P + η4 * nz_P2
    else
        @.. broadcast=false P3=η3 * P + η4 * P2
    end
    @.. broadcast=false D3=η3 * D + η4 * D2

    # tmp holds the right hand side of the linear system
    @.. broadcast=false tmp=η1 * uprev + η2 * tmp2
    @inbounds for i in eachindex(tmp)
        # see (3.25 f) in original paper
        tmp[i] += dt * (η5 * P[i, i] + η6 * P2[i, i])
    end

    build_mprk_matrix!(P3, P3, σ, dt, D3)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)
    integrator.stats.nsolve += 1

    σ .= linres

    @.. broadcast=false σ=σ + z * uprev * u / ρ
    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=σ + small_constant

    f.p(P3, u, p, t + c3 * dt) # evaluate production terms
    f.d(D3, u, p, t + c3 * dt) # evaluate nonconservative destruction terms
    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        nz_P3 = nonzeros(P3)
        @.. broadcast=false nz_P3=β30 * nz_P + β31 * nz_P2 + β32 * nz_P3
    else
        @.. broadcast=false P3=β30 * P + β31 * P2 + β32 * P3
    end
    @.. broadcast=false D3=β30 * D + β31 * D2 + β32 * D3
    integrator.stats.nf += 1

    # tmp holds the right hand side of the linear system
    @.. broadcast=false tmp=α30 * uprev + α31 * tmp2 + α32 * u
    @inbounds for i in eachindex(tmp)
        tmp[i] += dt * P3[i, i]
    end

    build_mprk_matrix!(P3, P3, σ, dt, D3)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    #TODO: Figure out if a second order approximation of the solution
    # is hidden somewhere.
    #=
    # Now tmp stores the error estimate
    @.. broadcast=false tmp=u - σ

    # Now tmp2 stores error residuals
    calculate_residuals!(tmp2, tmp, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp2, t)
    =#
end

@muladd function perform_step!(integrator, cache::SSPMPRK43ConservativeCache,
                               repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, tmp2, P, P2, P3, σ, ρ, linsolve = cache
    @unpack n1, n2, z, η1, η2, η3, η4, s, α10, α20, α21, α30, α31, α32, β10, β20, β21, β30, β31, β32, c3, small_constant = cache.tab

    # We use P3 to store the last evaluation of the PDS
    # as well as to store the system matrix of the linear system
    f.p(P, uprev, p, t) # evaluate production terms
    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P3 = nonzeros(P3)
        @.. broadcast=false nz_P3=β10 * nz_P
    else
        @.. broadcast=false P3=β10 * P
    end
    integrator.stats.nf += 1

    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=uprev + small_constant

    # tmp holds the right hand side of the linear system
    @.. broadcast=false tmp=α10 * uprev

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    tmp2 .= u
    integrator.stats.nsolve += 1

    @.. broadcast=false ρ=n1 * u + n2 * u^2 / σ
    @.. broadcast=false ρ=ρ + small_constant

    f.p(P2, u, p, t + β10 * dt) # evaluate production terms
    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        nz_P3 = nonzeros(P3)
        @.. broadcast=false nz_P3=β20 * nz_P + β21 * nz_P2
    else
        @.. broadcast=false P3=β20 * P + β21 * P2
    end
    integrator.stats.nf += 1

    @.. broadcast=false tmp=α20 * uprev + α21 * tmp2
    build_mprk_matrix!(P3, P3, ρ, dt)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    @.. broadcast=false σ=σ^(1 - s) * tmp2^s
    @.. broadcast=false σ=σ + small_constant

    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        nz_P3 = nonzeros(P3)
        @.. broadcast=false nz_P3=η3 * nz_P + η4 * nz_P2
    else
        @.. broadcast=false P3=η3 * P + η4 * P2
    end
    @.. broadcast=false tmp=η1 * uprev + η2 * tmp2

    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)
    integrator.stats.nsolve += 1

    σ .= linres
    @.. broadcast=false σ=σ + z * uprev * u / ρ
    # avoid division by zero due to zero Patankar weights
    @.. broadcast=false σ=σ + small_constant

    f.p(P3, u, p, t + c3 * dt) # evaluate production terms
    if issparse(P)
        # We need to keep the structural nonzeros of the production terms.
        # However, this is not guaranteed by broadcasting, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190
        nz_P = nonzeros(P)
        nz_P2 = nonzeros(P2)
        nz_P3 = nonzeros(P3)
        @.. broadcast=false nz_P3=β30 * nz_P + β31 * nz_P2 + β32 * nz_P3
    else
        @.. broadcast=false P3=β30 * P + β31 * P2 + β32 * P3
    end
    integrator.stats.nf += 1

    @.. broadcast=false tmp=α30 * uprev + α31 * tmp2 + α32 * u
    build_mprk_matrix!(P3, P3, σ, dt)

    # Same as linres = P3 \ tmp
    linsolve.A = P3
    linres = solve!(linsolve)

    u .= linres
    integrator.stats.nsolve += 1

    #TODO: Figure out if a second order approximation of the solution
    # is hidden somewhere.
    #=
    # Now tmp stores the error estimate
    @.. broadcast=false tmp=u - σ

    # Now tmp2 stores error residuals
    calculate_residuals!(tmp2, tmp, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp2, t)
    =#
end
