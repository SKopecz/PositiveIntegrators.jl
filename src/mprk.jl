import OrdinaryDiffEq: @muladd, @unpack, @cache, @..,
       OrdinaryDiffEqAdaptiveAlgorithm, calculate_residuals, calculate_residuals!, False, 
       OrdinaryDiffEqMutableCache, OrdinaryDiffEqConstantCache, 
       alg_cache, initialize!, perform_step!

### MPE #####################################################################################
struct MPE <: OrdinaryDiffEqAlgorithm end

#@cache 
struct MPECache{uType, rateType, PType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    k::rateType
    fsalfirst::rateType
    P::PType
    D::uType
end

function alg_cache(alg::MPE, u, rate_prototype, ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
    dt, reltol, p, calck,
    ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    MPECache(u, uprev, zero(u), zero(rate_prototype), zero(rate_prototype),zeros(eltype(u),length(u),length(u)),zero(u))
end

struct MPEConstantCache <: OrdinaryDiffEqConstantCache end

function alg_cache(alg::MPE, u, rate_prototype, ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
    dt, reltol, p, calck,
    ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    MPEConstantCache()
end

function initialize!(integrator, cache::MPEConstantCache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.stats.nf += 1

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
end

function perform_step!(integrator, cache::MPEConstantCache, repeat_step = false)
    @unpack t, dt, uprev, f, p = integrator
 
    # Attention: Implementation assumes that the pds is positive and conservative,
    # i.e. f.p[i,i] == 0 for all i

    P = f.p(uprev,p,t) # evaluate production terms 
    D = vec(sum(P,dims=1)) # compute sum of destruction terms from P
    M = -dt*P./reshape(uprev,1,:) # divide production terms by Patankar-weights
    M[diagind(M)] .+= 1.0 .+ dt*D./uprev # add destruction terms on diagonal
    u = M\uprev

    k = f(u, p, t + dt) # For the interpolation, needs k at the updated point
    integrator.stats.nf += 1
    integrator.fsallast = k
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
end


function initialize!(integrator, cache::MPECache)
    integrator.kshortsize = 2
    @unpack k, fsalfirst = cache
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t) # For the interpolation, needs k at the updated point
    integrator.stats.nf += 1
end



function perform_step!(integrator, cache::MPECache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack P, D = cache
    #@muladd @.. broadcast=false u=0*uprev + 123.0# + dt * integrator.fsalfirst

    P .= 0.0
    f.p(P, uprev, p, t) #evaluate production terms
    sum!(D', P) # sum destruction terms
    for j=1:length(u)        
        for i = 1:length(u)
            if i == j
                P[i,i] = 1.0 .+ dt*D[i]/uprev[i]
            else
                P[i,j] = -dt*P[i,j]/uprev[j]
            end
        end
    end
    tmp = P\uprev #needs to be implemented without allocations.
    u .= tmp

    f(integrator.fsallast, u, p, t + dt) # For the interpolation, needs k at the updated point
    integrator.stats.nf += 1
end

### MPRK #####################################################################################
struct MPRK22{T,Thread} <: OrdinaryDiffEqAdaptiveAlgorithm
    alpha::T
    thread::Thread
end

function MPRK22(alpha; thread = False())
    MPRK22{typeof(alpha), typeof(thread)}(alpha, thread)
end

OrdinaryDiffEq.alg_order(alg::MPRK22) = 2

#@cache 
struct MPRK22Cache{uType, rateType, PType,tabType,Thread} <: OrdinaryDiffEqMutableCache
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
end

function alg_cache(alg::MPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
    dt, reltol, p, calck,
    ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    tab = MPRK22ConstantCache(alg.alpha,1-1/(2*alg.alpha),1/(2*alg.alpha),alg.alpha,floatmin(uEltypeNoUnits))
    MPRK22Cache(u, uprev, zero(u), zero(u), zero(rate_prototype), zero(rate_prototype),zeros(eltype(u),length(u),length(u)),
      zeros(eltype(u),length(u),length(u)),zero(u),zero(u),zeros(eltype(u),length(u),length(u)),zero(u),tab,alg.thread)
end

struct MPRK22ConstantCache{T} <: OrdinaryDiffEqConstantCache 
    a21::T
    b1::T
    b2::T
    c2::T
    smallconst::T
end

function alg_cache(alg::MPRK22, u, rate_prototype, ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
    dt, reltol, p, calck,
    ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}

    #Should assert alg.alpha >= 0.5

    MPRK22ConstantCache(alg.alpha,1-1/(2*alg.alpha),1/(2*alg.alpha),alg.alpha,floatmin(uEltypeNoUnits))
end

function initialize!(integrator, cache::MPRK22ConstantCache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.stats.nf += 1

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
end

function perform_step!(integrator, cache::MPRK22ConstantCache, repeat_step = false)
    @unpack t, dt, uprev, f, p = integrator
    @unpack a21, b1, b2, c2 = cache

    safeguard = floatmin(eltype(uprev))

    uprev .= uprev .+ safeguard

    P = f.p(uprev,p,t) # evaluate production terms 
    D = vec(sum(P,dims=1)) # sum destruction terms

    M = -dt*a21*P./reshape(uprev,1,:) # divide production terms by Patankar-weights
    M[diagind(M)] .+= 1.0 .+ dt*a21*D./uprev
    u = M\uprev

    u .= u .+ safeguard

    σ = uprev.*(u./uprev).^(1/a21) .+ safeguard
    
    P2 = f.p(u,p,t+a21*dt)
    D2 = vec(sum(P2,dims=1))
    M .= -dt*(b1*P + b2*P2)./reshape(σ,1,:)
    M[diagind(M)] .+= 1.0 .+ dt*(b1*D + b2*D2)./σ
    u = M\uprev

    u .= u .+ safeguard

    k = f(u, p, t + dt) # For the interpolation, needs k at the updated point
    integrator.stats.nf += 1
    integrator.fsallast = k

    #copied from perform_step for HeunConstantCache
    tmp = u - σ
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
            integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)


    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
end


function initialize!(integrator, cache::MPRK22Cache)
    integrator.kshortsize = 2
    @unpack k, fsalfirst = cache
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t) # For the interpolation, needs k at the updated point
    integrator.stats.nf += 1
end



function perform_step!(integrator, cache::MPRK22Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, atmp, P, P2, D, D2, M, σ, thread = cache
    @unpack a21, b1, b2, c2, smallconst = cache.tab

    uprev .= uprev .+ smallconst

    f.p(P, uprev, p, t) #evaluate production terms
    sum!(D', P) # sum destruction terms
    for j=1:length(u)        
        for i = 1:length(u)
            if i == j
                M[i,i] = 1.0 .+ dt*a21*D[i]/uprev[i]
            else
                M[i,j] = -dt*a21*P[i,j]/uprev[j]
            end
        end
    end
    tmp = M\uprev #needs to be implemented without allocations.
    u .= tmp

    u .= u .+ smallconst

    σ .= uprev.*(u./uprev).^(1/a21) .+ smallconst

    f.p(P2, u, p, t+a21*dt) #evaluate production terms
    sum!(D2', P2) # sum destruction terms
    for j=1:length(u)        
        for i = 1:length(u)
            if i == j
                M[i,i] = 1.0 .+ dt*(b1*D[i] + b2*D2[i])/σ[i]
            else
                M[i,j] = -dt*(b1*P[i,j] + b2*P2[i,j])/σ[j]
            end
        end
    end
    tmp = M\uprev #needs to be implemented without allocations.
    u .= tmp

    tmp .= u .- σ
    calculate_residuals!(atmp, tmp, uprev, u, integrator.opts.abstol,
    integrator.opts.reltol, integrator.opts.internalnorm, t,
    thread)
integrator.EEst = integrator.opts.internalnorm(atmp, t)

    f(integrator.fsallast, u, p, t + dt) # For the interpolation, needs k at the updated point
    integrator.stats.nf += 1
end