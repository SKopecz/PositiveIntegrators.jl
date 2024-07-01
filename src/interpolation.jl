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

#######################################################################################
# interpolation specializations
function interp_summary(::Type{cacheType},
                        dense::Bool) where {
                                            cacheType <:
                                            Union{MPEConstantCache, MPECache,
                                                  MPRK22ConstantCache, MPRK22Cache,
                                                  MPRK43ConstantCache, MPRK43Cache,
                                                  SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                                  SSPMPRK43ConstantCache, SSPMPRK43Cache}}
    "1st order linear"
end

function _ode_interpolant(Θ, dt, u0, u1, k,
                          cache::Union{MPEConstantCache, MPECache, MPRK22ConstantCache,
                                       MPRK22Cache,
                                       MPRK43ConstantCache, MPRK43Cache,
                                       SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                       SSPMPRK43ConstantCache, SSPMPRK43Cache},
                          idxs, # Optionally specialize for ::Nothing and others
                          T::Type{Val{0}},
                          differential_vars::Nothing)
    linear_interpolant(Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant!(out, Θ, dt, u0, u1, k,
                           cache::Union{MPEConstantCache, MPECache, MPRK22ConstantCache,
                                        MPRK22Cache,
                                        MPRK43ConstantCache, MPRK43Cache,
                                        SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                        SSPMPRK43ConstantCache, SSPMPRK43Cache},
                           idxs, # Optionally specialize for ::Nothing and others
                           T::Type{Val{0}},
                           differential_vars::Nothing)
    linear_interpolant!(out, Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant(Θ, dt, u0, u1, k,
                          cache::Union{MPEConstantCache, MPECache, MPRK22ConstantCache,
                                       MPRK22Cache,
                                       MPRK43ConstantCache, MPRK43Cache,
                                       SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                       SSPMPRK43ConstantCache, SSPMPRK43Cache},
                          idxs, # Optionally specialize for ::Nothing and others
                          T::Type{Val{1}},
                          differential_vars::Nothing)
    linear_interpolant(Θ, dt, u0, u1, idxs, T)
end

function _ode_interpolant!(out, Θ, dt, u0, u1, k,
                           cache::Union{MPEConstantCache, MPECache, MPRK22ConstantCache,
                                        MPRK22Cache,
                                        MPRK43ConstantCache, MPRK43Cache,
                                        SSPMPRK22ConstantCache, SSPMPRK22Cache,
                                        SSPMPRK43ConstantCache, SSPMPRK43Cache},
                           idxs, # Optionally specialize for ::Nothing and others
                           T::Type{Val{1}},
                           differential_vars::Nothing)
    linear_interpolant!(out, Θ, dt, u0, u1, idxs, T)
end
