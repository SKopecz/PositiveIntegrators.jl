
# New ODE function ProdDestFunction
struct ProdDestFunction{iip, specialize, P, D, PrototypeP, PrototypeD, TMM, C,
                        Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ, S, S2, S3, O,
                        TCV, SYS} <: AbstractODEFunction{iip}
    p::P
    d::D
    p_prototype::PrototypeP
    d_prototype::PrototypeD
    mass_matrix::TMM
    cache::C
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    indepsym::S2
    paramsyms::S3
    observed::O
    colorvec::TCV
    sys::SYS
end

ProdDestFunction(P, D; kwargs...) = ProdDestFunction{isinplace(P, 4)}(P, D; kwargs...)

function ProdDestFunction{iip}(P, D; kwargs...) where {iip}
    ProdDestFunction{iip, FullSpecialize}(P, D; kwargs...)
end

function ProdDestFunction{iip, specialize}(P, D;
    p_prototype = nothing,
    d_prototype = nothing,
    mass_matrix=__has_mass_matrix(P) ?
                P.mass_matrix : I,
    _func_cache=nothing,
    analytic=__has_analytic(P) ? P.analytic :
             nothing,
    tgrad=__has_tgrad(P) ? P.tgrad : nothing,
    jac=__has_jac(P) ? P.jac : nothing,
    jvp=__has_jvp(P) ? P.jvp : nothing,
    vjp=__has_vjp(P) ? P.vjp : nothing,
    jac_prototype=__has_jac_prototype(P) ?
                  P.jac_prototype :
                  nothing,
    sparsity=__has_sparsity(P) ? P.sparsity :
             jac_prototype,
    Wfact=__has_Wfact(P) ? P.Wfact : nothing,
    Wfact_t=__has_Wfact_t(P) ? P.Wfact_t : nothing,
    paramjac=__has_paramjac(P) ? P.paramjac :
             nothing,
    syms=__has_syms(P) ? P.syms : nothing,
    indepsym=__has_indepsym(P) ? P.indepsym :
             nothing,
    paramsyms=__has_paramsyms(P) ? P.paramsyms :
              nothing,
    observed=__has_observed(P) ? P.observed :
             DEFAULT_OBSERVED,
    colorvec=__has_colorvec(P) ? P.colorvec :
             nothing,
    sys=__has_sys(P) ? P.sys : nothing
) where {iip, specialize }
    if specialize === NoSpecialize
        ProdDestFunction{iip,specialize,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,
            Any,Any,Any,Any,Any,
            Any,Any,Any,Any,Any,Any}(P, D, mass_matrix, _func_cache,
            analytic,
            tgrad, jac, jvp, vjp, jac_prototype,
            sparsity, Wfact, Wfact_t, paramjac,
            syms, indepsym, paramsyms,
            observed, colorvec, sys)
    else
        ProdDestFunction{iip,specialize,typeof(P),typeof(D),typeof(p_prototype),typeof(d_prototype),
            typeof(mass_matrix),typeof(_func_cache),typeof(analytic),
            typeof(tgrad),typeof(jac),typeof(jvp),typeof(vjp),
            typeof(jac_prototype),typeof(sparsity),
            typeof(Wfact),typeof(Wfact_t),typeof(paramjac),typeof(syms),
            typeof(indepsym),typeof(paramsyms),typeof(observed),
            typeof(colorvec),
            typeof(sys)}(P, D, p_prototype, d_prototype, mass_matrix,
            _func_cache, analytic, tgrad, jac,
            jvp, vjp, jac_prototype,
            sparsity, Wfact, Wfact_t, paramjac, syms, indepsym,
            paramsyms, observed, colorvec, sys)
    end
end


@add_kwonly function ProdDestFunction(P, D, p_prototype, d_prototype, mass_matrix, cache, analytic, tgrad, jac, jvp,
    vjp, jac_prototype, sparsity, Wfact, Wfact_t, paramjac,
    syms, indepsym, paramsyms, observed, colorvec, sys)
    P = ODEFunction(P)
    D = ODEFunction(D)

    #if !(typeof(P) <: AbstractSciMLOperator || typeof(f1.f) <: AbstractSciMLOperator) &&
    #isinplace(f1) != isinplace(f2)
    #throw(NonconformingFunctionsError(["f2"]))
    #end

    ProdDestFunction{isinplace(P, 4),FullSpecialize,typeof(P),typeof(D),
        typeof(p_prototype),typeof(d_prototype),typeof(mass_matrix),
        typeof(cache),typeof(analytic),typeof(tgrad),typeof(jac),typeof(jvp),
        typeof(vjp),typeof(jac_prototype),typeof(sparsity),
        typeof(Wfact),typeof(Wfact_t),typeof(paramjac),typeof(syms),
        typeof(indepsym),typeof(paramsyms),typeof(observed),typeof(colorvec),
        typeof(sys)}(P, D, mass_matrix, cache, analytic, tgrad, jac, jvp, vjp,
        jac_prototype, sparsity, Wfact, Wfact_t, paramjac, syms,
        indepsym,
        paramsyms, observed, colorvec, sys)
end

(PD::ProdDestFunction)(u, p, t) = diag(PD.p(u,p,t)) + vec(sum(PD.p(u, p, t),dims=2)) - vec(sum(PD.p(u,p,t),dims=1)) - vec(PD.d(u,p,t))

function (PD::ProdDestFunction)(du, u, p, t)
    PD.p(PD.p_prototype, u, p, t)

    if PD.p_prototype isa AbstractSparseMatrix
        # Same result but more efficient - at least currently for SparseMatrixCSC
        fill!(PD.d_prototype, one(eltype(PD.d_prototype)))
        mul!(vec(du), PD.p_prototype, PD.d_prototype)
    else
        sum!(vec(du), PD.p_prototype)
    end

    for i=1:length(u)  #vec(du) .+= diag(PD.p_prototype)
        du[i] += PD.p_prototype[i,i]
    end
    sum!(PD.d_prototype', PD.p_prototype)
    vec(du) .-= PD.d_prototype
    PD.d(PD.d_prototype, u, p, t)
    vec(du) .-= PD.d_prototype
    return nothing
end

# New ODE function ConsProdDestFunction
struct ConsProdDestFunction{iip,specialize,P,PrototypeP,TMP} <: AbstractODEFunction{iip}
    p::P
    p_prototype::PrototypeP
    tmp::TMP
end

function ConsProdDestFunction{iip,FullSpecialize}(P; p_prototype=nothing, tmp=nothing) where {iip}
    ConsProdDestFunction{iip,FullSpecialize,typeof(P),typeof(p_prototype),typeof(tmp)}(P, p_prototype, tmp)
end

function ConsProdDestFunction{iip}(P; kwargs...) where {iip}
    ConsProdDestFunction{iip,FullSpecialize}(P; kwargs...)
end

ConsProdDestFunction(P; kwargs...) = ConsProdDestFunction{isinplace(P, 4)}(P; kwargs...)

#=
#Do we really need this????
@add_kwonly function ConsProdDestFunction(P, p_prototype, tmp)
    P = ODEFunction(P)

    ConsProdDestFunction{isinplace(P, 4),FullSpecialize,typeof(P), typeof(p_prototype),typeof(tmp)}
    (P, D, mass_matrix, cache, analytic, tgrad, jac, jvp, vjp,
        jac_prototype, sparsity, Wfact, Wfact_t, paramjac, syms,
        indepsym,
        paramsyms, observed, colorvec, sys)
end
=#

(PD::ConsProdDestFunction)(u, p, t) = vec(sum(PD.p(u, p, t),dims=2)) - vec(sum(PD.p(u,p,t),dims=1))

function (PD::ConsProdDestFunction)(du, u, p, t)
    PD.p(PD.p_prototype, u, p, t)

    if PD.p_prototype isa AbstractSparseMatrix
        # Same result but more efficient - at least currently for SparseMatrixCSC
        fill!(PD.tmp, one(eltype(PD.tmp)))
        mul!(vec(du), PD.p_prototype, PD.tmp)
        sum!(PD.tmp', PD.p_prototype)
        vec(du) .-= tmp
    else
        # This implementation does not need any auxiliary vectors
        for i = 1:length(u)
            du[i] = zero(eltype(du))
            for j = 1:length(u)
                du[i] += PD.p_prototype[i,j] - PD.p_prototype[j,i]
            end
        end
    end
    return nothing
end


# New problem type ProdDestODEProblem
abstract type AbstractProdDestODEProblem end

"""
    ProdDestODEProblem(P, D, u0, tspan, p = NullParameters();
                       p_prototype = similar(u0, (length(u0), length(u0))),
                       d_prototype = similar(u0, (length(u0),)),)

A structure describing a production-destruction ordinary differential equation.
`P` and `D` denote the production/destruction terms in form of a matrix.
`u0` is the vector of initial conditions and `tspan` the time span
`(t_initial, t_final)` of the problem. The optional argument `p` can be used
to pass additional parameters to the functions.

The functions `P` and `D` can be used either in the out-of-place form with signature
`production_terms = P(u, p, t)` or the in-place form `P(production_terms, u, p, t)`.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
"""
struct ProdDestODEProblem{iip} <: AbstractProdDestODEProblem end

function ProdDestODEProblem(P, D, u0, tspan, p = NullParameters();
                            p_prototype = similar(u0, (length(u0), length(u0))),
                            d_prototype = similar(u0, (length(u0),)),
                            kwargs...)
    p_prototype .= zero(eltype(p_prototype))
    d_prototype .= zero(eltype(d_prototype))
    PD = ProdDestFunction(P, D; p_prototype, d_prototype)
    ProdDestODEProblem(PD, u0, tspan, p; kwargs...)
end

function ProdDestODEProblem(PD::ProdDestFunction, u0, tspan, p = NullParameters(); kwargs...)
    ProdDestODEProblem{isinplace(PD)}(PD, u0, tspan, p; kwargs...)
end

function ProdDestODEProblem{iip}(PD::ProdDestFunction, u0, tspan, p = NullParameters();
                              kwargs...) where {iip}
    #if f.cache === nothing && iip
    #    cache = similar(u0)
    #    f = SplitFunction{iip}(f.f1, f.f2; mass_matrix = f.mass_matrix,
    #                           _func_cache = cache, analytic = f.analytic)
    #end
    ODEProblem(PD, u0, tspan, p, ProdDestODEProblem{iip}(); kwargs...)
end

# New problem type ConsProdDestODEProblem
"""
    ConsProdDestODEProblem(P, u0, tspan, p = NullParameters();
                            p_prototype = similar(u0, (length(u0), length(u0))))

A structure describing a conservative ordinary differential equation in form of a production-destruction system.
`P` denotes the production terms in form of a matrix.
`u0` is the vector of initial conditions and `tspan` the time span
`(t_initial, t_final)` of the problem. The optional argument `p` can be used
to pass additional parameters to the functions.

The function `P` can be used either in the out-of-place form with signature
`production_terms = P(u, p, t)` or the in-place form `P(production_terms, u, p, t)`.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
"""
struct ConsProdDestODEProblem{iip} <: AbstractProdDestODEProblem end

function ConsProdDestODEProblem(P, u0, tspan, p = NullParameters();
                            p_prototype = similar(u0, (length(u0), length(u0))), kwargs...)
    p_prototype .= zero(eltype(p_prototype))

    if p_prototype isa AbstractSparseMatrix
        tmp = zeros(eltype(p_prototype), (length(u0),))
    else
        tmp = nothing
    end
    PD = ConsProdDestFunction(P; p_prototype, tmp)
    ConsProdDestODEProblem(PD, u0, tspan, p; kwargs...)
end

function ConsProdDestODEProblem(PD::ConsProdDestFunction, u0, tspan, p = NullParameters(); kwargs...)
    ConsProdDestODEProblem{isinplace(PD)}(PD, u0, tspan, p; kwargs...)
end

function ConsProdDestODEProblem{iip}(PD::ConsProdDestFunction, u0, tspan, p = NullParameters(); kwargs...) where {iip}
    ODEProblem(PD, u0, tspan, p, ProdDestODEProblem{iip}(); kwargs...)
end


