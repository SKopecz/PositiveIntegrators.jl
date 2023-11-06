
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
                                           mass_matrix = __has_mass_matrix(P) ?
                                                         P.mass_matrix : I,
                                           _func_cache = nothing,
                                           analytic = __has_analytic(P) ? P.analytic :
                                                      nothing,
                                           tgrad = __has_tgrad(P) ? P.tgrad : nothing,
                                           jac = __has_jac(P) ? P.jac : nothing,
                                           jvp = __has_jvp(P) ? P.jvp : nothing,
                                           vjp = __has_vjp(P) ? P.vjp : nothing,
                                           jac_prototype = __has_jac_prototype(P) ?
                                                           P.jac_prototype :
                                                           nothing,
                                           sparsity = __has_sparsity(P) ? P.sparsity :
                                                      jac_prototype,
                                           Wfact = __has_Wfact(P) ? P.Wfact : nothing,
                                           Wfact_t = __has_Wfact_t(P) ? P.Wfact_t : nothing,
                                           paramjac = __has_paramjac(P) ? P.paramjac :
                                                      nothing,
                                           syms = __has_syms(P) ? P.syms : nothing,
                                           indepsym = __has_indepsym(P) ? P.indepsym :
                                                      nothing,
                                           paramsyms = __has_paramsyms(P) ? P.paramsyms :
                                                       nothing,
                                           observed = __has_observed(P) ? P.observed :
                                                      DEFAULT_OBSERVED,
                                           colorvec = __has_colorvec(P) ? P.colorvec :
                                                      nothing,
                                           sys = __has_sys(P) ? P.sys : nothing) where {iip,
                                                                                        specialize
                                                                                        }
    if specialize === NoSpecialize
        ProdDestFunction{iip, specialize, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any,
                         Any,
                         Any, Any, Any, Any, Any,
                         Any, Any, Any, Any, Any, Any}(P, D, mass_matrix, _func_cache,
                                                       analytic,
                                                       tgrad, jac, jvp, vjp, jac_prototype,
                                                       sparsity, Wfact, Wfact_t, paramjac,
                                                       syms, indepsym, paramsyms,
                                                       observed, colorvec, sys)
    else
        ProdDestFunction{iip, specialize, typeof(P), typeof(D), typeof(p_prototype),
                         typeof(d_prototype),
                         typeof(mass_matrix), typeof(_func_cache), typeof(analytic),
                         typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
                         typeof(jac_prototype), typeof(sparsity),
                         typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                         typeof(indepsym), typeof(paramsyms), typeof(observed),
                         typeof(colorvec),
                         typeof(sys)}(P, D, p_prototype, d_prototype, mass_matrix,
                                      _func_cache, analytic, tgrad, jac,
                                      jvp, vjp, jac_prototype,
                                      sparsity, Wfact, Wfact_t, paramjac, syms, indepsym,
                                      paramsyms, observed, colorvec, sys)
    end
end

@add_kwonly function ProdDestFunction(P, D, p_prototype, d_prototype, mass_matrix, cache,
                                      analytic, tgrad, jac, jvp,
                                      vjp, jac_prototype, sparsity, Wfact, Wfact_t,
                                      paramjac,
                                      syms, indepsym, paramsyms, observed, colorvec, sys)
    P = ODEFunction(P)
    D = ODEFunction(D)

    #if !(typeof(P) <: AbstractSciMLOperator || typeof(f1.f) <: AbstractSciMLOperator) &&
    #isinplace(f1) != isinplace(f2)
    #throw(NonconformingFunctionsError(["f2"]))
    #end

    ProdDestFunction{isinplace(P, 4), FullSpecialize, typeof(P), typeof(D),
                     typeof(p_prototype), typeof(d_prototype), typeof(mass_matrix),
                     typeof(cache), typeof(analytic), typeof(tgrad), typeof(jac),
                     typeof(jvp),
                     typeof(vjp), typeof(jac_prototype), typeof(sparsity),
                     typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                     typeof(indepsym), typeof(paramsyms), typeof(observed),
                     typeof(colorvec),
                     typeof(sys)}(P, D, mass_matrix, cache, analytic, tgrad, jac, jvp, vjp,
                                  jac_prototype, sparsity, Wfact, Wfact_t, paramjac, syms,
                                  indepsym,
                                  paramsyms, observed, colorvec, sys)
end

function (PD::ProdDestFunction)(u, p, t)
    diag(PD.p(u, p, t)) + vec(sum(PD.p(u, p, t), dims = 2)) -
    vec(sum(PD.p(u, p, t), dims = 1)) - vec(PD.d(u, p, t))
end

function (PD::ProdDestFunction)(du, u, p, t)
    PD.p(PD.p_prototype, u, p, t)

    if PD.p_prototype isa AbstractSparseMatrix
        # Same result but more efficient - at least currently for SparseMatrixCSC
        fill!(PD.d_prototype, one(eltype(PD.d_prototype)))
        mul!(vec(du), PD.p_prototype, PD.d_prototype)
    else
        sum!(vec(du), PD.p_prototype)
    end

    for i in 1:length(u)  #vec(du) .+= diag(PD.p_prototype)
        du[i] += PD.p_prototype[i, i]
    end
    sum!(PD.d_prototype', PD.p_prototype)
    vec(du) .-= PD.d_prototype
    PD.d(PD.d_prototype, u, p, t)
    vec(du) .-= PD.d_prototype
    return nothing
end

# New ODE function ConservativePDSFunction
struct ConservativePDSFunction{iip, specialize, P, PrototypeP, TMP, Ta} <:
       AbstractODEFunction{iip}
    p::P
    p_prototype::PrototypeP
    tmp::TMP
    analytic::Ta
end

function Base.getproperty(obj::ConservativePDSFunction, sym::Symbol)
    if sym === :mass_matrix
        return I
    elseif sym === :jac_prototype
        return nothing
    elseif sym === :colorvec
        return nothing
    elseif sym === :sparsity
        return nothing
    else # fallback to getfield
        return getfield(obj, sym)
    end
end

function ConservativePDSFunction{iip, FullSpecialize}(P; p_prototype = nothing,
                                                      analytic = nothing) where {iip}
    if p_prototype isa AbstractSparseMatrix
        tmp = zeros(eltype(p_prototype), (size(p_prototype, 1),))
    else
        tmp = nothing
    end
    ConservativePDSFunction{iip, FullSpecialize, typeof(P), typeof(p_prototype),
                            typeof(tmp), typeof(analytic)}(P, p_prototype, tmp, analytic)
end

#function ConservativePDSFunction{iip}(P; kwargs...) where {iip}
#    ConservativePDSFunction{iip,FullSpecialize}(P; kwargs...)
#end

function ConservativePDSFunction(P; kwargs...)
    ConservativePDSFunction{isinplace(P, 4), FullSpecialize}(P; kwargs...)
end

#=
#Do we really need this????
@add_kwonly function ConservativePDSFunction(P, p_prototype, tmp)
    P = ODEFunction(P)

    ConservativePDSFunction{isinplace(P, 4),FullSpecialize,typeof(P), typeof(p_prototype),typeof(tmp)}
    (P, D, mass_matrix, cache, analytic, tgrad, jac, jvp, vjp,
        jac_prototype, sparsity, Wfact, Wfact_t, paramjac, syms,
        indepsym,
        paramsyms, observed, colorvec, sys)
end
=#

# Evaluation of a ConservativePDSFunction (out-of-place)
function (PD::ConservativePDSFunction)(u, p, t)
    vec(sum(PD.p(u, p, t), dims = 2)) - vec(sum(PD.p(u, p, t), dims = 1))
end

# Evaluation of a ConservativePDSFunction (in-place)
function (PD::ConservativePDSFunction)(du, u, p, t)
    PD.p(PD.p_prototype, u, p, t)

    if PD.p_prototype isa AbstractSparseMatrix
        # Same result but more efficient - at least currently for SparseMatrixCSC
        fill!(PD.tmp, one(eltype(PD.tmp)))
        mul!(vec(du), PD.p_prototype, PD.tmp)
        sum!(PD.tmp', PD.p_prototype)
        vec(du) .-= PD.tmp
    else
        # This implementation does not need any auxiliary vectors
        for i in 1:length(u)
            du[i] = zero(eltype(du))
            for j in 1:length(u)
                du[i] += PD.p_prototype[i, j] - PD.p_prototype[j, i]
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

function ProdDestODEProblem(PD::ProdDestFunction, u0, tspan, p = NullParameters();
                            kwargs...)
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

# New problem type ConservativePDSProblem
"""
    ConservativePDSProblem(P, u0, tspan, p = NullParameters();
                            p_prototype = similar(u0, (length(u0), length(u0))), analytic=nothing)

A structure describing a conservative ordinary differential equation in form of a production-destruction system (PDS).
`P` denotes the production matrix.
`u0` is the vector of initial conditions and `tspan` the time span
`(t_initial, t_final)` of the problem. The optional argument `p` can be used
to pass additional parameters to the functions.

The function `P` can be given either in the out-of-place form with signature
`production_terms = P(u, p, t)` or the in-place form `P(production_terms, u, p, t)`.

### Keyword arguments: ###

- `p_prototype`: If `P` is given in in-place form, `p_prototype` is used to store evaluations of `P`. 
    If `p_prototype` is not specified explicitly and `P` is in-place, then `p_prototype` will be internally
  set to `zeros(eltype(u0), (length(u0), length(u0)))`. 
- `analytic`: The analytic solution of a PDS must be given in the form `f(u0,p,t)`. 
    Specifying the analytic solution can be useful for plotting and convergence tests.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
"""
struct ConservativePDSProblem{iip} <: AbstractProdDestODEProblem end

# General constructor for ConservativePDSProblems
function ConservativePDSProblem(P, u0, tspan, p = NullParameters();
                                p_prototype = nothing, analytic = nothing, kwargs...)

    # p_prototype is used to store evaluations of P, if P is in-place.
    if isinplace(P, 4) && isnothing(p_prototype)
        p_prototype = zeros(eltype(u0), (length(u0), length(u0)))
    end

    PD = ConservativePDSFunction(P; p_prototype = p_prototype, analytic = analytic)
    ConservativePDSProblem(PD, u0, tspan, p; kwargs...)
end


# Specialized constructor for ConservativePDSProblems with in-place P matrix.
function ConservativePDSProblem{true}(P, u0, tspan, p = NullParameters();
    p_prototype = nothing, analytic = nothing, kwargs...)

    # p_prototype is used to store evaluations of P, if P is in-place.
    if isnothing(p_prototype)#
        p_prototype = zeros(eltype(u0), (length(u0), length(u0)))
    end

    ### Internal isinplace must be avoided!!!
    PD = ConservativePDSFunction(P; p_prototype = p_prototype, analytic = analytic)
    ConservativePDSProblem(PD, u0, tspan, p; kwargs...)
end

# Construct ConservativePDSProblem from ConservativePDSFunction
function ConservativePDSProblem(PD::ConservativePDSFunction, u0, tspan,
                                p = NullParameters(); kwargs...)
    ConservativePDSProblem{isinplace(PD)}(PD, u0, tspan, p; kwargs...)
end
function ConservativePDSProblem{iip}(PD::ConservativePDSFunction, u0, tspan,
                                     p = NullParameters(); kwargs...) where {iip}
    ODEProblem(PD, u0, tspan, p, ConservativePDSProblem{iip}(); kwargs...)
end
