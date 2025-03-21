"""
    MPDeC(K; [nodes = :gausslobatto, linsolve = ..., small_constant = ...])

A family of arbitrary order modified Patankar-Runge-Kutta algorithms for
production-destruction systems. Each member of this family is an adaptive, one-step method which is
Kth order accurate, unconditionally positivity-preserving, and linearly
implicit. The integer K must be chosen to satisfy 2 ≤ K ≤ 10. 
Available node choices are Lagrange or Gauss-Lobatto nodes, with the latter being the default.
These methods support adaptive time stepping, using the numerical solution obtained with one correction step less as a lower-order approximation to estimate the error.
The MPDeC schemes were introduced by Torlo and Öffner (2020) for autonomous conservative production-destruction systems and
further investigated in Torlo, Öffner and Ranocha (2022).

For nonconservative production–destruction systems we use a straight forward extension
analogous to [`MPE`](@ref).
A general discussion of DeC schemes applied to non-autonomous differential equations 
and using general integration nodes is given by Ong and Spiteri (2020).

The MPDeC methods require the special structure of a
[`PDSProblem`](@ref) or a [`ConservativePDSProblem`](@ref).

You can optionally choose the linear solver to be used by passing an
algorithm from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
as keyword argument `linsolve`.
You can also choose the parameter `small_constant` which is added to all Patankar-weight denominators
to avoid divisions by zero. You can pass a value explicitly, otherwise `small_constant` is set to
`1e-300` in double precision computations or `floatmin` of the floating point type used.

## References

- Davide Torlo and Philipp Öffner.
  "Arbitrary high-order, conservative and positivity preserving Patankar-type deferred correction schemes."
  Applied Numerical Mathematics 153 (2020): 15-34.
  [DOI: 10.1016/j.apnum.2020.01.025](https://doi.org/10.1016/j.apnum.2020.01.025)
- Davide Torlo, Philipp Öffner, and Hendrik Ranocha.
  "Issues with positivity-preserving Patankar-type schemes."
  Applied Numerical Mathematics 182 (2022): 117-147.
  [DOI: 10.1016/j.apnum.2022.07.014](https://doi.org/10.1016/j.apnum.2022.07.014)

- Benjamin W. Ong and Raymond J. Spiteri.
  "Deferred Correction Methods for Ordinary Differential Equations."
  Journal of Scientific Computing 83 (2020): Article 60.
  [DOI: 10.1007/s10915-020-01235-8](https://doi.org/10.1007/s10915-020-01235-8)
"""
struct MPDeC{N, F, T2} <: OrdinaryDiffEqAdaptiveAlgorithm
    K::Int
    M::Int
    nodes::N
    linsolve::F
    small_constant_function::T2
end

function small_constant_function_MPDeC(type)
    if type == Float64
        # small_constant is chosen such that 
        # the testet "Zero initial values" passes.
        small_constant = 1e-300
    else
        small_constant = floatmin(type)
    end
    return small_constant
end

function MPDeC(K::Integer;
               nodes = :gausslobatto,
               linsolve = LUFactorization(),
               small_constant = small_constant_function_MPDeC)

    if small_constant isa Number
        small_constant_function = Returns(small_constant)
    else # assume small_constant isa Function
        small_constant_function = small_constant
    end

    if nodes === :lagrange
        M = K - 1
    else # :gausslobatto 
        M = cld(K, 2)
    end

    MPDeC{typeof(nodes), typeof(linsolve), typeof(small_constant_function)}(K, M,
                                                                            nodes,
                                                                            linsolve,
                                                                            small_constant_function)
end

alg_order(alg::MPDeC) = alg.K
isfsal(::MPDeC) = false

function get_constant_parameters(alg::MPDeC)
    if alg.nodes == :lagrange
        nodes = collect(0:(1 / alg.M):1)
        # M is one less than the methods order
        if alg.M == 1
            theta = [0 0.5; 0 0.5]
        elseif alg.M == 2
            theta = [0.0 0.20833333333333337 0.16666666666666652;
                     0.0 0.33333333333333337 0.6666666666666667;
                     0.0 -0.04166666666666667 0.16666666666666663]
        elseif alg.M == 3
            theta = [0.0 0.125 0.11111111111111116 0.125;
                     0.0 0.26388888888888895 0.4444444444444451 0.3750000000000009;
                     0.0 -0.0694444444444445 0.11111111111111072 0.375;
                     0.0 0.013888888888888888 5.551115123125783e-17 0.12499999999999978]
        elseif alg.M == 4
            theta = [0.0 0.08715277777777783 0.0805555555555556 0.08437500000000009 0.07777777777777928;
                     0.0 0.2243055555555556 0.34444444444444455 0.31874999999999964 0.35555555555555785;
                     0.0 -0.09166666666666667 0.06666666666666643 0.22499999999999964 0.13333333333333286;
                     0.0 0.036805555555555564 0.011111111111111016 0.1312500000000001 0.35555555555555607;
                     0.0 -0.0065972222222222265 -0.0027777777777778234 -0.009374999999999911 0.0777777777777775]
        elseif alg.M == 5
            theta = [0.0 0.06597222222222225 0.0622222222222224 0.06374999999999953 0.06222222222222573 0.06597222222222676;
                     0.0 0.1981944444444445 0.28666666666666707 0.2737500000000015 0.28444444444444983 0.2604166666666785;
                     0.0 -0.1108333333333335 0.031111111111109757 0.14249999999999297 0.10666666666664959 0.1736111111111054;
                     0.0 0.06694444444444442 0.031111111111110645 0.1424999999999983 0.2844444444444356 0.17361111111109295;
                     0.0 -0.024027777777777766 -0.013333333333332975 -0.026250000000000107 0.06222222222222484 0.2604166666666785;
                     0.0 0.0037500000000000033 0.0022222222222222365 0.003750000000000364 0.0 0.06597222222222454]
        elseif alg.M == 6
            theta = [0.0 0.05259865520282189 0.05022045855379187 0.05096726190476186 0.050440917107582806 0.05118772045855735 0.04880952380950987;
                     0.0 0.179431216931217 0.2486772486772483 0.24107142857142755 0.24550264550265144 0.239748677248647 0.25714285714280294;
                     0.0 -0.12803406084656088 0.0014550264550234893 0.08638392857142296 0.06772486772487873 0.08783895502639893 0.0321428571428477;
                     0.0 0.1033509700176369 0.058553791887126866 0.16190476190476288 0.26525573192239804 0.22045855379185753 0.3238095238095582;
                     0.0 -0.055696097883597806 -0.03558201058201127 -0.05424107142856682 0.030687830687839757 0.16017691798938927 0.03214285714290277;
                     0.0 0.017394179894179934 0.011640211640212117 0.01607142857142918 0.008465608465618502 0.07771164021165333 0.2571428571429166;
                     0.0 -0.00237819664902998 -0.0016313932980599258 -0.002157738095238018 -0.0014109347442683717 -0.003789131393296341 0.048809523809520694]
        elseif alg.M == 7
            theta = [0.0 0.04346064814814822 0.0418367346938779 0.04225127551020513 0.04202569916855836 0.042251275510214015 0.041836734693902144 0.04346064814816941;
                     0.0 0.1651655801209374 0.22161753590324995 0.21667729591836782 0.21889644746787695 0.21686626039310397 0.2204081632653896 0.20700231481467313;
                     0.0 -0.14384566326530607 -0.02414965986394635 0.04390943877551834 0.032653061224550584 0.04118835034029189 0.02755102040850943 0.0765625000003638;
                     0.0 0.1454235166288738 0.09251700680271657 0.18899872448978527 0.26969009826137835 0.24580144557796757 0.2775510204078273 0.17297453703622523;
                     0.0 -0.10457648337112646 -0.07282690854119345 -0.09671556122447367 -0.016024187452693184 0.08045753023436752 0.02755102040857693 0.17297453703748644;
                     0.0 0.04901147959183677 0.03537414965986363 0.04390943877552189 0.03265306122450795 0.1007121598638605 0.22040816326534696 0.07656249999990905;
                     0.0 -0.013405848450491272 -0.009863945578231281 -0.011894132653058165 -0.009674981103547253 -0.014615221088391195 0.04183673469395899 0.2070023148147584;
                     0.0 0.001623913454270594 0.0012093726379440242 0.0014349489795916492 0.0012093726379429626 0.0016239134542663791 -1.0658141036401503e-14 0.04346064814814099]
        elseif alg.M == 8
            theta = [0.0 0.03685850005511468 0.035688932980599594 0.03594029017857134 0.0358289241622568 0.035914937444895045 0.035803571428594694 0.03605492862659787 0.034885361552085214;
                     0.0 0.15387641920194012 0.2012610229276906 0.19782924107142996 0.1990828924162642 0.19819740685622378 0.1992857142857929 0.1969121334875581 0.2076895943564523;
                     0.0 -0.15861283344356245 -0.046840828924162636 0.009592633928562577 0.0021516754850381403 0.0065018050044045594 0.0016071428572104196 0.011744309413188603 -0.03273368606460281;
                     0.0 0.19274133322310427 0.13237213403880332 0.22303013392857451 0.2888183421517425 0.2741522679677928 0.2878571428567511 0.2618484760821502 0.3702292768929283;
                     0.0 -0.17337411816578485 -0.12799823633157104 -0.15669642857142208 -0.08007054673731773 -0.003444664903213379 -0.0321428571432989 0.013233024690180173 -0.16014109348088823;
                     0.0 0.10838080081569673 0.08237213403880084 0.09607700892858873 0.08141093474421268 0.14719914296711067 0.23785714285577342 0.1774879436707124 0.37022927689395146;
                     0.0 -0.044477995480599525 -0.03434082892416224 -0.03923549107142321 -0.03488536155200972 -0.04232631999563363 0.01410714285702852 0.1258791473759011 -0.03273368606702931;
                     0.0 0.010777460868606693 0.008403880070546516 0.009492187499996696 0.008606701940021111 0.009860353284821599 0.006428571428614305 0.053813175154459714 0.2076895943564523;
                     0.0 -0.0011695670745149895 -0.0009182098765432678 -0.0010295758928574872 -0.0009435626102302086 -0.0010549286265453262 -0.0008035714285732354 -0.0019731385031036552 0.03488536155197153]

        elseif alg.M == 9
            theta = [0.0 0.03188616071428564 0.031009210268469034 0.03117187499999874 0.031111111111105316 0.031149339236719698 0.031111111111073342 0.03117187499992724 0.031009210268393872 0.031886160714066136;
                     0.0 0.14467159330295903 0.18532725847540765 0.1828236607142859 0.18359396433471176 0.1831509191895293 0.18357142857178133 0.18292556155574857 0.18461297276007826 0.17568080357159488;
                     0.0 -0.1725594013325496 -0.06735057809132261 -0.0193750000000108 -0.02461297276119012 -0.022122403488083364 -0.02428571428754367 -0.021130829907633597 -0.02909660984732909 0.012053571401338559;
                     0.0 0.24498946698020774 0.1776641191456143 0.26335317460326735 0.3186204193613946 0.30879507152621954 0.31587301587512684 0.3064180384141082 0.3290926905915512 0.21589285713457684;
                     0.0 -0.2646060834313152 -0.20377621007249935 -0.23694196428579062 -0.16401332549457948 -0.10071817435664343 -0.11857142856752034 -0.098733067548892 -0.14234763861895772 0.06448660725436639;
                     0.0 0.20683424578679332 0.1632196747011676 0.18305803571432477 0.1652047815015294 0.22849993263764645 0.3014285714274365 0.26826281722094336 0.329092690586549 0.06448660718228894;
                     0.0 -0.11319983343131501 -0.09052518126592918 -0.09998015873019028 -0.09290221438379831 -0.10272756221900181 -0.04746031746481094 0.03822873799435911 -0.029096609840053134 0.21589285708614625;
                     0.0 0.04115018126592196 0.03318440133255063 0.036339285714291236 0.034175974916735186 0.03666654418941562 0.03142857142916 0.07940414951872299 0.18461297276189725 0.0120535714286234;
                     0.0 -0.008932169189692364 -0.007244757985499331 -0.007890625000002531 -0.007470115618263051 -0.00791316076327453 -0.007142857142902415 -0.009646454903545987 0.03100921026901915 0.17568080356954852;
                     0.0 0.0008769504458161903 0.0007142857142857367 0.000775049603174427 0.0007368214775622661 0.0007750496031686538 0.0007142857142810044 0.0008769504457859512 5.684341886080802e-14 0.03188616071417982]
        else
            error("MPDeC requires 2 ≤ K ≤ 10.")
        end
    else # alg.nodes == :gausslobatto 
        if alg.M == 1
            nodes = [0.0, 1.0]
            theta = [0.0 0.5; 0.0 0.5]
        elseif alg.M == 2
            nodes = [0.0, 0.5, 1.0]
            theta = [0.0 0.20833333333333337 0.16666666666666652;
                     0.0 0.33333333333333337 0.6666666666666667;
                     0.0 -0.04166666666666667 0.16666666666666663]
        elseif alg.M == 3
            nodes = [0.0, 0.27639320225002106, 0.7236067977499789, 1.0]
            theta = [0.0 0.11030056647916493 0.07303276685416865 0.08333333333333393;
                     0.0 0.1896994335208352 0.45057403089581083 0.41666666666666785;
                     0.0 -0.033907364229143935 0.22696723314583123 0.4166666666666661;
                     0.0 0.010300566479164913 -0.026967233145831604 0.08333333333333326]
        elseif alg.M == 4
            nodes = [0.0, 0.1726731646460114, 0.5, 0.8273268353539887, 1.0]
            theta = [0.0 0.0677284321861569 0.04062499999999991 0.05370013924241501 0.05000000000000071;
                     0.0 0.11974476934341162 0.30318418332304287 0.2615863979968083 0.27222222222222214;
                     0.0 -0.021735721866558116 0.17777777777777748 0.3772912774221129 0.35555555555555296;
                     0.0 0.010635824225415487 -0.0309619611008205 0.1524774528788102 0.27222222222222037;
                     0.0 -0.0037001392424145354 0.009375000000000022 -0.017728432186157494 0.04999999999999982]
        elseif alg.M == 5
            nodes = [
                0.0,
                0.11747233803526769,
                0.3573842417596774,
                0.6426157582403226,
                0.8825276619647323,
                1.0
            ]
            theta = [0.0 0.04567980513375505 0.025908385387879762 0.03746264288972734 0.03168990732937349 0.033333333333333215;
                     0.0 0.08186781700897068 0.2138408086328255 0.177429781771262 0.19370925858950017 0.18923747814892522;
                     0.0 -0.01487460578908985 0.13396073565086075 0.30143326325089315 0.2698015123994857 0.2774291885177327;
                     0.0 0.007627676118250971 -0.024004074733154912 0.14346845286688126 0.2923037943068376 0.27742918851774334;
                     0.0 -0.004471780440573713 0.011807696377659743 -0.02460333048390262 0.10736966113994595 0.18923747814891745;
                     0.0 0.0016434260039545345 -0.004129309556393734 0.007424947945453564 -0.012346471800418257 0.03333333333333588]
        else
            error("MPDeC requires 2 ≤ K ≤ 10.")
        end
    end
    return nodes, theta
end

struct MPDeCConstantCache{NType, T, T2} <: OrdinaryDiffEqConstantCache
    K::Int
    M::Int
    nodes::NType
    theta::T2
    small_constant::T
end

# Out-of-place
function alg_cache(alg::MPDeC, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    if !(f isa PDSFunction || f isa ConservativePDSFunction)
        throw(ArgumentError("MPDeC can only be applied to production-destruction systems"))
    end

    nodes, theta = get_constant_parameters(alg)
    MPDeCConstantCache(alg.K, alg.M, nodes, theta,
                       alg.small_constant_function(uEltypeNoUnits))
end

function initialize!(integrator, cache::MPDeCConstantCache)
end

# out-of-place
function build_mpdec_matrix_and_rhs_oop(uprev, m, f, C, p, t, dt, nodes, theta,
                                        small_constant)
    N = length(uprev)
    if f isa PDSFunction
        # Additional destruction terms 
        Mmat, rhs = _build_mpdec_matrix_and_rhs_oop(uprev, m, f.p, C, p, t, dt, nodes,
                                                    theta,
                                                    small_constant, f.d)
    else
        # No additional destruction terms 
        Mmat, rhs = _build_mpdec_matrix_and_rhs_oop(uprev, m, f.p, C, p, t, dt, nodes,
                                                    theta,
                                                    small_constant)
    end

    if uprev isa StaticArray
        return SMatrix{N, N}(Mmat), SVector{N}(rhs)
    else
        return Mmat, rhs
    end
end

# out-of-place for dense arrays
@muladd function _build_mpdec_matrix_and_rhs_oop(uprev, m, prod, C, p, t, dt, nodes, theta,
                                                 small_constant, dest = nothing)
    N, M = size(C)
    M = M - 1

    # Create linear system matrix and rhs
    if uprev isa StaticArray
        Mmat = MMatrix{N, N}(zeros(eltype(uprev), N, N))
    else
        Mmat = zeros(eltype(uprev), N, N)
    end
    rhs = similar(uprev)

    # Initialize
    oneMmat = one(eltype(Mmat))
    @inbounds for i in 1:N
        Mmat[i, i] = oneMmat
    end
    rhs .= uprev

    σ = add_small_constant(C[:, m], small_constant)

    @fastmath @inbounds @simd for r in 1:(M + 1)
        th = theta[r, m]
        dt_th = dt * th
        P = prod(C[:, r], p, t + nodes[r] * dt)
        if !isnothing(dest)
            d = dest(C[:, r], p, t + nodes[r] * dt)
        else
            d = nothing
        end
        _build_mpdec_matrix_and_rhs!(Mmat, rhs, P, dt_th, σ, d)
    end

    return Mmat, rhs
end

# in-place for dense arrays
@muladd function build_mpdec_matrix_and_rhs_ip!(Mmat, rhs, m, prod, P, C, p, t, dt, σ, tmp,
                                                nodes, theta, small_constant,
                                                dest = nothing, d = nothing)
    N, M = size(C)
    M = M - 1

    oneMmat = one(eltype(Mmat))
    zeroMmat = zero(eltype(Mmat))

    #Initialize Mmat as identity matrix
    if Mmat isa Tridiagonal
        Mmat.d .= oneMmat
        Mmat.du .= zeroMmat
        Mmat.dl .= zeroMmat
    elseif issparse(Mmat)
        # Fill sparse matrix with zeros without changing the sparsity pattern, see
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/190#issuecomment-1186690008.
        fill!(Mmat.nzval, false)

        M_rows = rowvals(Mmat)
        M_vals = nonzeros(Mmat)
        for j in 1:N
            for idx_M in nzrange(Mmat, j)
                i = M_rows[idx_M]
                if i == j
                    M_vals[idx_M] = oneMmat
                end
            end
        end
    else
        fill!(Mmat, zeroMmat)
        @inbounds for i in 1:N
            Mmat[i, i] = oneMmat
        end
    end

    σ .= C[:, m] .+ small_constant

    @fastmath @inbounds @simd for r in 1:(M + 1)
        th = theta[r, m]
        dt_th = dt * th

        prod(P, C[:, r], p, t + nodes[r] * dt)
        if !isnothing(dest)
            dest(d, C[:, r], p, t + nodes[r] * dt)
        end

        if issparse(Mmat)
            _build_mpdec_matrix_and_rhs!(Mmat, rhs, P, dt_th, σ, d, tmp)
        else
            _build_mpdec_matrix_and_rhs!(Mmat, rhs, P, dt_th, σ, d)
        end
    end
end

function _build_mpdec_matrix_and_rhs!(M, rhs, P, dt_th, σ, d = nothing)
    Base.require_one_based_indexing(M, P, σ)
    @assert size(M, 1) == size(M, 2) == size(P, 1) == size(P, 2) == length(σ)

    if dt_th ≥ 0
        @fastmath @inbounds @simd for I in CartesianIndices(P)
            if !iszero(P[I])
                dt_th_P = dt_th * P[I]
                if I[1] != I[2]
                    M[I] -= dt_th_P / σ[I[2]]
                    M[I[2], I[2]] += dt_th_P / σ[I[2]]
                else # diagonal elements
                    rhs[I[1]] += dt_th_P
                end
            end
        end

        if !isnothing(d)
            @fastmath @inbounds @simd for i in eachindex(d)
                if !iszero(d[i])
                    M[i, i] += dt_th * d[i] / σ[i]
                end
            end
        end
    else # dt_th ≤ 0
        @fastmath @inbounds @simd for I in CartesianIndices(P)
            if !iszero(P[I])
                dt_th_P = dt_th * P[I]
                if I[1] != I[2]
                    M[I[2], I[1]] += dt_th_P / σ[I[1]]
                    M[I[1], I[1]] -= dt_th_P / σ[I[1]]
                else # diagonal elements
                    M[I] -= dt_th_P / σ[I[1]]
                end
            end
        end

        if !isnothing(d)
            @fastmath @inbounds @simd for i in eachindex(d)
                if !iszero(d[i])
                    rhs[i] -= dt_th * d[i]
                end
            end
        end
    end
end

# optimized version for Tridiagonal matrices
function _build_mpdec_matrix_and_rhs!(M::Tridiagonal, rhs, P::Tridiagonal, dt_th, σ,
                                      d = nothing)
    Base.require_one_based_indexing(M.dl, M.d, M.du, P.dl, P.d, P.du, σ)
    @assert length(M.dl) + 1 == length(M.d) == length(M.du) + 1 ==
            length(P.dl) + 1 == length(P.d) == length(P.du) + 1 == length(σ)

    if dt_th ≥ 0
        @fastmath @inbounds @simd for i in eachindex(P.d, rhs)
            rhs[i] += dt_th * P.d[i]
        end

        for i in eachindex(M.dl, P.dl)
            dt_th_P = dt_th * P.dl[i]
            M.dl[i] -= dt_th_P / σ[i]
            M.d[i] += dt_th_P / σ[i]
        end

        for i in eachindex(M.du, P.du)
            dt_th_P = dt_th * P.du[i]
            M.du[i] -= dt_th_P / σ[i + 1]
            M.d[i + 1] += dt_th_P / σ[i + 1]
        end

        if !isnothing(d)
            @fastmath @inbounds @simd for i in eachindex(M.d, σ, d)
                M.d[i] += dt_th * d[i] / σ[i]
            end
        end
    else # dt_th ≤ 0
        @fastmath @inbounds @simd for i in eachindex(M.d, P.d, σ)
            M.d[i] -= dt_th * P.d[i] / σ[i]
        end

        for i in eachindex(M.dl, P.dl)
            dt_th_P = dt_th * P.dl[i]
            M.du[i] += dt_th_P / σ[i + 1]
            M.d[i + 1] -= dt_th_P / σ[i + 1]
        end

        for i in eachindex(M.du, P.du)
            dt_th_P = dt_th * P.du[i]
            M.dl[i] += dt_th_P / σ[i]
            M.d[i] -= dt_th_P / σ[i]
        end

        if !isnothing(d)
            @fastmath @inbounds @simd for i in eachindex(rhs, d)
                rhs[i] -= dt_th * d[i]
            end
        end
    end
end

# optimized version for sparse matrices
function _build_mpdec_matrix_and_rhs!(M::AbstractSparseMatrix, rhs, P::AbstractSparseMatrix,
                                      dt_th, σ,
                                      d = nothing, tmp = nothing)
    Base.require_one_based_indexing(M, P, σ)
    @assert size(M, 1) == size(M, 2) == size(P, 1) == size(P, 2) == length(σ)
    if !isnothing(d)
        Base.require_one_based_indexing(d)
        @assert length(σ) == length(d)
    end

    # By construction M and P share the same sparsity pattern.
    M_rows = rowvals(M)
    M_vals = nonzeros(M)
    P_rows = rowvals(P)
    P_vals = nonzeros(P)
    n = size(M, 2)

    # tmp[j] = M[j,j]
    fill!(tmp, zero(eltype(tmp)))

    if dt_th ≥ 0
        for j in 1:n # run through columns of P  
            for idx_P in nzrange(P, j) # run through rows of P
                i = P_rows[idx_P]
                dt_th_P = dt_th * P_vals[idx_P]
                if i != j
                    for idx_M in nzrange(M, j)
                        if M_rows[idx_M] == i
                            M_vals[idx_M] -= dt_th_P / σ[j] # M_ij <- P_ij 
                            #break
                        end
                    end
                    tmp[j] += dt_th_P / σ[j] # M_jj <- P_ij = D_ji
                else
                    rhs[i] += dt_th_P # rhs_i <- P_ii
                end
            end
        end

        if !isnothing(d)
            for i in eachindex(d)
                tmp[i] += dt_th * d[i] / σ[i] # M_ii <- D_i
            end
        end

        for j in 1:n
            for idx_M in nzrange(M, j)
                i = M_rows[idx_M]
                if i == j
                    M_vals[idx_M] += tmp[j]
                    #break
                end
            end
        end
    else # dt ≤ 0
        for j in 1:n # j is column index 
            for idx_P in nzrange(P, j)
                i = P_rows[idx_P] # i is row index 
                dt_th_P = dt_th * P_vals[idx_P]
                if i != j
                    for idx_M in nzrange(M, i)
                        if M_rows[idx_M] == j
                            M_vals[idx_M] += dt_th_P / σ[i] # M_ji <- P_ij
                        end
                        #break
                    end
                    tmp[i] -= dt_th_P / σ[i]
                else
                    for idx_M in nzrange(M, j)
                        if i == M_rows[idx_M]
                            M_vals[idx_M] -= dt_th_P / σ[i] # M_ij <- P_ij
                            #break
                        end
                    end
                end
            end
        end

        for j in 1:n
            for idx_M in nzrange(M, j)
                i = M_rows[idx_M]
                if i == j
                    M_vals[idx_M] += tmp[j]
                    #break
                end
            end
        end

        if !isnothing(d)
            @.. broadcast=false rhs-=dt_th * d
        end
    end
end

@muladd function perform_step!(integrator, cache::MPDeCConstantCache, repeat_step = false)
    @unpack alg, t, dt, uprev, f, p = integrator
    @unpack K, M, nodes, theta, small_constant = cache

    N = length(uprev)

    if uprev isa StaticArray
        C = MMatrix{N, M + 1}(zeros(N, M + 1))
        C2 = MMatrix{N, M + 1}(zeros(N, M + 1))
    else
        C = zeros(N, M + 1)
        C2 = zeros(N, M + 1)
    end

    for i in 1:(M + 1)
        C2[:, i] = uprev
    end

    for _ in 1:K
        C .= C2
        for m in 2:(M + 1)
            Mmat, rhs = build_mpdec_matrix_and_rhs_oop(uprev, m, f, C, p, t, dt, nodes,
                                                       theta,
                                                       small_constant)
            # solve linear system
            linprob = LinearProblem(Mmat, rhs)
            sol = solve(linprob, alg.linsolve)
            C2[:, m] = sol.u
            integrator.stats.nsolve += 1
        end
    end
    u = C2[:, M + 1]
    u1 = C[:, M + 1] # one order less accurate

    tmp = u - u1
    atmp = calculate_residuals(tmp, uprev, u, integrator.opts.abstol,
                               integrator.opts.reltol, integrator.opts.internalnorm, t)
    integrator.EEst = integrator.opts.internalnorm(atmp, t)

    integrator.u = u
end

struct MPDeCCache{uType, PType, CType, tabType, F} <: MPRKMutableCache
    tmp::uType
    P::PType
    P2::PType
    d::uType
    σ::uType
    C::CType
    C2::CType
    tab::tabType
    linsolve_rhs::uType
    linsolve::F
end

struct MPDeCConservativeCache{uType, PType, CType, tabType, F} <: MPRKMutableCache
    tmp::uType
    P::PType
    P2::PType
    σ::uType
    C::CType
    C2::CType
    tab::tabType
    linsolve_rhs::uType
    linsolve::F
end

get_tmp_cache(integrator, ::MPDeC, cache::OrdinaryDiffEqMutableCache) = (cache.σ,)

# In-place
function alg_cache(alg::MPDeC, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                   uprev, uprev2, f, t, dt, reltol, p, calck,
                   ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    nodes, theta = get_constant_parameters(alg)
    tab = MPDeCConstantCache(alg.K, alg.M, nodes, theta,
                             alg.small_constant_function(uEltypeNoUnits))

    tmp = zero(u)
    P = p_prototype(u, f) # stores evaluation of the production matrix
    P2 = p_prototype(u, f) # stores the linear system matrix
    if issparse(P2)
        # We need to ensure that evaluating the production function does
        # not alter the sparsity pattern given by the production matrix prototype 
        f.p(P2, uprev, p, t)
        @assert P.rowval == P2.rowval&&P.colptr == P2.colptr "Evaluation of the production terms must not alter the sparsity pattern given by the prototype."

        if alg.K > 2
            # Negative weights of MPDeC(K) , K > 2 require
            # a symmetric sparsity pattern of the linear system matrix P2 
            P2 = P + P'
        end
    end
    d = zero(u)
    σ = zero(u)
    C = zeros(eltype(u), length(u), alg.M + 1)
    C2 = zeros(eltype(u), length(u), alg.M + 1)
    linsolve_rhs = zero(u)

    if f isa ConservativePDSFunction
        # The right hand side of the linear system is always uprev. But using
        # linsolve_rhs instead of uprev for the rhs we allow `alias_b=true`. uprev must
        # not be altered, since it is needed to compute the adaptive time step
        # size.
        linprob = LinearProblem(P2, _vec(linsolve_rhs))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        MPDeCConservativeCache(tmp, P, P2, σ, C, C2,
                               tab, #MPDeCConstantCache
                               linsolve_rhs,
                               linsolve)
    elseif f isa PDSFunction
        linprob = LinearProblem(P2, _vec(linsolve_rhs))
        linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                        assumptions = LinearSolve.OperatorAssumptions(true))

        MPDeCCache(tmp, P, P2, d, σ, C, C2,
                   tab, #MPDeCConstantCache
                   linsolve_rhs,
                   linsolve)
    else
        throw(ArgumentError("MPDeC can only be applied to production-destruction systems"))
    end
end

function initialize!(integrator, cache::Union{MPDeCCache, MPDeCConservativeCache})
end

@muladd function perform_step!(integrator, cache::MPDeCCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, P, P2, d, σ, C, C2, linsolve_rhs, linsolve = cache
    @unpack K, M, nodes, theta, small_constant = cache.tab

    # Initialize C matrices
    for i in 1:(M + 1)
        C2[:, i] .= uprev
    end

    for _ in 1:K
        C .= C2
        for m in 2:(M + 1)
            linsolve_rhs .= uprev
            build_mpdec_matrix_and_rhs_ip!(P2, linsolve_rhs, m, f.p, P, C, p, t, dt, σ, tmp,
                                           nodes,
                                           theta,
                                           small_constant, f.d, d)

            # Same as linres = P2 \ linsolve_rhs
            linsolve.A = P2
            linres = solve!(linsolve)
            C2[:, m] .= linres
            integrator.stats.nsolve += 1
        end
    end

    u .= C2[:, M + 1]
    σ .= C[:, M + 1] # one order less accurate

    # Now σ stores the error estimate
    @.. broadcast=false σ=u - σ

    # Now tmp stores error residuals
    calculate_residuals!(tmp, σ, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp, t)
end

@muladd function perform_step!(integrator, cache::MPDeCConservativeCache,
                               repeat_step = false)
    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, P, P2, σ, C, C2, linsolve_rhs, linsolve = cache
    @unpack K, M, nodes, theta, small_constant = cache.tab

    # Initialize right hand side of linear system
    linsolve_rhs .= uprev

    # Initialize C matrices
    for i in 1:(M + 1)
        C2[:, i] .= uprev
    end

    for _ in 1:K
        C .= C2
        for m in 2:(M + 1)
            linsolve_rhs .= uprev
            build_mpdec_matrix_and_rhs_ip!(P2, linsolve_rhs, m, f.p, P, C, p, t, dt, σ, tmp,
                                           nodes,
                                           theta,
                                           small_constant)

            # Same as linres = P2 \ linsolve_rhs
            linsolve.A = P2
            linres = solve!(linsolve)
            C2[:, m] .= linres
            integrator.stats.nsolve += 1
        end
    end

    u .= C2[:, M + 1]
    σ .= C[:, M + 1] # one order less accurate

    # Now σ stores the error estimate
    @.. broadcast=false σ=u - σ

    # Now tmp stores error residuals
    calculate_residuals!(tmp, σ, uprev, u, integrator.opts.abstol,
                         integrator.opts.reltol, integrator.opts.internalnorm, t,
                         False())
    integrator.EEst = integrator.opts.internalnorm(tmp, t)
end
