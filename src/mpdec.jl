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

function get_constant_parameters(alg::MPDeC, type)
    oneType = one(type)
    if alg.nodes == :lagrange
        nodes = collect(i * oneType / alg.M for i in 0:(alg.M))
        # M is one less than the methods order
        if alg.M == 1
            theta = [0 oneType/2; 0 oneType/2]
        elseif alg.M == 2
            theta = [0 5oneType/24 oneType/6;
                     0 oneType/3 2oneType/3;
                     0 -oneType/24 oneType/6]
        elseif alg.M == 3
            theta = [0 1oneType/8 1oneType/9 1oneType/8;
                     0 19oneType/72 4oneType/9 3oneType/8;
                     0 -5oneType/72 1oneType/9 3oneType/8;
                     0 1oneType/72 0 1oneType/8]
        elseif alg.M == 4
            theta = th2 = [0 251oneType/2880 29oneType/360 27oneType/320 7oneType/90;
                           0 323oneType/1440 31oneType/90 51oneType/160 16oneType/45;
                           0 -11oneType/120 1oneType/15 9oneType/40 2oneType/15;
                           0 53oneType/1440 1oneType/90 21oneType/160 16oneType/45;
                           0 -19oneType/2880 -1oneType/360 -3oneType/320 7oneType/90]
        elseif alg.M == 5
            theta = [0 19oneType/288 14oneType/225 51oneType/800 14oneType/225 19oneType/288;
                     0 1427oneType/7200 43oneType/150 219oneType/800 64oneType/225 25oneType/96;
                     0 -133oneType/1200 7oneType/225 57oneType/400 8oneType/75 25oneType/144;
                     0 241oneType/3600 7oneType/225 57oneType/400 64oneType/225 25oneType/144;
                     0 -173oneType/7200 -1oneType/75 -21oneType/800 14oneType/225 25oneType/96;
                     0 3oneType/800 1oneType/450 3oneType/800 0 19oneType/288]
        elseif alg.M == 6
            theta = [0 19087oneType/362880 1139oneType/22680 137oneType/2688 143oneType/2835 3715oneType/72576 41oneType/840;
                     0 2713oneType/15120 47oneType/189 27oneType/112 232oneType/945 725oneType/3024 9oneType/35;
                     0 -15487oneType/120960 11oneType/7560 387oneType/4480 64oneType/945 2125oneType/24192 9oneType/280;
                     0 293oneType/2835 166oneType/2835 17oneType/105 752oneType/2835 125oneType/567 34oneType/105;
                     0 -6737oneType/120960 -269oneType/7560 -243oneType/4480 29oneType/945 3875oneType/24192 9oneType/280;
                     0 263oneType/15120 11oneType/945 9oneType/560 8oneType/945 235oneType/3024 9oneType/35;
                     0 -863oneType/362880 -37oneType/22680 -29oneType/13440 -4oneType/2835 -275oneType/72576 41oneType/840]
        elseif alg.M == 7
            theta = [0 751oneType/17280 41oneType/980 265oneType/6272 278oneType/6615 265oneType/6272 41oneType/980 751oneType/17280;
                     0 139849oneType/846720 1466oneType/6615 1359oneType/6272 1448oneType/6615 36725oneType/169344 54oneType/245 3577oneType/17280;
                     0 -4511oneType/31360 -71oneType/2940 1377oneType/31360 8oneType/245 775oneType/18816 27oneType/980 49oneType/640;
                     0 123133oneType/846720 68oneType/735 5927oneType/31360 1784oneType/6615 4625oneType/18816 68oneType/245 2989oneType/17280;
                     0 -88547oneType/846720 -1927oneType/26460 -3033oneType/31360 -106oneType/6615 13625oneType/169344 27oneType/980 2989oneType/17280;
                     0 1537oneType/31360 26oneType/735 1377oneType/31360 8oneType/245 1895oneType/18816 54oneType/245 49oneType/640;
                     0 -11351oneType/846720 -29oneType/2940 -373oneType/31360 -64oneType/6615 -275oneType/18816 41oneType/980 3577oneType/17280;
                     0 275oneType/169344 8oneType/6615 9oneType/6272 8oneType/6615 275oneType/169344 0 751oneType/17280]
        elseif alg.M == 8
            theta = [0 1070017oneType/29030400 32377oneType/907200 12881oneType/358400 4063oneType/113400 41705oneType/1161216 401oneType/11200 149527oneType/4147200 989oneType/28350;
                     0 2233547oneType/14515200 22823oneType/113400 35451oneType/179200 2822oneType/14175 115075oneType/580608 279oneType/1400 408317oneType/2073600 2944oneType/14175;
                     0 -2302297oneType/14515200 -21247oneType/453600 1719oneType/179200 61oneType/28350 3775oneType/580608 9oneType/5600 24353oneType/2073600 -464oneType/14175;
                     0 2797679oneType/14515200 15011oneType/113400 39967oneType/179200 4094oneType/14175 159175oneType/580608 403oneType/1400 542969oneType/2073600 5248oneType/14175;
                     0 -31457oneType/181440 -2903oneType/22680 -351oneType/2240 -227oneType/2835 -125oneType/36288 -9oneType/280 343oneType/25920 -454oneType/2835;
                     0 1573169oneType/14515200 9341oneType/113400 17217oneType/179200 1154oneType/14175 85465oneType/580608 333oneType/1400 368039oneType/2073600 5248oneType/14175;
                     0 -645607oneType/14515200 -15577oneType/453600 -7031oneType/179200 -989oneType/28350 -24575oneType/580608 79oneType/5600 261023oneType/2073600 -464oneType/14175;
                     0 156437oneType/14515200 953oneType/113400 243oneType/25600 122oneType/14175 5725oneType/580608 9oneType/1400 111587oneType/2073600 2944oneType/14175;
                     0 -33953oneType/29030400 -119oneType/129600 -369oneType/358400 -107oneType/113400 -175oneType/165888 -9oneType/11200 -8183oneType/4147200 989oneType/28350]

        elseif alg.M == 9
            theta = [0 2857oneType/89600 3956oneType/127575 399oneType/12800 7oneType/225 81385oneType/2612736 7oneType/225 399oneType/12800 3956oneType/127575 2857oneType/89600;
                     0 9449717oneType/65318400 37829oneType/204120 16381oneType/89600 3346oneType/18225 478525oneType/2612736 257oneType/1400 341383oneType/1866240 23552oneType/127575 15741oneType/89600;
                     0 -1408913oneType/8164800 -34369oneType/510300 -31oneType/1600 -628oneType/25515 -7225oneType/326592 -17oneType/700 -24647oneType/1166400 -3712oneType/127575 27oneType/2240;
                     0 200029oneType/816480 45331oneType/255150 13273oneType/50400 40648oneType/127575 50425oneType/163296 199oneType/630 178703oneType/583200 41984oneType/127575 1209oneType/5600;
                     0 -8641823oneType/32659200 -103987oneType/510300 -2123oneType/8960 -20924oneType/127575 -131575oneType/1306368 -83oneType/700 -460649oneType/4665600 -3632oneType/25515 2889oneType/44800;
                     0 6755041oneType/32659200 83291oneType/510300 8201oneType/44800 21076oneType/127575 298505oneType/1306368 211oneType/700 1251607oneType/4665600 41984oneType/127575 2889oneType/44800;
                     0 -462127oneType/4082400 -9239oneType/102060 -5039oneType/50400 -11852oneType/127575 -16775oneType/163296 -299oneType/6300 4459oneType/116640 -3712oneType/127575 1209oneType/5600;
                     0 335983oneType/8164800 8467oneType/255150 407oneType/11200 872oneType/25515 11975oneType/326592 11oneType/350 92617oneType/1166400 23552oneType/127575 27oneType/2240;
                     0 -116687oneType/13063680 -3697oneType/510300 -101oneType/12800 -953oneType/127575 -20675oneType/2612736 -1oneType/140 -90013oneType/9331200 3956oneType/127575 15741oneType/89600;
                     0 8183oneType/9331200 1oneType/1400 25oneType/32256 94oneType/127575 25oneType/32256 1oneType/1400 8183oneType/9331200 0 2857oneType/89600]

        else
            error("MPDeC requires 2 ≤ K ≤ 10.")
        end
    else # alg.nodes == :gausslobatto 
        if alg.M == 1
            nodes = [0, oneType]
            theta = [0 oneType/2; 0 oneType/2]
        elseif alg.M == 2
            nodes = [0, oneType / 2, oneType]
            theta = [0 5oneType/24 oneType/6;
                     0 oneType/3 2oneType/3;
                     0 -oneType/24 oneType/6]
        elseif alg.M == 3
            nodes = [
                0,
                (1 - sqrt(oneType / 5)) / 2,
                (1 + sqrt(oneType / 5)) / 2,
                oneType
            ]
            theta = [0 (11 + sqrt(5oneType))/120 (11 - sqrt(5oneType))/120 oneType/12;
                     0 (25 - sqrt(5oneType))/120 (13 * sqrt(5oneType) + 25)/120 5oneType/12;
                     0 (25 - 13 * sqrt(5oneType))/120 (sqrt(5oneType) + 25)/120 5oneType/12;
                     0 (sqrt(5oneType) - 1)/120 (-sqrt(5oneType) - 1)/120 oneType/12]
        elseif alg.M == 4
            nodes = [
                0,
                (1 - sqrt(3oneType / 7)) / 2,
                oneType / 2,
                (1 + sqrt(3oneType / 7)) / 2,
                oneType
            ]

            theta = [0 (3sqrt(21oneType) + 119)/1960 13oneType/320 (119 - 3sqrt(21oneType))/1960 oneType/20;
                     0 49oneType / 360-sqrt(21oneType) / 280 (7sqrt(21oneType)) / 192+49oneType / 360 (23sqrt(21oneType)) / 840+49oneType / 360 49oneType/180;
                     0 8oneType / 45-(32sqrt(21oneType)) / 735 8oneType/45 (32sqrt(21oneType)) / 735+8oneType / 45 16oneType/45;
                     0 49oneType / 360-(23sqrt(21oneType)) / 840 49oneType / 360-(7sqrt(21oneType)) / 192 sqrt(21oneType) / 280+49oneType / 360 49oneType/180;
                     0 ((3sqrt(21oneType)) / 1960-3oneType / 280) 3oneType/320 (-(3sqrt(21oneType)) / 1960-3oneType / 280) 1oneType/20]

        elseif alg.M == 5
            nodes = [
                0,
                oneType / 2 - sqrt((2sqrt(7oneType)) / 21 + oneType / 3) / 2,
                oneType / 2 - sqrt(oneType / 3 - (2sqrt(7oneType)) / 21) / 2,
                oneType / 2 + sqrt(oneType / 3 - (2sqrt(7oneType)) / 21) / 2,
                oneType / 2 + sqrt((2sqrt(7oneType)) / 21 + oneType / 3) / 2,
                oneType
            ]

            theta = [0 (sqrt(7oneType) / 756 + sqrt(6sqrt(7oneType) + 21) / 540 - sqrt(42sqrt(7oneType) + 147) / 3780+19oneType / 540) (19oneType / 540 - sqrt(21 - 6sqrt(7oneType)) / 540 - sqrt(147 - 42sqrt(7oneType)) / 3780-sqrt(7oneType) / 756) (sqrt(21 - 6sqrt(7oneType)) / 540-sqrt(7oneType) / 756+sqrt(147 - 42sqrt(7oneType))/3780+19oneType/540) (sqrt(7oneType) / 756-sqrt(6sqrt(7oneType) + 21) / 540+sqrt(42sqrt(7oneType) + 147)/3780+19oneType/540) oneType/30;
                     0 (7oneType / 60 - sqrt(42sqrt(7oneType) + 147) / 1260-sqrt(7oneType) / 120) (-((sqrt(3oneType) - 2sqrt(21oneType)) * (63sqrt(21oneType) + 35sqrt(2sqrt(7oneType) + 7) + 70sqrt(6sqrt(7oneType) + 21) + 20sqrt(14sqrt(7oneType) + 49) - 23sqrt(42sqrt(7oneType) + 147)))/22680) (-((sqrt(3oneType) - 2sqrt(21oneType)) * (63sqrt(21oneType) + 35sqrt(2sqrt(7oneType) + 7) - 70sqrt(6sqrt(7oneType) + 21) + 20sqrt(14sqrt(7oneType) + 49) + 23sqrt(42sqrt(7oneType) + 147)))/22680) (sqrt(42sqrt(7oneType) + 147) / 60 - sqrt(6sqrt(7oneType) + 21) / 36 - sqrt(7oneType) / 120+7oneType / 60) (7oneType / 30-sqrt(7oneType) / 60);
                     0 (-((sqrt(3oneType) + 2sqrt(21oneType)) * (21sqrt(2sqrt(7oneType) + 7) - 63sqrt(21oneType) + 70sqrt(6sqrt(7oneType) + 21) + 24 * sqrt(14sqrt(7oneType) + 49) - 25sqrt(42sqrt(7oneType) + 147)))/22680) (sqrt(7oneType) / 120 - sqrt(147 - 42sqrt(7oneType)) / 1260+7oneType / 60) (sqrt(7oneType)/120+sqrt(21 - 6sqrt(7oneType))/36+sqrt(147 - 42sqrt(7oneType))/60+7oneType/60) (((sqrt(3oneType) + 2sqrt(21oneType)) * (63sqrt(21oneType) + 21sqrt(2sqrt(7oneType) + 7) - 70sqrt(6sqrt(7oneType) + 21) + 24 * sqrt(14sqrt(7oneType) + 49) + 25sqrt(42sqrt(7oneType) + 147)))/22680) (sqrt(7oneType) / 60+7oneType / 30);
                     0 (-((sqrt(3oneType) + 2sqrt(21oneType)) * (21sqrt(2sqrt(7oneType) + 7) - 63sqrt(21oneType) - 70sqrt(6sqrt(7oneType) + 21) + 24 * sqrt(14sqrt(7oneType) + 49) + 25sqrt(42sqrt(7oneType) + 147)))/22680) (sqrt(7oneType) / 120 - sqrt(21 - 6sqrt(7oneType)) / 36 - sqrt(147 - 42sqrt(7oneType)) / 60+7oneType / 60) (((sqrt(7oneType) + 14) * (32 * sqrt(2sqrt(7oneType) + 7) - 10sqrt(14sqrt(7oneType) + 49) + 567))/68040) (((sqrt(3oneType) + 2sqrt(21oneType)) * (63sqrt(21oneType) + 21sqrt(2sqrt(7oneType) + 7) + 70sqrt(6sqrt(7oneType) + 21) + 24 * sqrt(14sqrt(7oneType) + 49) - 25sqrt(42sqrt(7oneType) + 147)))/22680) (sqrt(7oneType) / 60+7oneType / 30);
                     0 (sqrt(6sqrt(7oneType) + 21) / 36 - sqrt(7oneType) / 120 - sqrt(42sqrt(7oneType) + 147) / 60+7oneType / 60) (((sqrt(3oneType) - 2sqrt(21oneType)) * (35sqrt(2sqrt(7oneType) + 7) - 63sqrt(21oneType) - 70sqrt(6sqrt(7oneType) + 21) + 20sqrt(14sqrt(7oneType) + 49) + 23sqrt(42sqrt(7oneType) + 147)))/22680) (((sqrt(3oneType) - 2sqrt(21oneType)) * (35sqrt(2sqrt(7oneType) + 7) - 63sqrt(21oneType) + 70sqrt(6sqrt(7oneType) + 21) + 20sqrt(14sqrt(7oneType) + 49) - 23sqrt(42sqrt(7oneType) + 147)))/22680) (sqrt(42sqrt(7oneType) + 147) / 1260 - sqrt(7oneType) / 120+7oneType / 60) (7oneType / 30-sqrt(7oneType) / 60);
                     0 (sqrt(6sqrt(7oneType) + 21) / 540 - sqrt(7oneType) / 756 - sqrt(42sqrt(7oneType) + 147) / 3780-oneType / 540) (sqrt(7oneType) / 756 - sqrt(21 - 6sqrt(7oneType)) / 540 - sqrt(147 - 42sqrt(7oneType)) / 3780-oneType / 540) (sqrt(7oneType) / 756 + sqrt(21 - 6sqrt(7oneType)) / 540 + sqrt(147 - 42sqrt(7oneType)) / 3780-oneType / 540) (sqrt(42sqrt(7oneType) + 147) / 3780 - sqrt(6sqrt(7oneType) + 21) / 540 - sqrt(7oneType) / 756-oneType / 540) oneType/30]
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

    nodes, theta = get_constant_parameters(alg, uEltypeNoUnits)
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

    M_rows = rowvals(M)
    M_vals = nonzeros(M)
    P_rows = rowvals(P)
    P_vals = nonzeros(P)
    n = size(M, 2)

    # We use tmp as a buffer for the diagonal elements of M.
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
                            break
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
                    break
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
                            break
                        end
                    end
                    tmp[i] -= dt_th_P / σ[i]
                else
                    for idx_M in nzrange(M, j)
                        if i == M_rows[idx_M]
                            M_vals[idx_M] -= dt_th_P / σ[i] # M_ij <- P_ij
                            break
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
                    break
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
    nodes, theta = get_constant_parameters(alg, uEltypeNoUnits)
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
