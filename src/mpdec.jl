#TODO: Comment on choice of small_consant 
"""
    MPDeC(K; [nodes = :gausslobatto, linsolve = ..., small_constant = ...])

A family of arbitrary order modified Patankar-Runge-Kutta algorithms for
production-destruction systems. Each member of this family is an adaptive, one-step method which is
Kth order accurate, unconditionally positivity-preserving, and linearly
implicit. The integer K must be chosen to satisfy 2 ≤ K ≤ 10. 
Available node choices are Lagrange or Gauss-Lobatto nodes, with the latter being the default.
These methods support adaptive time stepping using the numerical solution obtained with one correction step less, as lower order approximation to estimate the error.

The MPDeC schemes were introduced by Torlo and Öffner (2020) for autonomous conservative production-destruction systems.
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
`floatmin` of the floating point type used.

## References

- Davide Torlo, and Philipp Öffner.
  "Arbitrary high-order, conservative and positivity preserving Patankar-type deferred correction schemes."
  Applied Numerical Mathematics 153 (2020): 15-34.
- Benjamin W. Ong & Raymond J. Spiteri.
  "Deferred Correction Methods for Ordinary Differential Equations."
  Journal of Scientific Computing 83 (2020): Article 60
"""
struct MPDeC{T, N, F, T2} <: OrdinaryDiffEqAdaptiveAlgorithm
    K::T
    M::T
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

function MPDeC(K; nodes = :gausslobatto, linsolve = LUFactorization(),
               small_constant = small_constant_function_MPDeC)
    if !(isinteger(K))
        throw(ArgumentError("MPDeC requires the parameter K to be an integer."))
    end
    if !(typeof(K) <: Integer)
        K = Int(K)
    end

    if small_constant isa Number
        small_constant_function = Returns(small_constant)
    else # assume small_constant isa Function
        small_constant_function = small_constant
    end

    if nodes == :lagrange
        M = K - 1
    else # :gausslobatto 
        M = ceil(Integer, K / 2)
    end

    MPDeC{typeof(K), typeof(nodes), typeof(linsolve), typeof(small_constant_function)}(K, M,
                                                                                       nodes,
                                                                                       linsolve,
                                                                                       small_constant_function)
end

alg_order(alg::MPDeC) = alg.K
isfsal(::MPDeC) = false

function get_constant_parameters(alg::MPDeC)
    if alg.nodes == :lagrange
        nodes = collect(0.0:(1 / alg.M):1.0)
        if alg.M == 1
            theta = [0.0 0.5; 0.0 0.5]
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
            #=
            theta = [0.0 0.03685850005511468 0.035688932980599594 0.03594029017857134 0.0358289241622568 0.035914937444895045 0.035803571428594694 0.03605492862659787 0.034885361552085214;
                     0.0 0.15387641920194012 0.2012610229276906 0.19782924107142996 0.1990828924162642 0.19819740685622378 0.1992857142857929 0.1969121334875581 0.2076895943564523;
                     0.0 -0.15861283344356245 -0.046840828924162636 0.009592633928562577 0.0021516754850381403 0.0065018050044045594 0.0016071428572104196 0.011744309413188603 -0.03273368606460281;
                     0.0 0.19274133322310427 0.13237213403880332 0.22303013392857451 0.2888183421517425 0.2741522679677928 0.2878571428567511 0.2618484760821502 0.3702292768929283;
                     0.0 -0.17337411816578485 -0.12799823633157104 -0.15669642857142208 -0.08007054673731773 -0.003444664903213379 -0.0321428571432989 0.013233024690180173 -0.16014109348088823;
                     0.0 0.10838080081569673 0.08237213403880084 0.09607700892858873 0.08141093474421268 0.14719914296711067 0.23785714285577342 0.1774879436707124 0.37022927689395146;
                     0.0 -0.044477995480599525 -0.03434082892416224 -0.03923549107142321 -0.03488536155200972 -0.04232631999563363 0.01410714285702852 0.1258791473759011 -0.03273368606702931;
                     0.0 0.010777460868606693 0.008403880070546516 0.009492187499996696 0.008606701940021111 0.009860353284821599 0.006428571428614305 0.053813175154459714 0.2076895943564523;
                     0.0 -0.0011695670745149895 -0.0009182098765432678 -0.0010295758928574872 -0.0009435626102302086 -0.0010549286265453262 -0.0008035714285732354 -0.0019731385031036552 0.03488536155197153]
                     =#
            theta = [0 1070017/29030400 32377/907200 12881/358400 4063/113400 41705/1161216 401/11200 149527/4147200 989/28350;
                     0 2233547/14515200 22823/113400 35451/179200 2822/14175 115075/580608 279/1400 408317/2073600 2944/14175;
                     0 -2302297/14515200 -21247/453600 1719/179200 61/28350 3775/580608 9/5600 24353/2073600 -464/14175;
                     0 2797679/14515200 15011/113400 39967/179200 4094/14175 159175/580608 403/1400 542969/2073600 5248/14175;
                     0 -31457/181440 -2903/22680 -351/2240 -227/2835 -125/36288 -9/280 343/25920 -454/2835;
                     0 1573169/14515200 9341/113400 17217/179200 1154/14175 85465/580608 333/1400 368039/2073600 5248/14175;
                     0 -645607/14515200 -15577/453600 -7031/179200 -989/28350 -24575/580608 79/5600 261023/2073600 -464/14175;
                     0 156437/14515200 953/113400 243/25600 122/14175 5725/580608 9/1400 111587/2073600 2944/14175;
                     0 -33953/29030400 -119/129600 -369/358400 -107/113400 -175/165888 -9/11200 -8183/4147200 989/28350]
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
        elseif alg.M == 10
            theta = [0.0 0.028018959644393687 0.027340374645930202 0.027451045048698663 0.0274153171930962 0.02743427683751065 0.027418831168816382 0.027437790813053198 0.02740206295788994 0.027512733360708808 0.02683414836155862;
                     0.0 0.13699028395729754 0.17247367858478835 0.17057771915585285 0.17108139597029037 0.17083711202639051 0.17102597402617903 0.17080197227005556 0.17121393832712783 0.16996083603953593 0.17753594143869122;
                     0.0 -0.18583978613015117 -0.08617167708834472 -0.04460141030846643 -0.048462401795699606 -0.04691594453948866 -0.048009740260582134 -0.046778097819711206 -0.04896713164639266 -0.04246829342450553 -0.08104357062261158;
                     0.0 0.30192072009780363 0.22804745871412313 0.3094549512987534 0.35692031425357507 0.34993098144275514 0.35402597402539016 0.34980383697848083 0.3569305756250287 0.33648092529256246 0.45494628807136905;
                     0.0 -0.3806483247655124 -0.3026606541606571 -0.3400126826297907 -0.2703953823952361 -0.21667333679079093 -0.2287597402537358 -0.2184080650258693 -0.23442039440851659 -0.19077242288039997 -0.43515512242447585;
                     0.0 0.35715424082090674 0.29001218534554596 0.31687012987021035 0.29602437069246434 0.3568823152170353 0.41774025973563766 0.3968945005565274 0.4237524451623358 0.35661038970238224 0.7137646304354348;
                     0.0 -0.24438269976550997 -0.2007347282347136 -0.21674705762993796 -0.20639538239549893 -0.2184817858628776 -0.16475974026434415 -0.0951424400191172 -0.13249446850386448 -0.05450679792556912 -0.43515512242447585;
                     0.0 0.11846536295494622 0.09801571268237676 0.10514245129872465 0.10092031425360393 0.10501530683775329 0.09802597402490676 0.14549133697096295 0.22689882957820373 0.15302556815186108 0.45494628810411086;
                     0.0 -0.038575277201579265 -0.03207643899310476 -0.03426547280844172 -0.03303383036714891 -0.03412762608719788 -0.032581168831256946 -0.03644216031966607 0.005128106447045866 0.10479621548802243 -0.08104357064439682;
                     0.0 0.007575105385869255 0.006322003099781336 0.006733969155847452 0.006509967398880434 0.0066988293985943415 0.006454545454857907 0.006958222269304315 0.005062262842329801 0.04054565746628214 0.17753594143141527;
                     0.0 -0.0006785849984634729 -0.0005679145956923939 -0.000603642451298736 -0.0005846828069051568 -0.0006001284755665637 -0.000581168831175205 -0.0006168966868074222 -0.0005062262839601317 -0.0011848112826555735 0.026834148362013366]
        else
            error("MPDeC requires 2 ≤ K ≤ 9.")
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
        elseif alg.M == 6
            nodes = [
                0.0,
                0.08488805186071652,
                0.2655756032646429,
                0.5,
                0.7344243967353571,
                0.9151119481392835,
                1.0
            ]
            theta = [0.0 0.03284626432829264 0.018002223201815104 0.027529761904763195 0.02170954269630787 0.02464779425215191 0.023809523809520172;
                     0.0 0.059322894027551365 0.15770113064168867 0.1277882555598353 0.14409488824697547 0.13619592709181916 0.1384130236807124;
                     0.0 -0.01076859445118926 0.10235481204686203 0.23748565272164512 0.20629511050420746 0.2193616205757678 0.21587269060495373;
                     0.0 0.0055975917805697745 -0.018478259273458753 0.12190476190476146 0.2622877830829893 0.23821193202898883 0.24380952380948884;
                     0.0 -0.0034889299708074674 0.009577580100741362 -0.02161296211671493 0.1135178785580706 0.22664128505616077 0.2158726906049253;
                     0.0 0.002217096588914541 -0.005681864566224326 0.010624768120945982 -0.019288106960911655 0.07909012965322404 0.13841302368079056;
                     0.0 -0.000838270442615105 0.0020999811132187685 -0.0037202380952379155 0.005807300607708399 -0.00903674051876635 0.02380952380952195]
        elseif alg.M == 7
            nodes = [
                0.0,
                0.06412992574519666,
                0.20414990928342885,
                0.3953503910487606,
                0.6046496089512394,
                0.7958500907165711,
                0.9358700742548034,
                1.0
            ]
            theta = [0.0 0.024737514438875514 0.013258719822130019 0.021034356725210923 0.01577972028613317 0.019036721343624663 0.017385669593011244 0.01785714285716722;
                     0.0 0.044892662602755 0.12064973282409763 0.09628017831208524 0.11096097521429726 0.10224716355590147 0.10657833455800869 0.10535211357202456;
                     0.0 -0.008140767742389747 0.07999763578665159 0.1890416498443277 0.16114433643933257 0.17539401516496778 0.1687159595472414 0.17056134624175545;
                     0.0 0.004254082548545937 -0.014480830962625313 0.10124595564769123 0.22436671754081416 0.19859744812260516 0.2089336024046844 0.20622939732966472;
                     0.0 -0.002704205075015816 0.0076319492068338685 -0.018137320211455643 0.10498344168165086 0.22071022829197284 0.20197531478076014 0.20622939732958345;
                     0.0 0.0018453866944819865 -0.004832668923134664 0.009417009802429988 -0.01848030360257269 0.09056371045502942 0.1787021139839453 0.17056134624169772;
                     0.0 -0.0012262209861064882 0.00310495001592085 -0.005608861642537821 0.00907193525967287 -0.015297619252294226 0.060459450968835426 0.10535211357171193;
                     0.0 0.0004714732640502682 -0.001179578486445107 0.002077422571009513 -0.003177213868071016 0.0045984230350075705 -0.006880371581776679 0.017857142857105046]
        elseif alg.M == 8
            nodes = [
                0.0,
                0.05012100229426997,
                0.16140686024463113,
                0.3184412680869109,
                0.5,
                0.6815587319130891,
                0.8385931397553689,
                0.94987899770573,
                1.0
            ]
            theta = [0.0 0.019293838201043245 0.01018408040822739 0.01656936984357027 0.011990017361096061 0.01514127656190567 0.013175811859724718 0.01417408466352299 0.013888888888914153;
                     0.0 0.03512552097762184 0.09508650844936217 0.07509351709620726 0.08786983372308654 0.07945805621039881 0.08459518591985216 0.0820139081133675 0.08274768078126726;
                     0.0 -0.006364102418704803 0.0639319956284379 0.15288206102488378 0.12868222230659399 0.14236568211174472 0.13451403216049584 0.13834341694882824 0.13726935625072656;
                     0.0 0.0033337771969983894 -0.011585731333842608 0.0840854786889147 0.18975808031592045 0.16521365191611892 0.17719508127352412 0.1717199563247931 0.17321425548699665;
                     0.0 -0.002136847017608247 0.006149936900551798 -0.015130673172334241 0.09287981859410088 0.20089031036043536 0.17960970028764223 0.18789648420624872 0.1857596371874024;
                     0.0 0.0014942991627282308 -0.003980825787182701 0.008000603570397224 -0.0165438248294123 0.08912877679762232 0.18479998682039422 0.16988047828840536 0.17321425548630032;
                     0.0 -0.001074060699381661 0.0027553240894192185 -0.0050963258617406915 0.008587133943507297 -0.015612704774753183 0.0733373606215082 0.14363345866922828 0.13726935624890757;
                     0.0 0.0007337726665392911 -0.0018475051393835457 0.0032896245700402282 -0.005122152942657721 0.007654163684208015 -0.012338827668713748 0.04762215980304063 0.08274768078103989;
                     0.0 -0.00028519577496630263 0.0007130770290413452 -0.0012523876730271555 0.0018988715277794554 -0.002680480954676767 0.0037048084806841075 -0.005404949312023177 0.013888888889184159]
        elseif alg.M == 9
            nodes = [
                0.0,
                0.04023304591677057,
                0.13061306744724743,
                0.26103752509477773,
                0.4173605211668065,
                0.5826394788331936,
                0.7389624749052223,
                0.8693869325527526,
                0.9597669540832294,
                1.0
            ]
            theta = [0.0 0.015465148886122208 0.008074564828863144 0.013376018525479871 0.009424872116881033 0.012319010010855891 0.010311035538052238 0.01156730916181914 0.010928591594165482 0.011111111110039928;
                     0.0 0.02821797600073163 0.07677451467523239 0.060184542475466785 0.07119971523047752 0.06348355183104104 0.06872200531256567 0.06548282490166457 0.06711913714752882 0.0666529954141879;
                     0.0 -0.0051090759506765785 0.05211803626519543 0.12565307727248143 0.10482643149413295 0.11734393918791586 0.10937232662209873 0.11414511256730009 0.11177491077136281 0.1124446710354885;
                     0.0 0.0026797026134601862 -0.009449705457448665 0.0703555323098648 0.1607094677113423 0.1383496630545551 0.15043195838711654 0.14368298780891564 0.14692275221750606 0.1460213418404237;
                     0.0 -0.0017246233123523063 0.005034230848115648 -0.012689136533024836 0.08096672386450621 0.17827052516622643 0.15700518309495237 0.1670603392740304 0.16255069057842775 0.1637698805952823;
                     0.0 0.0012191900098376621 -0.003290458683960145 0.006764697496483674 -0.014500644574205523 0.08280315672743654 0.17645901712467094 0.15873564974754117 0.16549450390448328 0.1637698805971013;
                     0.0 -0.0009014103789669575 0.002338354035534229 -0.004410616548205404 0.007671678785357017 -0.014688125871543889 0.07566580953016455 0.15547104729739658 0.1433416392246727 0.1460213418395142;
                     0.0 0.0006697602581937584 -0.00170044153617834 0.0030723444103197828 -0.0048992681562261 0.007618239537521276 -0.013208406240892145 0.06032663476707967 0.11755374698441301 0.1124446710346092;
                     0.0 -0.000466141727447515 0.0011701705230945658 -0.002069009887745693 0.0031694435943515065 -0.004546719804944366 0.0064684529496542575 -0.010121519251002686 0.03843501942257066 0.0666529954255306;
                     0.0 0.0001825195178684848 -0.0004561980512006905 0.0008000755736511378 -0.0012078988997608064 0.00168623899423892 -0.002264907414485151 0.00303654628237382 -0.004354037773737218 0.011111111112313665]
        else
            error("MPDeC requires 2 ≤ K ≤ 9.")
        end
    end
    return nodes, theta
end

struct MPDeCConstantCache{KType, NType, T, T2} <: OrdinaryDiffEqConstantCache
    K::KType
    M::KType
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

#=
function _build_mpdec_matrix_and_rhs_old!(Mmat, rhs, P, dt_th, σ, d = nothing)
    N = length(rhs)
    @fastmath @inbounds @simd for i in 1:N
        # Add nonconservative destruction terms to diagonal (PDSFunctions only!)
        if !isnothing(d)
            if dt_th >= 0
                Mmat[i, i] += dt_th * d[i] / σ[i]
                rhs[i] += dt_th * P[i, i]
            else
                Mmat[i, i] -= dt_th * P[i, i] / σ[i]
                rhs[i] -= dt_th * d[i]
            end
        end
        @fastmath @inbounds @simd for j in 1:N
            if j != i
                if dt_th >= 0
                    Mmat[i, j] -= dt_th * P[i, j] / σ[j]
                    Mmat[i, i] += dt_th * P[j, i] / σ[i] # P[j, i] = D[i, j]
                else
                    Mmat[i, j] += dt_th * P[j, i] / σ[j] # P[j, i] = D[i, j]
                    Mmat[i, i] -= dt_th * P[i, j] / σ[i]
                end
            end
        end
    end
end
=#
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

    #TODO: It is crucial that evaluation of the production function 
    # does not alter the sparsity pattern of p_prototype. This should be checked earlier. 

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

    #TODO: Remove this check
    # Check only valid for autonomous conserverative PDS
    #u_check = MPDeC_check(K, M, uprev, theta, f, p, t, dt)
    #@assert u ≈ u_check

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
    if issparse(P2) && alg.K > 2
        # Negative weights of MPDeC(K) , K >=3 require
        # a symmetric sparsity pattern
        P2 = P2 + P2'
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

# TODO: Remove the code below after testing is complete
########################################################################################################
########################################################################################################
#### The functions below are provided with the original paper and used for validation.              ####
#### They will be removed in the future.                                                            ####
########################################################################################################
########################################################################################################
function patanker_type_dec(prod_p, dest_p, delta_t, m_substep::Int, M_sub, theta, u_p, dim)
    #This function builds and solves the system u^{(k+1)}=Mu^{(k)} for a subtimestep m

    mass = Matrix{Float64}(I, dim, dim)
    #println(mass)
    for i in 1:dim
        for r in 1:(M_sub + 1)
            if theta[r, m_substep] > 0
                for j in 1:dim
                    mass[i, j] = mass[i, j] -
                                 delta_t * theta[r, m_substep] *
                                 (prod_p[i, j, r] / u_p[j, m_substep])
                    mass[i, i] = mass[i, i] +
                                 delta_t * theta[r, m_substep] *
                                 (dest_p[i, j, r] / u_p[i, m_substep])
                end
            elseif theta[r, m_substep] < 0
                for j in 1:dim
                    mass[i, i] = mass[i, i] -
                                 delta_t * theta[r, m_substep] *
                                 (prod_p[i, j, r] / u_p[i, m_substep])
                    mass[i, j] = mass[i, j] +
                                 delta_t * theta[r, m_substep] *
                                 (dest_p[i, j, r] / u_p[j, m_substep])
                end
            end
        end
    end
    return mass \ u_p[:, 1]
end

function MPDeC_check(K, M, uprev, theta, f, p, t, dt)
    M_sub = M
    K_corr = K
    dim = length(uprev)

    #Variables of the DeC procedure: one for each subtimesteps and one for previous correction up, one for current correction ua
    u_p = zeros(dim, M_sub + 1)
    u_a = zeros(dim, M_sub + 1)

    #Matrices of production and destructions applied to up at different subtimesteps
    prod_p = zeros(dim, dim, M_sub + 1)
    dest_p = zeros(dim, dim, M_sub + 1)

    #coefficients for time integration
    Theta = theta

    # Subtimesteps loop to update variables at iteration 0
    for m in 1:(M_sub + 1)
        u_a[:, m] = uprev
        u_p[:, m] = uprev
    end

    #Loop for iterations K
    for k in 2:(K_corr + 1)
        # Updating previous iteration variabls
        u_p = copy(u_a)
        #Loop on subtimesteps to compute the production destruction terms
        for r in 1:(M_sub + 1)
            #prod_p[:,:,r], dest_p[:,:,r]=prod_dest(u_p[:,r])
            prod_p[:, :, r] = f.p(u_p[:, r], p, t)
            dest_p[:, :, r] = prod_p[:, :, r]'
        end
        # Loop on subtimesteps to compute the new variables
        for m in 2:(M_sub + 1)
            u_a[:, m] = patanker_type_dec(prod_p, dest_p, dt, m, M_sub, Theta, u_p,
                                          dim)
        end
    end
    u1 = u_a[:, M_sub + 1]
    return u1
end
