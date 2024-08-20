# linear model problem

P_linmod(u, p, t) = @SMatrix [0.0 u[2]; 5.0*u[1] 0.0]
f_linmod(u, p, t) = @SVector [u[2] - 5.0 * u[1]; 5.0 * u[1] - u[2]]
function f_linmod_analytic(u0, p, t)
    u₁⁰, u₂⁰ = u0
    a = 5.0
    b = 1.0
    c = a + b
    return ((u₁⁰ + u₂⁰) * [b; a] + exp(-c * t) * (a * u₁⁰ - b * u₂⁰) * [1; -1]) / c
end
u0_linmod = @SVector [0.9, 0.1]
"""
    prob_pds_linmod

Positive and conservative autonomous linear PDS
```math
\\begin{aligned}
u_1' &= u_2 - 5u_1,\\\\
u_2' &= 5u_1 - u_2,
\\end{aligned}
```
with initial value ``\\mathbf{u}_0 = (0.9, 0.1)^T`` and time domain ``(0.0, 2.0)``.
There is one independent linear invariant, e.g. ``u_1+u_2 = 1``.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
"""
prob_pds_linmod = ConservativePDSProblem(P_linmod, u0_linmod, (0.0, 2.0),
                                         analytic = f_linmod_analytic, std_rhs = f_linmod)

function P_linmod!(P, u, p, t)
    P .= P_linmod(u, p, t)
    return nothing
end
function f_linmod!(du, u, p, t)
    du .= f_linmod(u, p, t)
    return nothing
end

"""
    prob_pds_linmod_inplace

Same as [`prob_pds_linmod`](@ref) but with in-place computation.
"""
prob_pds_linmod_inplace = ConservativePDSProblem(P_linmod!, Array(u0_linmod),
                                                 (0.0, 2.0),
                                                 analytic = f_linmod_analytic,
                                                 std_rhs = f_linmod!)

# nonlinear model problem
function P_nonlinmod(u, p, t)
    @SMatrix [0.0 0.0 0.0; u[2] * u[1]/(u[1] + 1.0) 0.0 0.0; 0.0 0.3*u[2] 0.0]
end
function f_nonlinmod(u, p, t)
    @SVector [-u[2] * u[1] / (u[1] + 1);
              u[2] * u[1] / (u[1] + 1) - 0.3 * u[2];
              0.3 * u[2]]
end
u0_nonlinmod = @SVector [9.98; 0.01; 0.01]
"""
    prob_pds_nonlinmod

Positive and conservative autonomous nonlinear PDS
```math
\\begin{aligned}
u_1' &= -\\frac{u_1u_2}{u_1 + 1.0},\\\\
u_2' &= \\frac{u_1u_2}{u_1 + 1.0} - 0.3u_2,\\\\
u_3' &= 0.3 u_2,
\\end{aligned}
```
with initial value ``\\mathbf{u}_0 = (9.98, 0.01, 0.01)^T`` and time domain ``(0.0, 30.0)``.
There is one independent linear invariant, e.g. ``u_1+u_2+u_3 = 10.0``.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "A high-order conservative Patankar-type discretisation for stiff systems of
  production-destruction equations."
  Applied Numerical Mathematics 47.1 (2003): 1-30.
  [DOI: 10.1016/S0168-9274(03)00101-6](https://doi.org/10.1016/S0168-9274(03)00101-6)
"""
prob_pds_nonlinmod = ConservativePDSProblem(P_nonlinmod, u0_nonlinmod, (0.0, 30.0),
                                            std_rhs = f_nonlinmod)

# robertson problem
function P_robertson(u, p, t)
    @SMatrix [0.0 1e4*u[2]*u[3] 0.0; 4e-2*u[1] 0.0 0.0; 0.0 3e7*u[2]^2 0.0]
end
function f_robertson(u, p, t)
    return @SVector [1e4 * u[2] * u[3] - 4e-2 * u[1];
                     4e-2 * u[1] - 1e4 * u[2] * u[3] - 3e7 * u[2]^2;
                     3e7 * u[2]^2]
end;
u0_robertson = @SVector [1.0, 0.0, 0.0]
"""
    prob_pds_robertson

Positive and conservative autonomous nonlinear PDS
```math
\\begin{aligned}
u_1' &= -0.04u_1+10^4 u_2u_3,\\\\
u_2' &=  0.04u_1-10^4 u_2u_3-3⋅10^7 u_2^2,\\\\
u_3' &= 3⋅10^7 u_2^2,
\\end{aligned}
```
with initial value ``\\mathbf{u}_0 = (1.0, 0.0, 0.0)^T`` and time domain ``(0.0, 10^{11})``.
There is one independent linear invariant, e.g. ``u_1+u_2+u_3 = 1.0``.

## References

- Ernst Hairer, Gerd Wanner.
  "Solving Ordinary Differential Equations II - Stiff and Differential-Algebraic Problems."
  2nd Edition, Springer (2002): Section IV.1.
"""
prob_pds_robertson = ConservativePDSProblem(P_robertson, u0_robertson, (0.0, 1.0e11),
                                            std_rhs = f_robertson)

# brusselator problem
function P_brusselator(u, p, t)
    u2u5 = u[2] * u[5]

    @SMatrix [0.0 0.0 0.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0 0.0 0.0;
              0.0 u2u5 0.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0 u[5] 0.0;
              u[1] 0.0 0.0 0.0 0.0 u[5]^2*u[6];
              0.0 0.0 0.0 0.0 u2u5 0.0]
end;
function f_brusselator(u, p, t)
    @SVector [-u[1];
              -u[2] * u[5];
              u[2] * u[5];
              u[5];
              u[1] - u[2] * u[5] + u[5]^2 * u[6] - u[5];
              u[2] * u[5] - u[5]^2 * u[6]]
end;
u0_brusselator = @SVector [10.0, 10.0, 0.0, 0.0, 0.1, 0.1]
"""
    prob_pds_brusselator

Positive and conservative autonomous nonlinear PDS
```math
\\begin{aligned}
u_1' &= -u_1,\\\\
u_2' &= -u_2u_5,\\\\
u_3' &= u_2u_5,\\\\
u_4' &= u_5,\\\\
u_5' &= u_1 - u_2u_5 + u_5^2u_6 - u_5,\\\\
u_6' &= u_2u_5 - u_5^2u_6,
\\end{aligned}
```
with initial value ``\\mathbf{u}_0 = (10.0, 10.0, 0.0, 0.0, 0.1, 0.1)^T`` and time domain ``(0.0, 20.0)``.
There are two independent linear invariants, e.g. ``u_1+u_4+u_5+u_6 = 10.2`` and ``u_2+u_3 = 10.0``.

## References

- Luca Bonaventura,  and Alessandro Della Rocca.
  "Unconditionally Strong Stability Preserving Extensions of the TR-BDF2 Method."
  Journal of Scientific Computing 70 (2017): 859 - 895.
  [DOI: 10.1007/s10915-016-0267-9](https://doi.org/10.1007/s10915-016-0267-9)
"""
prob_pds_brusselator = ConservativePDSProblem(P_brusselator, u0_brusselator, (0.0, 10.0),
                                              std_rhs = f_brusselator)

# SIR problem
P_sir(u, p, t) = @SMatrix [0.0 0.0 0.0; 2*u[1]*u[2] 0.0 0.0; 0.0 u[2] 0.0]
f_sir(u, p, t) = @SVector [-2 * u[1] * u[2];
                           2 * u[1] * u[2] - u[2];
                           u[2]];
u0_sir = @SVector [0.99, 0.005, 0.005]
"""
    prob_pds_sir

Positive and conservative autonomous nonlinear PDS
```math
\\begin{aligned}
u_1' &= -2u_1u_2,\\\\
u_2' &= 2u_1u_2 - u_2,\\\\
u_3' &= u_2,
\\end{aligned}
```
with initial value ``\\mathbf{u}_0 = (0.99, 0.005, 0.005)^T`` and time domain ``(0.0, 20.0)``.
There is one independent linear invariant, e.g. ``u_1+u_2+u_3 = 1.0``.

## References

- Ronald E. Mickens, and Talitha M. Washington.
  "NSFD discretizations of interacting population models satisfying conservation laws."
  Computers and Mathematics with Applications 66 (2013): 2307-2316.
  [DOI: 10.1016/j.camwa.2013.06.011](https://doi.org/10.1016/j.camwa.2013.06.011)
"""
prob_pds_sir = ConservativePDSProblem(P_sir, u0_sir, (0.0, 20.0), std_rhs = f_sir)

# bertolazzi problem
function P_bertolazzi(u, p, t)
    f1 = 5 * u[2] * u[3] / (1e-2 + (u[2] * u[3])^2) +
         u[2] * u[3] / (1e-16 + u[2] * u[3] * (1e-8 + u[2] * u[3]))
    f2 = 10 * u[1] * u[3]^2
    f3 = 0.1 * (u[3] - u[2] - 2.5)^2 * u[1] * u[2]

    return @SMatrix [0.0 f1 f1; f2 0.0 f2; f3 f3 0.0]
end
function f_bertolazzi(u, p, t)
    f1 = 5 * u[2] * u[3] / (1e-2 + (u[2] * u[3])^2) +
         u[2] * u[3] / (1e-16 + u[2] * u[3] * (1e-8 + u[2] * u[3]))
    f2 = 10 * u[1] * u[3]^2
    f3 = 0.1 * (u[3] - u[2] - 2.5)^2 * u[1] * u[2]

    return @SVector [2 * f1 - f2 - f3; -f1 + 2 * f2 - f3; -f1 - f2 + 2 * f3]
end
u0_bertolazzi = @SVector [0.0, 1.0, 2.0]
"""
    prob_pds_bertolazzi

Positive and conservative autonomous nonlinear PDS
```math
\\begin{aligned}
\\mathbf{u}'=\\begin{pmatrix}2 &-1 &-1\\\\-1 &2 &-1\\\\-1& -1& 2\\end{pmatrix}\\begin{pmatrix}5u_2u_3/(10^{-2} + (u_2u_3)^2) + u_2u_3/(10^{-16} + u_2u_3(10^{-8} + u_2u_3))\\\\
10u_1u_3^2\\\\
0.1(u_3 - u_2 - 2.5)^2u_1u_2\\end{pmatrix}
\\end{aligned}
```
with initial value ``\\mathbf{u}_0 = (0.0, 1.0, 2.0)^T`` and time domain ``(0.0, 1.0)``.
There is one independent linear invariant, e.g. ``u_1+u_2+u_3 = 3.0``.

## References

- Enrico Bertolazzi.
  "Positive and conservative schemes for mass action kinetics."
  Computers and Mathematics with Applications 32 (1996): 29-43.
  [DOI: 10.1016/0898-1221(96)00142-3](https://doi.org/10.1016/0898-1221(96)00142-3)
"""
prob_pds_bertolazzi = ConservativePDSProblem(P_bertolazzi, u0_bertolazzi, (0.0, 1.0),
                                             std_rhs = f_bertolazzi)

# npzd problem
function P_npzd(u, p, t)
    dnp = u[1] / (0.01 + u[1]) * u[2]
    dpz = 0.5 * (1.0 - exp(-1.21 * u[2]^2)) * u[3]
    dpn = 0.01 * u[2]
    dzn = 0.01 * u[3]
    ddn = 0.003 * u[4]
    dpd = 0.05 * u[2]
    dzd = 0.02 * u[3]

    return @SMatrix [0.0 dpn dzn ddn; dnp 0.0 0.0 0.0; 0.0 dpz 0.0 0.0; 0.0 dpd dzd 0.0]
end
function f_npzd(u, p, t)
    dnp = u[1] / (0.01 + u[1]) * u[2]
    dpz = 0.5 * (1.0 - exp(-1.21 * u[2]^2)) * u[3]
    dpn = 0.01 * u[2]
    dzn = 0.01 * u[3]
    ddn = 0.003 * u[4]
    dpd = 0.05 * u[2]
    dzd = 0.02 * u[3]

    return @SVector [dpn + dzn + ddn - dnp;
                     dnp - dpn - dpz - dpd;
                     dpz - dzn - dzd;
                     dpd + dzd - ddn]
end
u0_npzd = @SVector [8.0, 2.0, 1.0, 4.0]
"""
    prob_pds_npzd

Positive and conservative autonomous nonlinear PDS
```math
\\begin{aligned}
u_1' &= 0.01u_2 + 0.01u_3 + 0.003u_4 - \\frac{u_1u_2}{0.01 + u_1},\\\\
u_2' &= \\frac{u_1u_2}{0.01 + u_1}- 0.01u_2 - 0.5( 1 - e^{-1.21u_2^2})u_3 - 0.05u_2,\\\\
u_3' &= 0.5(1 - e^{-1.21u_2^2})u_3 - 0.01u_3 - 0.02u_3,\\\\
u_4' &= 0.05u_2 + 0.02u_3 - 0.003u_4
\\end{aligned}
```
with initial value ``\\mathbf{u}_0 = (8.0, 2.0, 1.0, 4.0)^T`` and time domain ``(0.0, 10.0)``.
There is one independent linear invariant, e.g. ``u_1+u_2+u_3+u_4 = 15.0``.

## References

- Hans Burchard, Eric Deleersnijder, and Andreas Meister.
  "Application of modified Patankar schemes to stiff biogeochemical models for the water column."
  Ocean Dynamics 55 (2005): 326-337.
  [DOI: 10.1007/s10236-005-0001-x](https://doi.org/10.1007/s10236-005-0001-x)
"""
prob_pds_npzd = ConservativePDSProblem(P_npzd, u0_npzd, (0.0, 10.0), std_rhs = f_npzd)

# stratospheric reaction problem
function P_stratreac(u, p, t)
    O1D, O, O3, O2, NO, NO2 = u

    Tr = 4.5
    Ts = 19.5
    T = mod(t / 3600, 24)
    if (Tr <= T) && (T <= Ts)
        Tfrac = (2 * T - Tr - Ts) / (Ts - Tr)
        sigma = 0.5 + 0.5 * cos(pi * abs(Tfrac) * Tfrac)
    else
        sigma = 0.0
    end

    M = 8.120e16

    k1 = 2.643e-10 * sigma^3
    k2 = 8.018e-17
    k3 = 6.120e-4 * sigma
    k4 = 1.567e-15
    k5 = 1.070e-3 * sigma^2
    k6 = 7.110e-11
    k7 = 1.200e-10
    k8 = 6.062e-15
    k9 = 1.069e-11
    k10 = 1.289e-2 * sigma
    k11 = 1.0e-8

    r1 = k1 * O2
    r2 = k2 * O * O2
    r3 = k3 * O3
    r4 = k4 * O3 * O
    r5 = k5 * O3
    r6 = k6 * M * O1D
    r7 = k7 * O1D * O3
    r8 = k8 * O3 * NO
    r9 = k9 * NO2 * O
    r10 = k10 * NO2
    r11 = k11 * NO * O

    return @SMatrix [0.0 0.0 r5 0.0 0.0 0.0;
                     r6 r1+r10 r3 r1 0.0 0.0;
                     0.0 r2 0.0 0.0 0.0 0.0;
                     r7 r4+r9 r4+r7+r8 r3+r5 0.0 0.0;
                     0.0 0.0 0.0 0.0 0.0 r9+r10;
                     0.0 0.0 0.0 0.0 r8+r11 0.0]
end
function d_stratreac(u, p, t)
    O1D, O, O3, O2, NO, NO2 = u

    k2 = 8.018e-17
    k11 = 1.0e-8

    r2 = k2 * O * O2
    r11 = k11 * NO * O

    return @SVector [0.0, r11, 0.0, r2, 0.0, 0.0]
end
function f_stratreac(u, p, t)
    O1D, O, O3, O2, NO, NO2 = u

    Tr = 4.5
    Ts = 19.5
    T = mod(t / 3600, 24)
    if (Tr <= T) && (T <= Ts)
        Tfrac = (2 * T - Tr - Ts) / (Ts - Tr)
        sigma = 0.5 + 0.5 * cos(pi * abs(Tfrac) * Tfrac)
    else
        sigma = 0.0
    end

    M = 8.120e16

    k1 = 2.643e-10 * sigma^3
    k2 = 8.018e-17
    k3 = 6.120e-4 * sigma
    k4 = 1.567e-15
    k5 = 1.070e-3 * sigma^2
    k6 = 7.110e-11
    k7 = 1.200e-10
    k8 = 6.062e-15
    k9 = 1.069e-11
    k10 = 1.289e-2 * sigma
    k11 = 1.0e-8

    r1 = k1 * O2
    r2 = k2 * O * O2
    r3 = k3 * O3
    r4 = k4 * O3 * O
    r5 = k5 * O3
    r6 = k6 * M * O1D
    r7 = k7 * O1D * O3
    r8 = k8 * O3 * NO
    r9 = k9 * NO2 * O
    r10 = k10 * NO2
    r11 = k11 * NO * O

    return @SVector [r5 - r6 - r7;
                     2 * r1 - r2 + r3 - r4 + r6 - r9 + r10 - r11;
                     r2 - r3 - r4 - r5 - r7 - r8;
                     -r1 - r2 + r3 + 2 * r4 + r5 + 2 * r7 + r8 + r9;
                     -r8 + r9 + r10 - r11;
                     r8 - r9 - r10 + r11]
end
u0_stratreac = @SVector [9.906e1, 6.624e8, 5.326e11, 1.697e16, 4e6, 1.093e9]
"""
    prob_pds_stratreac

Positive and nonconservative autonomous nonlinear PDS
```math
\\begin{aligned}
u_1' &= r_5 - r_6 -  r_7,\\\\
u_2' &= 2r_1 - r_2 + r_3 - r_4 + r_6 - r_9 + r_{10} - r_{11},\\\\
u_3' &= r_2 - r_3 - r_4 - r_5 - r_7 - r_8,\\\\
u_4' &= -r_1 -r_2 + r_3 + 2r_4+r_5+2r_7+r_8+r_9,\\\\
u_5' &= -r_8+r_9+r_{10}-r_{11},\\\\
u_6' &= r_8-r_9-r_{10}+r_{11},
\\end{aligned}
```
with reaction rates
```math
\\begin{aligned}
r_1 &=2.643⋅ 10^{-10}σ^3 u_4, & r_2 &=8.018⋅10^{-17}u_2 u_4 , & r_3 &=6.12⋅10^{-4}σ u_3,\\\\
r_4 &=1.567⋅10^{-15}u_3 u_2 , & r_5 &= 1.07⋅ 10^{-3}σ^2u_3,  & r_6 &= 7.11⋅10^{-11}⋅ 8.12⋅10^6 u_1,\\\\
r_7 &= 1.2⋅10^{-10}u_1 u_3, & r_8 &= 6.062⋅10^{-15}u_3 u_5, & r_9 &= 1.069⋅10^{-11}u_6 u_2,\\\\
r_{10} &= 1.289⋅10^{-2}σ u_6, & r_{11} &= 10^{-8}u_5 u_2,
\\end{aligned}
```
where
```math
\\begin{aligned}
T &= t/3600 \\mod 24,\\quad T_r=4.5,\\quad T_s = 19.5,\\\\
σ(T) &= \\begin{cases}1, & T_r≤ T≤ T_s,\\\\0, & \\text{otherwise}.\\end{cases}
\\end{aligned}
```

The initial value is ``\\mathbf{u}_0 = (9.906⋅10^1, 6.624⋅10^8, 5.326⋅10^{11}, 1.697⋅10^{16}, 4⋅10^6, 1.093⋅10^9)^T`` and the time domain ``(4.32⋅ 10^{4}, 3.024⋅10^5)``.
There are two independent linear invariants, e.g. ``u_1+u_2+3u_3+2u_4+u_5+2u_6=(1,1,3,2,1,2)\\cdot\\mathbf{u}_0`` and ``u_5+u_6 = 1.097⋅10^9``.

## References

- Stephan Nüsslein, Hendrik Ranocha, and David I. Ketcheson.
  "Positivity-preserving adaptive Runge-Kutta methods."
  Communications in Applied Mathematics and Computer Science 16 (2021): 155-179.
  [DOI: 10.2140/camcos.2021.16.155](https://doi.org/10.2140/camcos.2021.16.155)
"""
prob_pds_stratreac = PDSProblem(P_stratreac, d_stratreac, u0_stratreac, (4.32e4, 3.024e5),
                                std_rhs = f_stratreac)

# mapk problem
function f_minmapk(u, p, t)
    k1 = 100 / 3
    k2 = 1 / 3
    k3 = 50
    k4 = 0.5
    k5 = 10 / 3
    k6 = 0.1
    k7 = 0.7

    return @SVector [k6 * u[6] - k7 * u[1] - k1 * u[1] * u[2] + k2 * u[4];
                     k5 * u[3] - k1 * u[1] * u[2];
                     k2 * u[4] - k3 * u[1] * u[3] + k4 * u[5] - k5 * u[3];
                     k1 * u[1] * u[2] - k2 * u[4];
                     k3 * u[1] * u[3] - k4 * u[5];
                     k7 * u[1] - k6 * u[6]]
end
function P_minmapk(u, p, t)
    k1 = 100 / 3
    k2 = 1 / 3
    k3 = 50
    k4 = 0.5
    k5 = 10 / 3
    k6 = 0.1
    k7 = 0.7

    return @SMatrix [0 0 0 k2*u[4] 0 k6*u[6]
                     0 0 k5*u[3] 0 0 0;
                     0 0 k2*u[4] 0 k4*u[5] 0;
                     k1*u[1]*u[2] 0 0 0 0 0;
                     0 0 k3*u[1]*u[3] 0 0 0;
                     k7*u[1] 0 0 0 0 0]
end
function D_minmapk(u, p, t)
    k1 = 100 / 3

    return @SVector [0; k1 * u[1] * u[2]; 0; 0; 0; 0]
end

u0 = @SVector [0.1; 0.175; 0.15; 1.15; 0.81; 0.5]
tspan = (0.0, 200.0)

"""
    prob_pds_minmapk

Positive and nonconservative autonomous nonlinear PDS
```math
\\begin{aligned}
u_1' &= k_6u_6-k_7u_1-k_1u_1u_2 +k_2u_4,\\\\
u_2' &= k_5u_3-k_1u_2u_2,\\\\
u_3' &= k_2u_4-k_3u_1u_3+k_4u_5-k_5u_3,\\\\
u_4' &= k_1u_1u_2-k_2u_4,\\\\
u_5' &= k_3u_1u_3-k_4u_5,\\\\
u_6' &= k_7u_1-k_6u_6,
\\end{aligned}
```
with constants
```math
\\begin{aligned}
k_1 &=\\frac{100}{3}, & k_2 &=\\frac{1}{3}, & k_3 &=50,\\\\
k_4 &=0.5, & k_5 &=\\frac{10}{3} ,  & k_6 &= 0.1,\\\\
k_7 &= 0.1.
\\end{aligned}

The initial value is ``\\mathbf{u}_0 = (0.1, 0.175, 0.15, 1.15, 0.81, 0.5)^T`` and the time domain ``(4.32⋅ 10^{4}, 3.024⋅10^5)``.
There are two independent linear invariants, e.g. ``u_1+u_4+u_6=1.75`` and ``u_2+u_3+u_4+u_5 =2.285``.

## References

- Sergio Blanes, Arieh Iserles, and Shev Macnamara.
  "Positivity preserving methods for ordinary differential equations."
  ESAIM: Mathematical Modelling and Numerical Analysis 56 (2022): 1843–1870.
  [DOI: 10.1051/m2an/2022042](https://doi.org/10.1051/m2an/2022042)
- Otto Hadač, František Muzika, Vladislav Nevoral, Michal Přibyl, and Igor Schreiber
  "Minimal oscillating subnetwork in the Huang-Ferrell model of the MAPK cascade."
  PLoS ONE 12 (2017): e0178457.
  [DOI: 10.1371/journal.pone.0178457](https://doi.org/10.1371/journal.pone.0178457)
"""
prob_pds_minmapk = PDSProblem(P_minmapk, D_minmapk, u0, tspan; std_rhs = f_minmapk)
