# linear model problem

P_linmod(u, p, t) = @SMatrix [0.0 u[2]; 5.0*u[1] 0.0]
function f_linmod_analytic(u0, p, t)
    u₁⁰, u₂⁰ = u0
    a = 5.0
    b = 1.0
    c = a + b
    return ((u₁⁰ + u₂⁰) * [b; a] + exp(-c * t) * (a * u₁⁰ - b * u₂⁰) * [1; -1]) / c
end
u0_linmod = @SVector [0.9, 0.1]
prob_pds_linmod = ConservativePDSProblem(P_linmod, u0_linmod, (0.0, 2.0), analytic = f_linmod_analytic)

# nonlinear model problem
function P_nonlinmod(u, p, t)
    @SMatrix [0.0 0.0 0.0; u[2] * u[1]/(u[1] + 1.0) 0.0 0.0; 0.0 0.3*u[2] 0.0]
end
u0_nonlinmod = @SVector [9.98; 0.01; 0.01]
prob_pds_nonlinmod = ConservativePDSProblem(P_nonlinmod, u0_nonlinmod, (0.0, 30.0))

# robertson problem 
function P_robertson(u, p, t)
    @SMatrix [0.0 1e4*u[2]*u[3] 0.0; 4e-2*u[1] 0.0 0.0; 0.0 3e7*u[2]^2 0.0]
end
u0_robertson = @SVector [1.0, 0.0, 0.0]
prob_pds_robertson = ConservativePDSProblem(P_robertson, u0_robertson, (0.0, 1.0e11))

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
u0_brusselator = @SVector [10.0, 10.0, 0.0, 0.0, 0.1, 0.1]
prob_pds_brusselator = ConservativePDSProblem(P_brusselator, u0_brusselator, (0.0, 10.0))

# SIR problem 
P_sir(u, p, t) = @SMatrix [0.0 0.0 0.0; 2*u[1]*u[2] 0.0 0.0; 0.0 u[2] 0.0]
u0_sir = @SVector [0.99, 0.005, 0.005]
prob_pds_sir = ConservativePDSProblem(P_sir, u0_sir, (0.0, 20.0))

# bertolazzi problem
function P_bertolazzi(u, p, t)
    f1 = 5 * u[2] * u[3] / (1e-2 + (u[2] * u[3])^2) +
         u[2] * u[3] / (1e-16 + u[2] * u[3] * (1e-8 + u[2] * u[3]))
    f2 = 10 * u[1] * u[3]^2
    f3 = 0.1 * (u[3] - u[2] - 2.5)^2 * u[1] * u[2]

    return @SMatrix [0.0 f1 f1; f2 0.0 f2; f3 f3 0.0]
end
u0_bertolazzi = @SVector [0.0, 1.0, 2.0]
prob_pds_bertolazzi = ConservativePDSProblem(P_bertolazzi, u0_bertolazzi, (0.0, 1.0))

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
u0_npzd = @SVector [8.0, 2.0, 1.0, 4.0]
prob_pds_npzd = ConservativePDSProblem(P_npzd, u0_npzd, (0.0, 10.0))

#strat reac problem
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
u0_stratreac = @SVector [9.906e1, 6.624e8, 5.326e11, 1.697e16, 4e6, 1.093e9]
prob_pds_stratreac = PDSProblem(P_stratreac, d_stratreac, u0_stratreac, (4.32e4, 3.024e5))
