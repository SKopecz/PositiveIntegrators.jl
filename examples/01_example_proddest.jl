# A production-destruction system (PDS) has the form
#   uᵢ'(t) = ∑ⱼ(pᵢⱼ(t) - dᵢⱼ(t))
# with pᵢⱼ(t), dᵢⱼ ≥ 0.
#
# Here, pᵢⱼ is a production term, i.e. a term with positive sign, in the ith equation with a corresponding
# destruction term, i.e. the same term with negative sign, in the jth equation. Positive terms of the ith equation
# without negative counterpart are gathered in pᵢᵢ. Similarly, dᵢᵢ contains the absolute values of the negative
# terms without positive counterparts.
#
# Example: u₁'(t) = 2 u₂(t) - u₁(t) + 3 u₁(t)
#          u₂'(t) = u₁(t) - 2 u₂(t) - 4 u₂(t)
# As the term u₁ appears with a positive sign in equation 2 and with negative sign in equation 1, we have p₂₁ = u₁ = d₁₂.
# Similarly, p₁₂ = 2 u₂ = d₂₁. Furthermore, we have p₁₁ = 3 u₁, d₁₁ = 0 and p₂₂ = 0, d₂₂ = 4 u₂.
# Note that this representation is not unique!
#
# The PDS is conservative if dᵢⱼ == pⱼᵢ and pᵢᵢ == dᵢᵢ == 0. In this case only the matrix (pᵢⱼ) needs to be stored,
# because (dᵢⱼ) is the transpose of (pᵢⱼ). Furthermore, all diagonal elements of (pᵢⱼ) are zero.
# If the PDS is not conservative, we can store the additional production terms pᵢᵢ on the diagonal of (pᵢⱼ) and need to
# store an additional vector for dᵢᵢ.
#
# Modified Patankar-Runge-Kutta (MPRK) schemes are based on the production-destruction representation of an ODE.
#
# In OrdinaryDiffEq an ODE u' = f(t,u) with t in tspan and parameters p is represented as an ODEProblem(f,u,tspan,p). To
# represent f as a PDS the new problem type ProdDestODEProblem(P,D,u,tspan,p) was added. Here P is the matrix (pᵢⱼ)
# from above, with possibly nonzero diagonal elements. D is the vector to store dᵢᵢ.

# Install packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()

# load new problem type for production-destruction systems
using PositiveIntegrators

using OrdinaryDiffEq
using SparseArrays

### Example 1: Linear model problem ######################################################################################
# This is an example of a conservative PDS
#
# ODE system: u₁' = -5 u₁ + 1 u₂, u₂' = 5 u₁ - 1 u₂
#
# Standard: f(t,u) = [-5 1; 5 -1]*u
#
# PDS: P = [0 u₂; 5 u₁ 0], D = [0; 0]
#
# We implement four variations (in-place and out-of-place for both standard and PDS) of this ODE system and check that we
# obtain equivalent solutions with a standard solver of OrdinaryDiffEq.jl.


# initial values
u0 = [0.9, 0.1]
# time domain
tspan = (0.0, 2.0)

# out-of-place syntax for f
A = [-5.0 1.0; 5.0 -1.0];
linmod(u,p,t) = A*u;
linmod_f_op = ODEProblem(linmod, u0, tspan)

# in-place syntax for f
function linmod!(du,u,p,t)
    u₁,u₂ = u
    du[1] = -5.0*u₁ + u₂
    du[2] = 5.0*u₁ - u₂
end
linmod_f_ip = ODEProblem(linmod!, u0, tspan)

# out-of-place syntax for PDS
linmodP(u,p,t) = [0.0 u[2]; 5.0*u[1] 0.0]
linmodD(u,p,t) = [0.0; 0.0]
linmod_PDS_op = ProdDestODEProblem(linmodP, linmodD, u0, tspan)

# in-place sytanx for PDS
function linmodP!(P,u,p,t)
    P .= 0.0
    P[1, 2] = u[2]
    P[2, 1] = 5.0*u[1]
    return nothing
end
function linmodD!(D,u,p,t)
    D .= 0.0
    return nothing
end
linmod_PDS_ip = ProdDestODEProblem(linmodP!, linmodD!, u0, tspan)

# solutions
sol_linmod_f_op = solve(linmod_f_op, Tsit5())
sol_linmod_f_ip = solve(linmod_f_ip,Tsit5())
sol_linmod_PDS_op = solve(linmod_PDS_op, Tsit5())
sol_linmod_PDS_ip = solve(linmod_PDS_ip,Tsit5())

# check equality of solutions
@assert sol_linmod_f_op.t ≈ sol_linmod_f_ip.t ≈ sol_linmod_PDS_op.t ≈ sol_linmod_PDS_ip.t
@assert sol_linmod_f_op.u ≈ sol_linmod_f_ip.u ≈ sol_linmod_PDS_op.u ≈ sol_linmod_PDS_ip.u

# check that we really do not use too many additional allocations for in-place implementations
alloc1 = @allocated(solve(linmod_f_ip, Tsit5()))
alloc2 = @allocated(solve(linmod_PDS_ip, Tsit5()))
@assert 0.95 < alloc1/alloc2 < 1.05

##########################################################################################################################
### Example 2: Lotka-Volterra ############################################################################################
# This is an example of a non-conservative PDS
#
# ODE system: N' = N - P*N, P' = P*N - P
#
# Standard: f(t,u) = [u₁ - u₁ u₂; u₁ u₂ - u₂]
#
# PDS: P = [u₁ 0; u₁ u₂ 0], D = [0; u₂]
#
# We implement four variations (in-place and out-of-place for both standard and PDS) of this ODE system and check that we
# obtain equivalent solutions with a standard solver of OrdinaryDiffEq.jl.

# initial values
u0 = [0.9, 0.1]
# time domain
tspan = (0.0, 20.0)

# out-of-place syntax for f
lotvol(u,p,t) = [u[1] - u[1]*u[2]; u[1]*u[2] - u[2]];
lotvol_f_op = ODEProblem(lotvol, u0, tspan)

# in-place syntax for f
function lotvol!(du,u,p,t)
    u₁,u₂ = u
    du[1] = u₁ - u₁*u₂
    du[2] = u₁*u₂ - u₂
end
lotvol_f_ip = ODEProblem(lotvol!, u0, tspan)

# out-of-place syntax for PDS
lotvolP(u,p,t) = [u[1] 0.0; u[1]*u[2] 0.0]
lotvolD(u,p,t) = [0.0; u[2]]
lotvol_PDS_op = ProdDestODEProblem(lotvolP, lotvolD, u0, tspan)

# in-place sytanx for PDS
function lotvolP!(P,u,p,t)
    P .= 0.0
    P[1, 1] = u[1]
    P[2, 1] = u[2]*u[1]
    return nothing
end
function lotvolD!(D,u,p,t)
    D .= 0.0
    D[2] = u[2]
    return nothing
end
lotvol_PDS_ip = ProdDestODEProblem(lotvolP!, lotvolD!, u0, tspan)

# solutions
sol_lotvol_f_op = solve(lotvol_f_op, Tsit5())
sol_lotvol_f_ip = solve(lotvol_f_ip,Tsit5())
sol_lotvol_PDS_op = solve(lotvol_PDS_op, Tsit5())
sol_lotvol_PDS_ip = solve(lotvol_PDS_ip,Tsit5())

# check equality of solutions
@assert sol_lotvol_f_op.t ≈ sol_lotvol_f_ip.t ≈ sol_lotvol_PDS_op.t ≈ sol_lotvol_PDS_ip.t
@assert sol_lotvol_f_op.u ≈ sol_lotvol_f_ip.u ≈ sol_lotvol_PDS_op.u ≈ sol_lotvol_PDS_ip.u

# check that we really do not use too many additional allocations for in-place implementations
alloc1 = @allocated(solve(lotvol_f_ip, Tsit5()))
alloc2 = @allocated(solve(lotvol_PDS_ip, Tsit5()))
@assert 0.95 < alloc1/alloc2 < 1.05

##########################################################################################################################
### Example 3: Linear advection discretized with finite differences and upwind, periodic boundary conditions #############
# This is an example of a large conservative PDS, which requires the use of sparese matrices.
#
# We implement the in-place versions for the standard rhs and the PDS representation of this ODE system. In addition, we
# compare the efficiency of dense and sparse matrices for the PDS version.

# number of nodes
N = 1000;
# initial data
u0 = sin.(π*LinRange(0.0,1.0,N+1))[2:end]
# time domain
tspan = (0.0, 1.0)

# in-place syntax for f
function fdupwind!(du,u,p,t)
    N = length(u);
    dx = 1/N;
    du[1] = -(u[1]-u[N])/dx
    for i=2:N
        du[i] = -(u[i]-u[i-1])/dx
    end
end
fdupwind_f = ODEProblem(fdupwind!, u0, tspan);

# in-place sytanx for PDS
function fdupwindP!(P,u,p,t)
    P .= 0.0
    N = length(u)
    dx = 1/N
    P[1, N] = u[N]/dx
    for i = 2:N
        P[i, i-1] = u[i-1]/dx
    end
    return nothing
end
function fdupwindP!(P::SparseMatrixCSC,u,p,t)
    N = length(u)
    dx = 1/N
    values = nonzeros(P)
    for col in axes(P, 2)
        for idx in nzrange(P, col)
            values[idx] = u[col] / dx
        end
    end
    return nothing
end
function fdupwindD!(D,u,p,t)
    D .= 0.0
    return nothing
end
# problem with dense matrices
fdupwind_PDS_dense = ProdDestODEProblem(fdupwindP!, fdupwindD!, u0, tspan);
# proboem with sparse matrices
p_prototype = spdiagm(-1 => ones(eltype(u0),N-1), N-1 => ones(eltype(u0),1))
d_prototype = zero(u0)
PD_sparse = ProdDestFunction(fdupwindP!,fdupwindD!;p_prototype = p_prototype, d_prototype = d_prototype);
fdupwind_PDS_sparse = ProdDestODEProblem(PD_sparse, u0, tspan);

# solutions
sol_fdupwind_f = solve(fdupwind_f, Tsit5());
sol_fdupwind_PDS_dense = solve(fdupwind_PDS_dense,Tsit5());
sol_fdupwind_PDS_sparse = solve(fdupwind_PDS_sparse,Tsit5());

# check equality of solutions
@assert sol_fdupwind_f.t ≈ sol_fdupwind_PDS_dense.t ≈ sol_fdupwind_PDS_sparse.t
@assert sol_fdupwind_f.u ≈ sol_fdupwind_PDS_dense.u ≈ sol_fdupwind_PDS_sparse.u

# Check that we really do not use too many additional allocations
alloc1 = @allocated(solve(fdupwind_f, Tsit5()))
alloc2 = @allocated(solve(fdupwind_PDS_dense, Tsit5()))
alloc3 = @allocated(solve(fdupwind_PDS_sparse, Tsit5()))
@assert 0.95 < alloc1/alloc2 < 1.05
@assert 0.95 < alloc1/alloc3 < 1.05

using BenchmarkTools
b1 = @benchmark solve(fdupwind_f, Tsit5())
b2 = @benchmark solve(fdupwind_PDS_dense, Tsit5())
b3 = @benchmark solve(fdupwind_PDS_sparse, Tsit5())

##########################################################################################################################