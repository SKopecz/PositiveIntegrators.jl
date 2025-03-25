using LinearAlgebra
using SparseArrays
using PositiveIntegrators
using LinearSolve

prod! = (P, u, p, t) -> begin
    fill!(P, zero(eltype(P)))
    # for j in axes(P, 2)
    #     for idx in nzrange(P, j)
    #         i = rowvals(P)[idx]
    #         nonzeros(P)[idx] = i * u[i]
    #     end
    # end
    for i in 1:(length(u) - 1)
        P[i, i + 1] = i * u[i]
    end
    return nothing
end

n = 4
P_tridiagonal = Tridiagonal([0.1, 0.2, 0.3],
                            zeros(n),
                            [0.4, 0.5, 0.6])
P_sparse = sparse(P_tridiagonal)
u0 = [1.0, 1.5, 2.0, 2.5]
tspan = (0.0, 1.0)
dt = 0.25

alg = MPE(; linsolve = KLUFactorization())
display(P_sparse)
prob_sparse_ip = ConservativePDSProblem(prod!, u0, tspan;
                                        p_prototype = P_sparse)
sol_sparse_ip = solve(prob_sparse_ip, alg; dt, adaptive = false)
