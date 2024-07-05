using Test
using LinearAlgebra
using SparseArrays
using Statistics: mean

using StaticArrays: MVector

using OrdinaryDiffEq
using PositiveIntegrators

using LinearSolve: RFLUFactorization, LUFactorization, KrylovJL_GMRES

using Aqua: Aqua

"""
    experimental_orders_of_convergence(prob, alg, dts;
                                      test_time = nothing,
                                      only_first_index = false,
                                      ref_alg = TRBDF2(autodiff = false))

Solve `prob` with `alg` and fixed time steps taken from `dts`, and compute
the errors at `test_time`. If`test_time` is not specified the error is computed 
at the final time.
Return the associated experimental orders of convergence.

If `only_first_index == true`, only the first solution component is used
to compute the error. If no analytic solution is available 
"""
function experimental_orders_of_convergence(prob, alg, dts; test_time = nothing,
                                            only_first_index = false,
                                            ref_alg = TRBDF2(autodiff = false))
    @assert length(dts) > 1
    errors = zeros(eltype(dts), length(dts))

    # use analytic solution
    if !(isnothing(prob.f.analytic))
        analytic = t -> prob.f.analytic(prob.u0, prob.p, t)
        if isnothing(test_time)
            if only_first_index
                for (i, dt) in enumerate(dts)
                    sol = solve(prob, alg; dt = dt, adaptive = false,
                                save_everystep = false)
                    errors[i] = norm(sol.u[end][1] - analytic(sol.t[end])[1])
                end
            else
                for (i, dt) in enumerate(dts)
                    sol = solve(prob, alg; dt = dt, adaptive = false,
                                save_everystep = false)
                    errors[i] = norm(sol.u[end] - analytic(sol.t[end]))
                end
            end
        else
            if only_first_index
                for (i, dt) in enumerate(dts)
                    sol = solve(prob, alg; dt = dt, adaptive = false)
                    errors[i] = norm(sol(test_time; idxs = 1) - first(analytic(test_time)))
                end
            else
                for (i, dt) in enumerate(dts)
                    sol = solve(prob, alg; dt = dt, adaptive = false)
                    errors[i] = norm(sol(test_time) - analytic(test_time))
                end
            end
        end
    else # need reference solution        
        if isnothing(test_time)
            dt0 = (prob.tspan[2] - prob.tspan[1]) / 1e5
            refsol = solve(prob, ref_alg; dt = dt0, adaptive = false,
                           save_everystep = false)
            if only_first_index
                for (i, dt) in enumerate(dts)
                    sol = solve(prob, alg; dt = dt, adaptive = false,
                                save_everystep = false)
                    errors[i] = norm(sol.u[end][1] - refsol.u[end][1])
                end
            else
                for (i, dt) in enumerate(dts)
                    sol = solve(prob, alg; dt = dt, adaptive = false,
                                save_everystep = false)
                    errors[i] = norm(sol.u[end] - refsol.u[end])
                end
            end
        else
            refsol = solve(prob, ref_alg; dt = dt0, adaptive = false)
            if only_first_index
                for (i, dt) in enumerate(dts)
                    sol = solve(prob, alg; dt = dt, adaptive = false)
                    errors[i] = norm(sol(test_time; idxs = 1) - refsol(test_time; idxs = 1))
                end
            else
                for (i, dt) in enumerate(dts)
                    sol = solve(prob, alg; dt = dt, adaptive = false)
                    errors[i] = norm(sol(test_time) - refsol(test_time))
                end
            end
        end
    end

    return experimental_orders_of_convergence(errors, dts)
end

"""
    experimental_orders_of_convergence(errors, dts)

Compute the experimental orders of convergence for given `errors` and
time step sizes `dts`.
"""
function experimental_orders_of_convergence(errors, dts)
    Base.require_one_based_indexing(errors, dts)
    @assert length(errors) == length(dts)
    orders = zeros(eltype(errors), length(errors) - 1)

    for i in eachindex(orders)
        orders[i] = log(errors[i] / errors[i + 1]) / log(dts[i] / dts[i + 1])
    end

    return orders
end

"""
    check_order(orders, alg_order; N = 3, atol = 0.1)

Check if `alg_order` can be found approximately in `N` consecutive elements of `orders`.
"""
function check_order(orders, alg_order; N = 3, atol = 0.1)
    check = false
    # The calculation of the indices of the permissible orders by
    #
    #    indices = findall(x -> isapprox(x, alg_order; atol = atol), orders)
    #
    # sometimes failed because the experimental order was too good. To accept
    # such cases, but avoid to decrease atol, we now use the following asymmetric criterion:
    indices = findall(x -> -atol ≤ x - alg_order ≤ 2 * atol, orders)

    for (i, idx) in enumerate(indices)
        if i + N - 1 ≤ length(indices) && indices[i:(i + N - 1)] == idx:(idx + N - 1)
            check = true
            break
        end
    end
    return check
end

const prob_pds_linmod_array = ConservativePDSProblem(prob_pds_linmod.f,
                                                     Array(prob_pds_linmod.u0),
                                                     prob_pds_linmod.tspan)
const prob_pds_linmod_mvector = ConservativePDSProblem(prob_pds_linmod_inplace.f,
                                                       MVector(prob_pds_linmod.u0),
                                                       prob_pds_linmod.tspan)

# analytic solution of linear model problem
function f_analytic(u0, p, t)
    u₁⁰, u₂⁰ = u0
    a, b = p
    c = a + b
    return ((u₁⁰ + u₂⁰) * [b; a] +
            exp(-c * t) * (a * u₁⁰ - b * u₂⁰) * [1; -1]) / c
end

# This is the usual conservative linear model problem, rewritten as
# u₁' = -3 u₁ + 0.5 u₂ - 2 u₁ + 0.5 u₂ (= -5 u₁ + u₂)
# u₂' =  3 u₁ - 0.5 u₂ - 0.5 u₂ + 2 u₁ (= 5 u₁ - u₂)  
# linear model problem - nonconservative - out-of-place
linmodP(u, p, t) = [0.5*u[2] 0.5*u[2]; 3*u[1] 2*u[1]]
linmodD(u, p, t) = [2 * u[1]; 0.5 * u[2]]
const prob_pds_linmod_nonconservative = PDSProblem(linmodP, linmodD, [0.9, 0.1], (0.0, 2.0),
                                                   [5.0, 1.0];
                                                   analytic = f_analytic)

# linear model problem - nonconservative -  in-place
function linmodP!(P, u, p, t)
    P[1, 1] = 0.5 * u[2]
    P[1, 2] = 0.5 * u[2]
    P[2, 1] = 3 * u[1]
    P[2, 2] = 2 * u[1]
    return nothing
end
function linmodD!(D, u, p, t)
    D[1] = 2 * u[1]
    D[2] = 0.5 * u[2]
    return nothing
end
const prob_pds_linmod_nonconservative_inplace = PDSProblem(linmodP!, linmodD!, [0.9, 0.1],
                                                           (0.0, 2.0), [5.0, 1.0];
                                                           analytic = f_analytic)
@testset "PositiveIntegrators.jl tests" begin
    @testset "Aqua.jl" begin
        # We do not test ambiguities since we get a lot of
        # false positives from dependencies
        Aqua.test_all(PositiveIntegrators;
                      ambiguities = false,)
    end

    @testset "ConservativePDSFunction" begin
        prod_1! = (P, u, p, t) -> begin
            fill!(P, zero(eltype(P)))
            for i in 1:(length(u) - 1)
                P[i, i + 1] = i * u[i]
            end
            return nothing
        end
        prod_2! = (P, u, p, t) -> begin
            fill!(P, zero(eltype(P)))
            for i in 1:(length(u) - 1)
                P[i + 1, i] = i * u[i + 1]
            end
            return nothing
        end
        prod_3! = (P, u, p, t) -> begin
            fill!(P, zero(eltype(P)))
            for i in 1:(length(u) - 1)
                P[i, i + 1] = i * u[i]
                P[i + 1, i] = i * u[i + 1]
            end
            return nothing
        end

        n = 10
        P_tridiagonal = Tridiagonal(rand(n - 1), zeros(n), rand(n - 1))
        P_dense = Matrix(P_tridiagonal)
        P_sparse = sparse(P_tridiagonal)
        u0 = rand(n)
        tspan = (0.0, 1.0)

        du_tridiagonal = similar(u0)
        du_dense = similar(u0)
        du_sparse = similar(u0)

        for prod! in (prod_1!, prod_2!, prod_3!)
            prob_tridiagonal = ConservativePDSProblem(prod!, u0, tspan;
                                                      p_prototype = P_tridiagonal)
            prob_dense = ConservativePDSProblem(prod!, u0, tspan;
                                                p_prototype = P_dense)
            prob_sparse = ConservativePDSProblem(prod!, u0, tspan;
                                                 p_prototype = P_sparse)

            prob_tridiagonal.f(du_tridiagonal, u0, nothing, 0.0)
            prob_dense.f(du_dense, u0, nothing, 0.0)
            prob_sparse.f(du_sparse, u0, nothing, 0.0)

            @test du_tridiagonal ≈ du_dense
            @test du_tridiagonal ≈ du_sparse
            @test du_dense ≈ du_sparse
        end
    end

    @testset "PDSProblem" begin
        @testset "Linear model problem" begin
            # This is an example of a conservative PDS
            #
            # ODE system: u₁' = -5 u₁ + 1 u₂, u₂' = 5 u₁ - 1 u₂
            #
            # Standard: f(t,u) = [-5 1; 5 -1]*u
            #
            # PDS: P = [0 u₂; 5 u₁ 0], D = [0; 0]
            #
            # We implement four variations (in-place and out-of-place
            # for both standard and PDS) of this ODE system and check
            # that we obtain equivalent solutions with a standard
            # solver of OrdinaryDiffEq.jl.

            # initial values
            u0 = [0.9, 0.1]
            # time domain
            tspan = (0.0, 2.0)

            # out-of-place syntax for standard ODE
            A = [-5.0 1.0; 5.0 -1.0]
            linmod(u, p, t) = A * u
            linmod_ODE_op = ODEProblem(linmod, u0, tspan)

            # in-place syntax for standard ODE
            function linmod!(du, u, p, t)
                u1, u2 = u
                du[1] = -5 * u1 + u2
                du[2] = 5 * u1 - u2
                return nothing
            end
            linmod_ODE_ip = ODEProblem(linmod!, u0, tspan)

            # out-of-place syntax for PDS
            linmodP(u, p, t) = [0 u[2]; 5*u[1] 0]
            linmodD(u, p, t) = [0.0; 0.0]
            linmod_PDS_op = PDSProblem(linmodP, linmodD, u0, tspan)

            # in-place sytanx for PDS
            function linmodP!(P, u, p, t)
                fill!(P, zero(eltype(P)))
                P[1, 2] = u[2]
                P[2, 1] = 5 * u[1]
                return nothing
            end
            function linmodD!(D, u, p, t)
                fill!(D, zero(eltype(D)))
                return nothing
            end
            linmod_PDS_ip = PDSProblem(linmodP!, linmodD!, u0, tspan)

            # solutions
            sol_linmod_ODE_op = solve(linmod_ODE_op, Tsit5())
            sol_linmod_ODE_ip = solve(linmod_ODE_ip, Tsit5())
            sol_linmod_PDS_op = solve(linmod_PDS_op, Tsit5())
            sol_linmod_PDS_ip = solve(linmod_PDS_ip, Tsit5())

            # check equality of solutions
            @test sol_linmod_ODE_op.t ≈ sol_linmod_ODE_ip.t
            @test sol_linmod_ODE_op.t ≈ sol_linmod_PDS_op.t
            @test sol_linmod_ODE_op.t ≈ sol_linmod_PDS_ip.t
            @test sol_linmod_ODE_op.u ≈ sol_linmod_ODE_ip.u
            @test sol_linmod_ODE_op.u ≈ sol_linmod_PDS_op.u
            @test sol_linmod_ODE_op.u ≈ sol_linmod_PDS_ip.u

            # check that we really do not use too many additional
            # allocations for in-place implementations
            alloc1 = @allocated(solve(linmod_ODE_ip, Tsit5()))
            alloc2 = @allocated(solve(linmod_PDS_ip, Tsit5()))
            @test 0.95 < alloc1 / alloc2 < 1.05
        end

        @testset "PDSProblem error handling" begin
            P(u, p, t) = 0.0
            D(du, u, p, t) = 0.0
            if VERSION >= v"1.8"
                @test_throws "in-place and out-of-place" PDSProblem(P, D, 0.0, (0.0, 1.0))
            end
        end
    end

    # Here we check that solutions of equivalten ODEProblems, PDSProblems or 
    # ConservativePDS Problems are approximately equal. 
    # We also check that solvers from OrdinaryDiffEq can solve PDSProblems and 
    # ConservativePDSProblems.
    @testset "Check compatibility of PositiveIntegrators and OrdinaryDiffEq" begin
        @testset "Linear model" begin
            # Linear model (conservative)
            u0 = [0.9, 0.1]
            tspan = (0.0, 2.0)
            A = [-5.0 1.0; 5.0 -1.0]
            linmod(u, p, t) = A * u
            linmod_f_op = ODEProblem(linmod, u0, tspan)
            # in-place syntax for f
            function linmod!(du, u, p, t)
                u₁, u₂ = u
                du[1] = -5.0 * u₁ + u₂
                du[2] = 5.0 * u₁ - u₂
            end
            linmod_f_ip = ODEProblem(linmod!, u0, tspan)
            # out-of-place syntax for PDS
            linmodP(u, p, t) = [0.0 u[2]; 5.0*u[1] 0.0]
            linmodD(u, p, t) = [0.0; 0.0]
            linmod_PDS_op = PDSProblem(linmodP, linmodD, u0, tspan)
            linmod_PDS_op_2 = PDSProblem{false}(linmodP, linmodD, u0, tspan)
            linmod_ConsPDS_op = ConservativePDSProblem(linmodP, u0, tspan)
            linmod_ConsPDS_op_2 = ConservativePDSProblem{false}(linmodP, u0, tspan)
            # in-place sytanx for PDS
            function linmodP!(P, u, p, t)
                P .= 0.0
                P[1, 2] = u[2]
                P[2, 1] = 5.0 * u[1]
                return nothing
            end
            function linmodD!(D, u, p, t)
                D .= 0.0
                return nothing
            end
            linmod_PDS_ip = PDSProblem(linmodP!, linmodD!, u0, tspan)
            linmod_PDS_ip_2 = PDSProblem{true}(linmodP!, linmodD!, u0, tspan)
            linmod_ConsPDS_ip = ConservativePDSProblem(linmodP!, u0, tspan)
            linmod_ConsPDS_ip_2 = ConservativePDSProblem{true}(linmodP!, u0, tspan)

            # solutions
            sol_linmod_f_op = solve(linmod_f_op, Tsit5())
            sol_linmod_f_ip = solve(linmod_f_ip, Tsit5())
            sol_linmod_PDS_op = solve(linmod_PDS_op, Tsit5())
            sol_linmod_PDS_op_2 = solve(linmod_PDS_op_2, Tsit5())
            sol_linmod_PDS_ip = solve(linmod_PDS_ip, Tsit5())
            sol_linmod_PDS_ip_2 = solve(linmod_PDS_ip_2, Tsit5())
            sol_linmod_ConsPDS_op = solve(linmod_ConsPDS_op, Tsit5())
            sol_linmod_ConsPDS_op_2 = solve(linmod_ConsPDS_op_2, Tsit5())
            sol_linmod_ConsPDS_ip = solve(linmod_ConsPDS_ip, Tsit5())
            sol_linmod_ConsPDS_ip_2 = solve(linmod_ConsPDS_ip_2, Tsit5())

            # check equality of solutions
            @test sol_linmod_f_op.t ≈ sol_linmod_f_ip.t ≈
                  sol_linmod_PDS_op.t ≈ sol_linmod_PDS_ip.t ≈
                  sol_linmod_PDS_op_2.t ≈ sol_linmod_PDS_ip_2.t ≈
                  sol_linmod_ConsPDS_op.t ≈ sol_linmod_ConsPDS_ip.t ≈
                  sol_linmod_ConsPDS_op_2.t ≈ sol_linmod_ConsPDS_ip_2.t
            @test sol_linmod_f_op.u ≈ sol_linmod_f_ip.u ≈
                  sol_linmod_PDS_op.u ≈ sol_linmod_PDS_ip.u ≈
                  sol_linmod_PDS_op_2.u ≈ sol_linmod_PDS_ip_2.u ≈
                  sol_linmod_ConsPDS_op.u ≈ sol_linmod_ConsPDS_ip.u ≈
                  sol_linmod_ConsPDS_op_2.u ≈ sol_linmod_ConsPDS_ip_2.u

            # check that we really do not use too many additional allocations for in-place implementations
            alloc1 = @allocated(solve(linmod_f_ip, Tsit5()))
            alloc2 = @allocated(solve(linmod_PDS_ip, Tsit5()))
            alloc3 = @allocated(solve(linmod_PDS_ip_2, Tsit5()))
            alloc4 = @allocated(solve(linmod_ConsPDS_ip, Tsit5()))
            alloc5 = @allocated(solve(linmod_ConsPDS_ip_2, Tsit5()))
            @test 0.95 < alloc1 / alloc2 < 1.05
            @test 0.95 < alloc1 / alloc3 < 1.05
            @test 0.95 < alloc1 / alloc4 < 1.05
            @test 0.95 < alloc1 / alloc5 < 1.05
        end
        @testset "Lotka-Volterra" begin
            # Lotka-Volterra (nonconservative)
            u0 = [0.9, 0.1]
            tspan = (0.0, 20.0)
            # out-of-place syntax for f
            lotvol(u, p, t) = [u[1] - u[1] * u[2]; u[1] * u[2] - u[2]]
            lotvol_f_op = ODEProblem(lotvol, u0, tspan)
            # in-place syntax for f
            function lotvol!(du, u, p, t)
                u₁, u₂ = u
                du[1] = u₁ - u₁ * u₂
                du[2] = u₁ * u₂ - u₂
            end
            lotvol_f_ip = ODEProblem(lotvol!, u0, tspan)
            # out-of-place syntax for PDS
            lotvolP(u, p, t) = [u[1] 0.0; u[1]*u[2] 0.0]
            lotvolD(u, p, t) = [0.0; u[2]]
            lotvol_PDS_op = PDSProblem(lotvolP, lotvolD, u0, tspan)
            lotvol_PDS_op_2 = PDSProblem{false}(lotvolP, lotvolD, u0, tspan)
            # in-place sytanx for PDS
            function lotvolP!(P, u, p, t)
                P .= 0.0
                P[1, 1] = u[1]
                P[2, 1] = u[2] * u[1]
                return nothing
            end
            function lotvolD!(D, u, p, t)
                D .= 0.0
                D[2] = u[2]
                return nothing
            end
            lotvol_PDS_ip = PDSProblem(lotvolP!, lotvolD!, u0, tspan)
            lotvol_PDS_ip_2 = PDSProblem{true}(lotvolP!, lotvolD!, u0, tspan)

            # solutions
            sol_lotvol_f_op = solve(lotvol_f_op, Tsit5())
            sol_lotvol_f_ip = solve(lotvol_f_ip, Tsit5())
            sol_lotvol_PDS_op = solve(lotvol_PDS_op, Tsit5())
            sol_lotvol_PDS_op_2 = solve(lotvol_PDS_op_2, Tsit5())
            sol_lotvol_PDS_ip = solve(lotvol_PDS_ip, Tsit5())
            sol_lotvol_PDS_ip_2 = solve(lotvol_PDS_ip_2, Tsit5())

            # check equality of solutions
            @test sol_lotvol_f_op.t ≈ sol_lotvol_f_ip.t ≈
                  sol_lotvol_PDS_op.t ≈ sol_lotvol_PDS_op_2.t ≈
                  sol_lotvol_PDS_ip.t ≈ sol_lotvol_PDS_ip_2.t
            @test sol_lotvol_f_op.u ≈ sol_lotvol_f_ip.u ≈
                  sol_lotvol_PDS_op.u ≈ sol_lotvol_PDS_op_2.u ≈
                  sol_lotvol_PDS_ip.u ≈ sol_lotvol_PDS_ip_2.u

            # check that we really do not use too many additional allocations for in-place implementations
            alloc1 = @allocated(solve(lotvol_f_ip, Tsit5()))
            alloc2 = @allocated(solve(lotvol_PDS_ip, Tsit5()))
            alloc3 = @allocated(solve(lotvol_PDS_ip_2, Tsit5()))
            @test 0.95 < alloc1 / alloc2 < 1.05
            @test 0.95 < alloc1 / alloc3 < 1.05
        end
        @testset "Linear advection" begin
            # Linear advection discretized with finite differences and upwind, periodic boundary conditions
            # number of nodes
            N = 1000
            u0 = sin.(π * LinRange(0.0, 1.0, N + 1))[2:end]
            tspan = (0.0, 1.0)
            # in-place syntax for f
            function fdupwind!(du, u, p, t)
                N = length(u)
                dx = 1 / N
                du[1] = -(u[1] - u[N]) / dx
                for i in 2:N
                    du[i] = -(u[i] - u[i - 1]) / dx
                end
            end
            fdupwind_f = ODEProblem(fdupwind!, u0, tspan)
            # in-place sytanx for PDS
            function fdupwindP!(P, u, p, t)
                P .= 0.0
                N = length(u)
                dx = 1 / N
                P[1, N] = u[N] / dx
                for i in 2:N
                    P[i, i - 1] = u[i - 1] / dx
                end
                return nothing
            end
            function fdupwindP!(P::SparseMatrixCSC, u, p, t)
                N = length(u)
                dx = 1 / N
                values = nonzeros(P)
                for col in axes(P, 2)
                    for idx in nzrange(P, col)
                        values[idx] = u[col] / dx
                    end
                end
                return nothing
            end
            function fdupwindD!(D, u, p, t)
                D .= 0.0
                return nothing
            end
            # problem with dense matrices
            fdupwind_PDS_dense = PDSProblem(fdupwindP!, fdupwindD!, u0, tspan)
            # problem with sparse matrices
            p_prototype = spdiagm(-1 => ones(eltype(u0), N - 1),
                                  N - 1 => ones(eltype(u0), 1))
            d_prototype = zero(u0)
            fdupwind_PDS_sparse = PDSProblem(fdupwindP!, fdupwindD!, u0, tspan;
                                             p_prototype = p_prototype,
                                             d_prototype = d_prototype)
            fdupwind_PDS_sparse_2 = PDSProblem{true}(fdupwindP!, fdupwindD!, u0, tspan;
                                                     p_prototype = p_prototype,
                                                     d_prototype = d_prototype)
            fdupwind_ConsPDS_sparse = ConservativePDSProblem(fdupwindP!, u0, tspan;
                                                             p_prototype = p_prototype)
            fdupwind_ConsPDS_sparse_2 = ConservativePDSProblem{true}(fdupwindP!, u0, tspan;
                                                                     p_prototype = p_prototype)

            # solutions
            sol_fdupwind_f = solve(fdupwind_f, Tsit5())
            sol_fdupwind_PDS_dense = solve(fdupwind_PDS_dense, Tsit5())
            sol_fdupwind_PDS_sparse = solve(fdupwind_PDS_sparse, Tsit5())
            sol_fdupwind_PDS_sparse_2 = solve(fdupwind_PDS_sparse_2, Tsit5())
            sol_fdupwind_ConsPDS_sparse = solve(fdupwind_ConsPDS_sparse, Tsit5())
            sol_fdupwind_ConsPDS_sparse_2 = solve(fdupwind_ConsPDS_sparse_2, Tsit5())

            # check equality of solutions
            @test sol_fdupwind_f.t ≈ sol_fdupwind_PDS_dense.t ≈
                  sol_fdupwind_PDS_sparse.t ≈ sol_fdupwind_PDS_sparse_2.t ≈
                  sol_fdupwind_ConsPDS_sparse.t ≈ sol_fdupwind_ConsPDS_sparse_2.t
            @test sol_fdupwind_f.u ≈ sol_fdupwind_PDS_dense.u ≈
                  sol_fdupwind_PDS_sparse.u ≈ sol_fdupwind_PDS_sparse_2.u ≈
                  sol_fdupwind_ConsPDS_sparse.u ≈ sol_fdupwind_ConsPDS_sparse_2.u

            # Check that we really do not use too many additional allocations
            #TODO: The tests below should pass.
            #=
            alloc1 = @allocated(solve(fdupwind_f, Tsit5()))
            alloc2 = @allocated(solve(fdupwind_PDS_dense, Tsit5()))
            alloc3 = @allocated(solve(fdupwind_PDS_sparse, Tsit5()))
            alloc4 = @allocated(solve(fdupwind_PDS_sparse_2, Tsit5()))
            alloc5 = @allocated(solve(fdupwind_ConsPDS_sparse, Tsit5()))
            alloc6 = @allocated(solve(fdupwind_ConsPDS_sparse_2, Tsit5()))
            @test 0.95 < alloc1 / alloc2 < 1.05
            @test 0.95 < alloc1 / alloc3 < 1.05
            @test 0.95 < alloc1 / alloc4 < 1.05
            @test 0.95 < alloc1 / alloc5 < 1.05
            @test 0.95 < alloc1 / alloc6 < 1.05
            =#
        end
    end

    @testset "PDS Solvers" begin
        # Here we check that MPRK schemes require a PDSProblem or ConservativePDSProblem.
        # We also check that only permissible parameters can be used.
        @testset "Error handling" begin
            f_oop = (u, p, t) -> u
            f_ip = (du, u, p, t) -> du .= u
            prob_oop = ODEProblem(f_oop, [1.0; 2.0], (0.0, 1.0))
            prob_ip = ODEProblem(f_ip, [1.0; 2.0], (0.0, 1.0))
            @test_throws "MPE can only be applied to production-destruction systems" solve(prob_oop,
                                                                                           MPE(),
                                                                                           dt = 0.25)
            @test_throws "MPE can only be applied to production-destruction systems" solve(prob_ip,
                                                                                           MPE(),
                                                                                           dt = 0.25)
            @test_throws "MPRK22 can only be applied to production-destruction systems" solve(prob_oop,
                                                                                              MPRK22(1.0))
            @test_throws "MPRK22 can only be applied to production-destruction systems" solve(prob_ip,
                                                                                              MPRK22(1.0))
            @test_throws "MPRK22 requires α ≥ 1/2." solve(prob_pds_linmod, MPRK22(0.25))
            @test_throws "MPRK43 can only be applied to production-destruction systems" solve(prob_oop,
                                                                                              MPRK43I(1.0,
                                                                                                      0.5))
            @test_throws "MPRK43 can only be applied to production-destruction systems" solve(prob_ip,
                                                                                              MPRK43I(1.0,
                                                                                                      0.5))
            @test_throws "MPRK43 can only be applied to production-destruction systems" solve(prob_oop,
                                                                                              MPRK43II(0.5))
            @test_throws "MPRK43 can only be applied to production-destruction systems" solve(prob_ip,
                                                                                              MPRK43II(0.5))
            @test_throws "MPRK43I requires α ≥ 1/3 and α ≠ 2/3." solve(prob_pds_linmod,
                                                                       MPRK43I(0.0, 0.5))
            @test_throws "MPRK43I requires α ≥ 1/3 and α ≠ 2/3." solve(prob_pds_linmod,
                                                                       MPRK43I(2.0 / 3.0,
                                                                               0.5))
            @test_throws "For this choice of α MPRK43I requires 2/3 ≤ β ≤ 3α(1-α)." solve(prob_pds_linmod,
                                                                                          MPRK43I(0.5,
                                                                                                  0.5))
            @test_throws "For this choice of α MPRK43I requires 2/3 ≤ β ≤ 3α(1-α)." solve(prob_pds_linmod,
                                                                                          MPRK43I(0.5,
                                                                                                  0.8))
            @test_throws "For this choice of α MPRK43I requires 3α(1-α) ≤ β ≤ 2/3." solve(prob_pds_linmod,
                                                                                          MPRK43I(0.7,
                                                                                                  0.7))
            @test_throws "For this choice of α MPRK43I requires 3α(1-α) ≤ β ≤ 2/3." solve(prob_pds_linmod,
                                                                                          MPRK43I(0.7,
                                                                                                  0.1))
            @test_throws "For this choice of α MPRK43I requires (3α-2)/(6α-3) ≤ β ≤ 2/3." solve(prob_pds_linmod,
                                                                                                MPRK43I(1.0,
                                                                                                        0.7))
            @test_throws "For this choice of α MPRK43I requires (3α-2)/(6α-3) ≤ β ≤ 2/3." solve(prob_pds_linmod,
                                                                                                MPRK43I(1.0,
                                                                                                        0.1))
            @test_throws "MPRK43II requires 3/8 ≤ γ ≤ 3/4." solve(prob_pds_linmod,
                                                                  MPRK43II(1.0))
            @test_throws "MPRK43II requires 3/8 ≤ γ ≤ 3/4." solve(prob_pds_linmod,
                                                                  MPRK43II(0.0))
            @test_throws "SSPMPRK22 can only be applied to production-destruction systems" solve(prob_oop,
                                                                                                 SSPMPRK22(0.5,
                                                                                                           1.0))
            @test_throws "SSPMPRK22 can only be applied to production-destruction systems" solve(prob_ip,
                                                                                                 SSPMPRK22(0.5,
                                                                                                           1.0))
            @test_throws "SSPMPRK22 requires 0 ≤ α ≤ 1, β ≥ 0 and αβ + 1/(2β) ≤ 1." solve(prob_pds_linmod,
                                                                                          SSPMPRK22(-1.0,
                                                                                                    1.0))
            @test_throws "SSPMPRK22 requires 0 ≤ α ≤ 1, β ≥ 0 and αβ + 1/(2β) ≤ 1." solve(prob_pds_linmod,
                                                                                          SSPMPRK22(0.0,
                                                                                                    -1.0))
            @test_throws "SSPMPRK22 requires 0 ≤ α ≤ 1, β ≥ 0 and αβ + 1/(2β) ≤ 1." solve(prob_pds_linmod,
                                                                                          SSPMPRK22(1.0,
                                                                                                    10.0))
            @test_throws "SSPMPRK43 can only be applied to production-destruction systems" solve(prob_oop,
                                                                                                 SSPMPRK43(),
                                                                                                 dt = 0.1)
            @test_throws "SSPMPRK43 can only be applied to production-destruction systems" solve(prob_ip,
                                                                                                 SSPMPRK43(),
                                                                                                 dt = 0.1)
        end

        # Here we check that MPE equals implicit Euler (IE) for a linear PDS
        @testset "Linear model problem: MPE = IE?" begin
            # problem data
            u0 = [0.9, 0.1]
            tspan = (0.0, 2.0)
            p = [5.0, 1.0]

            # analytic solution
            function f_analytic(u0, p, t)
                u₁⁰, u₂⁰ = u0
                a, b = p
                c = a + b
                return ((u₁⁰ + u₂⁰) * [b; a] +
                        exp(-c * t) * (a * u₁⁰ - b * u₂⁰) * [1; -1]) / c
            end

            # linear model problem - out-of-place
            linmodP(u, p, t) = [0 p[2]*u[2]; p[1]*u[1] 0]
            linmodD(u, p, t) = [0.0; 0.0]
            prob_op = PDSProblem(linmodP, linmodD, u0, tspan, p; analytic = f_analytic)
            prob_op_2 = ConservativePDSProblem(linmodP, u0, tspan, p; analytic = f_analytic)

            dt = 0.25
            sol_MPE_op = solve(prob_op, MPE(); dt)
            sol_MPE_op_2 = solve(prob_op_2, MPE(); dt)
            sol_IE_op = solve(prob_op, ImplicitEuler(autodiff = false);
                              dt, adaptive = false)
            @test sol_MPE_op.t ≈ sol_MPE_op_2.t ≈ sol_IE_op.t
            @test sol_MPE_op.u ≈ sol_MPE_op_2.u ≈ sol_IE_op.u

            # linear model problem - in-place
            function linmodP!(P, u, p, t)
                fill!(P, zero(eltype(P)))
                P[1, 2] = p[2] * u[2]
                P[2, 1] = p[1] * u[1]
                return nothing
            end
            function linmodD!(D, u, p, t)
                fill!(D, zero(eltype(D)))
                return nothing
            end
            prob_ip = PDSProblem(linmodP!, linmodD!, u0, tspan, p; analytic = f_analytic)
            prob_ip_2 = ConservativePDSProblem(linmodP!, u0, tspan, p;
                                               analytic = f_analytic)

            dt = 0.25
            sol_MPE_ip = solve(prob_ip, MPE(); dt)
            sol_MPE_ip_2 = solve(prob_ip_2, MPE(); dt)
            sol_IE_ip = solve(prob_ip, ImplicitEuler(autodiff = false);
                              dt, adaptive = false)
            @test sol_MPE_ip.t ≈ sol_MPE_ip_2.t ≈ sol_IE_ip.t
            @test sol_MPE_ip.u ≈ sol_MPE_ip_2.u ≈ sol_IE_ip.u
        end

        # Here we check that MPRK22(α) = SSPMPRK22(0,α)
        @testset "MPRK22(α) = SSPMPRK22(0, α)" begin
            for α in (0.5, 2.0 / 3.0, 1.0, 2.0)
                sol1 = solve(prob_pds_linmod, MPRK22(α))
                sol2 = solve(prob_pds_linmod, SSPMPRK22(0.0, α))
                sol3 = solve(prob_pds_linmod_inplace, MPRK22(α))
                sol4 = solve(prob_pds_linmod_inplace, SSPMPRK22(0.0, α))
                @test sol1.u ≈ sol2.u ≈ sol3.u ≈ sol4.u
            end
        end

        # Here we check that different linear solvers can be used
        @testset "Different linear solvers" begin
            # problem data
            u0 = [0.9, 0.1]
            tspan = (0.0, 2.0)
            p = [5.0, 1.0]

            # analytic solution
            function f_analytic(u0, p, t)
                u₁⁰, u₂⁰ = u0
                a, b = p
                c = a + b
                return ((u₁⁰ + u₂⁰) * [b; a] +
                        exp(-c * t) * (a * u₁⁰ - b * u₂⁰) * [1; -1]) / c
            end

            # linear model problem - in-place
            function linmodP!(P, u, p, t)
                fill!(P, zero(eltype(P)))
                P[1, 2] = p[2] * u[2]
                P[2, 1] = p[1] * u[1]
                return nothing
            end
            function linmodD!(D, u, p, t)
                fill!(D, zero(eltype(D)))
                return nothing
            end
            prob_ip = PDSProblem(linmodP!, linmodD!, u0, tspan, p; analytic = f_analytic)
            prob_ip_2 = ConservativePDSProblem(linmodP!, u0, tspan, p;
                                               analytic = f_analytic)

            algs = (MPE, (; kwargs...) -> MPRK22(1.0; kwargs...),
                    (; kwargs...) -> MPRK22(0.5; kwargs...),
                    (; kwargs...) -> MPRK22(2.0; kwargs...),
                    (; kwargs...) -> MPRK43I(1.0, 0.5; kwargs...),
                    (; kwargs...) -> MPRK43I(0.5, 0.75; kwargs...),
                    (; kwargs...) -> MPRK43II(0.5; kwargs...),
                    (; kwargs...) -> MPRK43II(2.0 / 3.0; kwargs...),
                    (; kwargs...) -> SSPMPRK22(0.5, 1.0; kwargs...),
                    (; kwargs...) -> SSPMPRK43(; kwargs...))

            for alg in algs
                # Check different linear solvers
                dt = 0.25
                sol1 = solve(prob_ip, alg(); dt)
                sol1_2 = solve(prob_ip_2, alg(); dt)
                sol2 = solve(prob_ip, alg(linsolve = RFLUFactorization()); dt)
                sol2_2 = solve(prob_ip_2, alg(linsolve = RFLUFactorization()); dt)
                sol3 = solve(prob_ip, alg(linsolve = LUFactorization()); dt)
                sol3_2 = solve(prob_ip_2, alg(linsolve = LUFactorization()); dt)
                sol4 = solve(prob_ip, alg(linsolve = KrylovJL_GMRES()); dt)
                sol4_2 = solve(prob_ip_2, alg(linsolve = KrylovJL_GMRES()); dt)
                @test sol1.t ≈ sol2.t ≈ sol3.t ≈ sol4.t ≈ sol1_2.t ≈ sol2_2.t ≈ sol3_2.t ≈
                      sol4_2.t
                @test sol1.u ≈ sol2.u ≈ sol3.u ≈ sol4.u ≈ sol1_2.u ≈ sol2_2.u ≈ sol3_2.u ≈
                      sol4_2.u
            end
        end

        # Here we check that in-place and out-of-place implementations
        # deliver the same results
        @testset "Different matrix types (conservative)" begin
            prod_1! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i, i + 1] = i * u[i]
                end
                return nothing
            end

            prod_2! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i + 1, i] = i * u[i + 1]
                end
                return nothing
            end

            prod_3! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i, i + 1] = i * u[i]
                    P[i + 1, i] = i * u[i + 1]
                end
                return nothing
            end

            n = 4
            P_tridiagonal = Tridiagonal([0.1, 0.2, 0.3],
                                        zeros(n),
                                        [0.4, 0.5, 0.6])
            P_dense = Matrix(P_tridiagonal)
            P_sparse = sparse(P_tridiagonal)
            u0 = [1.0, 1.5, 2.0, 2.5]
            tspan = (0.0, 1.0)
            dt = 0.25

            @testset "$alg" for alg in (MPE(),
                                        MPRK22(0.5), MPRK22(1.0),
                                        MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                                        MPRK43II(2.0 / 3.0), MPRK43II(0.5),
                                        SSPMPRK22(0.5, 1.0), SSPMPRK43())
                for prod! in (prod_1!, prod_2!, prod_3!)
                    prod = (u, p, t) -> begin
                        P = similar(u, (length(u), length(u)))
                        prod!(P, u, p, t)
                        return P
                    end
                    prob_tridiagonal_ip = ConservativePDSProblem(prod!, u0, tspan;
                                                                 p_prototype = P_tridiagonal)
                    prob_tridiagonal_op = ConservativePDSProblem(prod, u0, tspan;
                                                                 p_prototype = P_tridiagonal)
                    prob_dense_ip = ConservativePDSProblem(prod!, u0, tspan;
                                                           p_prototype = P_dense)
                    prob_dense_op = ConservativePDSProblem(prod, u0, tspan;
                                                           p_prototype = P_dense)
                    prob_sparse_ip = ConservativePDSProblem(prod!, u0, tspan;
                                                            p_prototype = P_sparse)
                    prob_sparse_op = ConservativePDSProblem(prod, u0, tspan;
                                                            p_prototype = P_sparse)

                    sol_tridiagonal_ip = solve(prob_tridiagonal_ip, alg; dt,
                                               adaptive = false)
                    sol_tridiagonal_op = solve(prob_tridiagonal_op, alg; dt,
                                               adaptive = false)
                    sol_dense_ip = solve(prob_dense_ip, alg; dt, adaptive = false)
                    sol_dense_op = solve(prob_dense_op, alg; dt, adaptive = false)
                    sol_sparse_ip = solve(prob_sparse_ip, alg; dt, adaptive = false)
                    sol_sparse_op = solve(prob_sparse_op, alg; dt, adaptive = false)

                    @test sol_tridiagonal_ip.t ≈ sol_tridiagonal_op.t
                    @test sol_dense_ip.t ≈ sol_dense_op.t
                    @test sol_sparse_ip.t ≈ sol_sparse_op.t
                    @test sol_tridiagonal_ip.t ≈ sol_dense_ip.t
                    @test sol_tridiagonal_ip.t ≈ sol_sparse_ip.t

                    @test sol_tridiagonal_ip.u ≈ sol_tridiagonal_op.u
                    @test sol_dense_ip.u ≈ sol_dense_op.u
                    @test sol_sparse_ip.u ≈ sol_sparse_op.u
                    @test sol_tridiagonal_ip.u ≈ sol_dense_ip.u
                    @test sol_tridiagonal_ip.u ≈ sol_sparse_ip.u
                end
            end
        end

        @testset "Different matrix types (conservative, adaptive)" begin
            prod_1! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i, i + 1] = i * u[i]
                end
                return nothing
            end

            prod_2! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i + 1, i] = i * u[i + 1]
                end
                return nothing
            end

            prod_3! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i, i + 1] = i * u[i]
                    P[i + 1, i] = i * u[i + 1]
                end
                return nothing
            end

            n = 4
            P_tridiagonal = Tridiagonal([0.1, 0.2, 0.3],
                                        zeros(n),
                                        [0.4, 0.5, 0.6])
            P_dense = Matrix(P_tridiagonal)
            P_sparse = sparse(P_tridiagonal)
            u0 = [1.0, 1.5, 2.0, 2.5]
            tspan = (0.0, 1.0)
            dt = 0.25

            rtol = sqrt(eps(Float32))

            @testset "$alg" for alg in (MPE(),
                                        MPRK22(0.5), MPRK22(1.0),
                                        MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                                        MPRK43II(2.0 / 3.0), MPRK43II(0.5),
                                        SSPMPRK22(0.5, 1.0), SSPMPRK43())
                for prod! in (prod_1!, prod_2!, prod_3!)
                    prod = (u, p, t) -> begin
                        P = similar(u, (length(u), length(u)))
                        prod!(P, u, p, t)
                        return P
                    end
                    prob_tridiagonal_ip = ConservativePDSProblem(prod!, u0, tspan;
                                                                 p_prototype = P_tridiagonal)
                    prob_tridiagonal_op = ConservativePDSProblem(prod, u0, tspan;
                                                                 p_prototype = P_tridiagonal)
                    prob_dense_ip = ConservativePDSProblem(prod!, u0, tspan;
                                                           p_prototype = P_dense)
                    prob_dense_op = ConservativePDSProblem(prod, u0, tspan;
                                                           p_prototype = P_dense)
                    prob_sparse_ip = ConservativePDSProblem(prod!, u0, tspan;
                                                            p_prototype = P_sparse)
                    prob_sparse_op = ConservativePDSProblem(prod, u0, tspan;
                                                            p_prototype = P_sparse)

                    sol_tridiagonal_ip = solve(prob_tridiagonal_ip, alg; dt)
                    sol_tridiagonal_op = solve(prob_tridiagonal_op, alg; dt)
                    sol_dense_ip = solve(prob_dense_ip, alg; dt)
                    sol_dense_op = solve(prob_dense_op, alg; dt)
                    sol_sparse_ip = solve(prob_sparse_ip, alg; dt)
                    sol_sparse_op = solve(prob_sparse_op, alg; dt)

                    @test isapprox(sol_tridiagonal_ip.t, sol_tridiagonal_op.t; rtol)
                    @test isapprox(sol_dense_ip.t, sol_dense_op.t; rtol)
                    @test isapprox(sol_sparse_ip.t, sol_sparse_op.t; rtol)
                    @test isapprox(sol_tridiagonal_ip.t, sol_dense_ip.t; rtol)
                    @test isapprox(sol_tridiagonal_ip.t, sol_sparse_ip.t; rtol)

                    @test isapprox(sol_tridiagonal_ip.u, sol_tridiagonal_op.u; rtol)
                    @test isapprox(sol_dense_ip.u, sol_dense_op.u; rtol)
                    @test isapprox(sol_sparse_ip.u, sol_sparse_op.u; rtol)
                    @test isapprox(sol_tridiagonal_ip.u, sol_dense_ip.u; rtol)
                    @test isapprox(sol_tridiagonal_ip.u, sol_sparse_ip.u; rtol)
                end
            end
        end

        # Here we check that in-place and out-of-place implementations
        # deliver the same results
        @testset "Different matrix types (nonconservative)" begin
            prod_1! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i, i + 1] = i * u[i]
                end
                for i in 1:length(u)
                    P[i, i] = i * u[i]
                end
                return nothing
            end
            dest_1! = (D, u, p, t) -> begin
                fill!(D, zero(eltype(D)))
                for i in 1:length(u)
                    D[i] = (i + 1) * u[i]
                end
                return nothing
            end

            prod_2! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i + 1, i] = i * u[i + 1]
                end
                for i in 1:length(u)
                    P[i, i] = (i - 1) * u[i]
                end
                return nothing
            end
            dest_2! = (D, u, p, t) -> begin
                fill!(D, zero(eltype(D)))
                for i in 1:length(u)
                    D[i] = i * u[i]
                end
                return nothing
            end

            prod_3! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i, i + 1] = i * u[i]
                    P[i + 1, i] = i * u[i + 1]
                end
                for i in 1:length(u)
                    P[i, i] = (i + 1) * u[i]
                end
                return nothing
            end
            dest_3! = (D, u, p, t) -> begin
                fill!(D, zero(eltype(D)))
                for i in 1:length(u)
                    D[i] = (i - 1) * u[i]
                end
                return nothing
            end

            n = 4
            P_tridiagonal = Tridiagonal([0.1, 0.2, 0.3],
                                        zeros(n),
                                        [0.4, 0.5, 0.6])
            P_dense = Matrix(P_tridiagonal)
            P_sparse = sparse(P_tridiagonal)
            u0 = [1.0, 1.5, 2.0, 2.5]
            D = u0
            tspan = (0.0, 1.0)
            dt = 0.25

            @testset "$alg" for alg in (MPE(),
                                        MPRK22(0.5), MPRK22(1.0),
                                        MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                                        MPRK43II(2.0 / 3.0), MPRK43II(0.5),
                                        SSPMPRK22(0.5, 1.0), SSPMPRK43())
                for (prod!, dest!) in zip((prod_1!, prod_2!, prod_3!),
                                          (dest_1!, dest_2!, dest_3!))
                    prod = (u, p, t) -> begin
                        P = similar(u, (length(u), length(u)))
                        prod!(P, u, p, t)
                        return P
                    end
                    dest = (u, p, t) -> begin
                        D = similar(u)
                        dest!(D, u, p, t)
                        return D
                    end
                    prob_tridiagonal_ip = PDSProblem(prod!, dest!, u0, tspan;
                                                     p_prototype = P_tridiagonal)
                    prob_tridiagonal_op = PDSProblem(prod, dest, u0, tspan;
                                                     p_prototype = P_tridiagonal)
                    prob_dense_ip = PDSProblem(prod!, dest!, u0, tspan;
                                               p_prototype = P_dense)
                    prob_dense_op = PDSProblem(prod, dest, u0, tspan;
                                               p_prototype = P_dense)
                    prob_sparse_ip = PDSProblem(prod!, dest!, u0, tspan;
                                                p_prototype = P_sparse)
                    prob_sparse_op = PDSProblem(prod, dest, u0, tspan;
                                                p_prototype = P_sparse)

                    sol_tridiagonal_ip = solve(prob_tridiagonal_ip, alg;
                                               dt, adaptive = false)
                    sol_tridiagonal_op = solve(prob_tridiagonal_op, alg;
                                               dt, adaptive = false)
                    sol_dense_ip = solve(prob_dense_ip, alg;
                                         dt, adaptive = false)
                    sol_dense_op = solve(prob_dense_op, alg;
                                         dt, adaptive = false)
                    sol_sparse_ip = solve(prob_sparse_ip, alg;
                                          dt, adaptive = false)
                    sol_sparse_op = solve(prob_sparse_op, alg;
                                          dt, adaptive = false)

                    @test sol_tridiagonal_ip.t ≈ sol_tridiagonal_op.t
                    @test sol_dense_ip.t ≈ sol_dense_op.t
                    @test sol_sparse_ip.t ≈ sol_sparse_op.t
                    @test sol_tridiagonal_ip.t ≈ sol_dense_ip.t
                    @test sol_tridiagonal_ip.t ≈ sol_sparse_ip.t

                    @test sol_tridiagonal_ip.u ≈ sol_tridiagonal_op.u
                    @test sol_dense_ip.u ≈ sol_dense_op.u
                    @test sol_sparse_ip.u ≈ sol_sparse_op.u
                    @test sol_tridiagonal_ip.u ≈ sol_dense_ip.u
                    @test sol_tridiagonal_ip.u ≈ sol_sparse_ip.u
                end
            end
        end

        @testset "Different matrix types (nonconservative, adaptive)" begin
            prod_1! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i, i + 1] = i * u[i]
                end
                for i in 1:length(u)
                    P[i, i] = i * u[i]
                end
                return nothing
            end
            dest_1! = (D, u, p, t) -> begin
                fill!(D, zero(eltype(D)))
                for i in 1:length(u)
                    D[i] = (i + 1) * u[i]
                end
                return nothing
            end

            prod_2! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i + 1, i] = i * u[i + 1]
                end
                for i in 1:length(u)
                    P[i, i] = (i - 1) * u[i]
                end
                return nothing
            end
            dest_2! = (D, u, p, t) -> begin
                fill!(D, zero(eltype(D)))
                for i in 1:length(u)
                    D[i] = i * u[i]
                end
                return nothing
            end

            prod_3! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                for i in 1:(length(u) - 1)
                    P[i, i + 1] = i * u[i]
                    P[i + 1, i] = i * u[i + 1]
                end
                for i in 1:length(u)
                    P[i, i] = (i + 1) * u[i]
                end
                return nothing
            end
            dest_3! = (D, u, p, t) -> begin
                fill!(D, zero(eltype(D)))
                for i in 1:length(u)
                    D[i] = (i - 1) * u[i]
                end
                return nothing
            end

            n = 4
            P_tridiagonal = Tridiagonal([0.1, 0.2, 0.3],
                                        zeros(n),
                                        [0.4, 0.5, 0.6])
            P_dense = Matrix(P_tridiagonal)
            P_sparse = sparse(P_tridiagonal)
            u0 = [1.0, 1.5, 2.0, 2.5]
            D = u0
            tspan = (0.0, 1.0)
            dt = 0.25

            rtol = sqrt(eps(Float32))
            @testset "$alg" for alg in (MPE(),
                                        MPRK22(0.5), MPRK22(1.0),
                                        MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                                        MPRK43II(2.0 / 3.0), MPRK43II(0.5),
                                        SSPMPRK22(0.5, 1.0), SSPMPRK43())
                for (prod!, dest!) in zip((prod_1!, prod_2!, prod_3!),
                                          (dest_1!, dest_2!, dest_3!))
                    prod! = prod_3!
                    dest! = dest_3!
                    prod = (u, p, t) -> begin
                        P = similar(u, (length(u), length(u)))
                        prod!(P, u, p, t)
                        return P
                    end
                    dest = (u, p, t) -> begin
                        D = similar(u)
                        dest!(D, u, p, t)
                        return D
                    end
                    prob_tridiagonal_ip = PDSProblem(prod!, dest!, u0, tspan;
                                                     p_prototype = P_tridiagonal)
                    prob_tridiagonal_op = PDSProblem(prod, dest, u0, tspan;
                                                     p_prototype = P_tridiagonal)
                    prob_dense_ip = PDSProblem(prod!, dest!, u0, tspan;
                                               p_prototype = P_dense)
                    prob_dense_op = PDSProblem(prod, dest, u0, tspan;
                                               p_prototype = P_dense)
                    prob_sparse_ip = PDSProblem(prod!, dest!, u0, tspan;
                                                p_prototype = P_sparse)
                    prob_sparse_op = PDSProblem(prod, dest, u0, tspan;
                                                p_prototype = P_sparse)

                    sol_tridiagonal_ip = solve(prob_tridiagonal_ip, alg;
                                               dt)
                    sol_tridiagonal_op = solve(prob_tridiagonal_op, alg;
                                               dt)
                    sol_dense_ip = solve(prob_dense_ip, alg;
                                         dt)
                    sol_dense_op = solve(prob_dense_op, alg;
                                         dt)
                    sol_sparse_ip = solve(prob_sparse_ip, alg;
                                          dt)
                    sol_sparse_op = solve(prob_sparse_op, alg;
                                          dt)

                    @test isapprox(sol_tridiagonal_ip.t, sol_tridiagonal_op.t; rtol)
                    @test isapprox(sol_dense_ip.t, sol_dense_op.t; rtol)
                    @test isapprox(sol_sparse_ip.t, sol_sparse_op.t; rtol)
                    @test isapprox(sol_tridiagonal_ip.t, sol_dense_ip.t; rtol)
                    @test isapprox(sol_tridiagonal_ip.t, sol_sparse_ip.t; rtol)

                    @test isapprox(sol_tridiagonal_ip.u, sol_tridiagonal_op.u; rtol)
                    @test isapprox(sol_dense_ip.u, sol_dense_op.u; rtol)
                    @test isapprox(sol_sparse_ip.u, sol_sparse_op.u; rtol)
                    @test isapprox(sol_tridiagonal_ip.u, sol_dense_ip.u; rtol)
                    @test isapprox(sol_tridiagonal_ip.u, sol_sparse_ip.u; rtol)
                end
            end
        end

        # Here we check the convergence order of pth-order schemes for which
        # also an interpolation of order p is available
        @testset "Convergence tests (conservative)" begin
            algs = (MPE(), MPRK22(0.5), MPRK22(1.0), MPRK22(2.0), SSPMPRK22(0.5, 1.0))
            dts = 0.5 .^ (4:15)
            problems = (prob_pds_linmod, prob_pds_linmod_array,
                        prob_pds_linmod_mvector, prob_pds_linmod_inplace)

            @testset "$alg" for alg in algs
                alg = MPRK22(1.0)
                for prob in problems
                    prob = problems[1]
                    orders = experimental_orders_of_convergence(prob, alg, dts)
                    @test check_order(orders, PositiveIntegrators.alg_order(alg))

                    test_times = [
                        0.123456789, 1 / pi, exp(-1),
                        1.23456789, 1 + 1 / pi, 1 + exp(-1),
                    ]
                    for test_time in test_times
                        orders = experimental_orders_of_convergence(prob, alg,
                                                                    dts;
                                                                    test_time)
                        @test check_order(orders, PositiveIntegrators.alg_order(alg),
                                          atol = 0.2)
                        orders = experimental_orders_of_convergence(prob, alg,
                                                                    dts;
                                                                    test_time,
                                                                    only_first_index = true)
                        @test check_order(orders, PositiveIntegrators.alg_order(alg),
                                          atol = 0.2)
                    end
                end
            end
        end

        # Here we check the convergence order of pth-order schemes for which
        # also an interpolation of order p is available
        @testset "Convergence tests (nonconservative)" begin
            algs = (MPE(), MPRK22(0.5), MPRK22(1.0), MPRK22(2.0), SSPMPRK22(0.5, 1.0))
            dts = 0.5 .^ (4:15)
            problems = (prob_pds_linmod_nonconservative,
                        prob_pds_linmod_nonconservative_inplace)
            @testset "$alg" for alg in algs
                alg = MPRK22(1.0)
                for prob in problems
                    orders = experimental_orders_of_convergence(prob, alg, dts)
                    @test check_order(orders, PositiveIntegrators.alg_order(alg))

                    test_times = [
                        0.123456789, 1 / pi, exp(-1),
                        1.23456789, 1 + 1 / pi, 1 + exp(-1),
                    ]
                    for test_time in test_times
                        orders = experimental_orders_of_convergence(prob, alg,
                                                                    dts;
                                                                    test_time)
                        @test check_order(orders, PositiveIntegrators.alg_order(alg),
                                          atol = 0.2)
                        orders = experimental_orders_of_convergence(prob, alg,
                                                                    dts;
                                                                    test_time,
                                                                    only_first_index = true)
                        @test check_order(orders, PositiveIntegrators.alg_order(alg),
                                          atol = 0.2)
                    end
                end
            end
        end

        # Here we check the convergence order of pth-order schemes for which
        # no interpolation of order p is available
        @testset "Convergence tests (conservative)" begin
            dts = 0.5 .^ (4:12)
            problems = (prob_pds_linmod, prob_pds_linmod_array,
                        prob_pds_linmod_mvector, prob_pds_linmod_inplace)
            algs = (MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                    MPRK43II(0.5), MPRK43II(2.0 / 3.0), SSPMPRK43())
            for alg in algs, prob in problems
                orders = experimental_orders_of_convergence(prob, alg, dts)
                @test check_order(orders, PositiveIntegrators.alg_order(alg), atol = 0.2)
            end
        end

        # Here we check the convergence order of pth-order schemes for which
        # no interpolation of order p is available
        @testset "Convergence tests (nonconservative)" begin
            dts = 0.5 .^ (4:12)
            problems = (prob_pds_linmod_nonconservative,
                        prob_pds_linmod_nonconservative_inplace)
            algs = (MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                    MPRK43II(0.5), MPRK43II(2.0 / 3.0), SSPMPRK43())
            for alg in algs, prob in problems
                orders = experimental_orders_of_convergence(prob, alg, dts)
                @test check_order(orders, PositiveIntegrators.alg_order(alg), atol = 0.2)
            end
        end

        @testset "Interpolation tests (conservative)" begin
            algs = (MPE(), MPRK22(0.5), MPRK22(1.0), MPRK22(2.0), MPRK43I(1.0, 0.5),
                    MPRK43I(0.5, 0.75), MPRK43II(0.5), MPRK43II(2.0 / 3.0),
                    SSPMPRK22(0.5, 1.0), SSPMPRK43())
            dt = 0.5^6
            problems = (prob_pds_linmod, prob_pds_linmod_array,
                        prob_pds_linmod_mvector, prob_pds_linmod_inplace)
            for alg in algs
                for prob in problems
                    sol = solve(prob, alg; dt, adaptive = false)
                    # check derivative of interpolation
                    @test_nowarn sol(0.5, Val{1})
                    @test_nowarn sol(0.5, Val{1}; idxs = 1)
                end
            end
        end

        @testset "Check convergence order (nonautonomous conservative PDS)" begin
            prod! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                P[1, 2] = sin(t)^2 * u[2]
                P[2, 1] = cos(2 * t)^2 * u[1]
                return nothing
            end
            prod = (u, p, t) -> begin
                P = similar(u, (length(u), length(u)))
                prod!(P, u, p, t)
                return P
            end
            u0 = [1.0; 0.0]
            tspan = (0.0, 1.0)
            prob_oop = ConservativePDSProblem(prod, u0, tspan) #out-of-place
            prob_ip = ConservativePDSProblem(prod!, u0, tspan) #in-place

            dts = 0.5 .^ (4:15)
            algs = (MPE(), MPRK22(0.5), MPRK22(1.0), MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                    MPRK43II(2.0 / 3.0), MPRK43II(0.5), SSPMPRK22(0.5, 1.0), SSPMPRK43())
            @testset "$alg" for alg in algs
                orders = experimental_orders_of_convergence(prob_oop, alg, dts)
                @test check_order(orders, PositiveIntegrators.alg_order(alg), atol = 0.2)
                orders = experimental_orders_of_convergence(prob_ip, alg, dts)
                @test check_order(orders, PositiveIntegrators.alg_order(alg), atol = 0.2)
            end
        end

        @testset "Check convergence order (nonautonomous nonconservative PDS)" begin
            prod! = (P, u, p, t) -> begin
                fill!(P, zero(eltype(P)))
                P[1, 2] = sin(t)^2 * u[2]
                P[2, 1] = cos(2 * t)^2 * u[1]
                P[2, 2] = cos(t)^2 * u[2]
                return nothing
            end
            dest! = (d, u, p, t) -> begin
                fill!(d, zero(eltype(d)))
                d[1] = sin(2 * t)^2 * u[1]
                d[2] = sin(0.5 * t)^2 * u[2]
                return nothing
            end
            prod = (u, p, t) -> begin
                P = similar(u, (length(u), length(u)))
                prod!(P, u, p, t)
                return P
            end
            dest = (u, p, t) -> begin
                d = similar(u, (length(u),))
                dest!(d, u, p, t)
                return d
            end
            u0 = [1.0; 0.0]
            tspan = (0.0, 1.0)
            prob_oop = PDSProblem(prod, dest, u0, tspan) #out-of-place
            prob_ip = PDSProblem(prod!, dest!, u0, tspan) #in-place

            dts = 0.5 .^ (4:15)
            algs = (MPE(), MPRK22(0.5), MPRK22(1.0), MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                    MPRK43II(2.0 / 3.0), MPRK43II(0.5), SSPMPRK22(0.5, 1.0), SSPMPRK43())
            @testset "$alg" for alg in algs
                orders = experimental_orders_of_convergence(prob_oop, alg, dts)
                @test check_order(orders, PositiveIntegrators.alg_order(alg), atol = 0.2)
                orders = experimental_orders_of_convergence(prob_ip, alg, dts)
                @test check_order(orders, PositiveIntegrators.alg_order(alg), atol = 0.2)
            end
        end

        @testset "Interpolation tests (nonconservative)" begin
            algs = (MPE(), MPRK22(0.5), MPRK22(1.0), MPRK22(2.0), MPRK43I(1.0, 0.5),
                    MPRK43I(0.5, 0.75), MPRK43II(0.5), MPRK43II(2.0 / 3.0),
                    SSPMPRK22(0.5, 1.0), SSPMPRK43())
            dt = 0.5^6
            problems = (prob_pds_linmod_nonconservative,
                        prob_pds_linmod_nonconservative_inplace)
            for alg in algs
                for prob in problems
                    sol = solve(prob, alg; dt, adaptive = false)
                    # check derivative of interpolation
                    @test_nowarn sol(0.5, Val{1})
                    @test_nowarn sol(0.5, Val{1}; idxs = 1)
                end
            end
        end

        # Check that the schemes accept zero initial values
        @testset "Zero initial values" begin
            # Do a single step and check that no NaNs occur
            u0 = [1.0, 0.0]
            dt = 1.0
            tspan = (0.0, dt)
            p = 1000.0
            function prod!(P, u, p, t)
                λ = p
                fill!(P, zero(eltype(P)))
                P[2, 1] = λ * u[1]
            end
            function dest!(D, u, p, t)
                fill!(D, zero(eltype(D)))
            end
            function prod(u, p, t)
                P = similar(u, (length(u), length(u)))
                prod!(P, u, p, t)
                return P
            end
            function dest(u, p, t)
                d = similar(u)
                dest!(d, u, p, t)
                return d
            end

            prob_ip = ConservativePDSProblem(prod!, u0, tspan, p)
            prob_ip_2 = PDSProblem(prod!, dest!, u0, tspan, p)
            prob_oop = ConservativePDSProblem(prod, u0, tspan, p)
            prob_oop_2 = PDSProblem(prod, dest, u0, tspan, p)

            algs = (MPE(), MPRK22(0.5), MPRK22(1.0), MPRK22(2.0),
                    MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75), MPRK43II(0.5),
                    MPRK43II(2.0 / 3.0), SSPMPRK22(0.5, 1.0), SSPMPRK43())

            for alg in algs
                sol = solve(prob_ip, alg; dt = dt, adaptive = false)
                @test !any(isnan, sol.u[end])
                sol = solve(prob_ip_2, alg; dt = dt, adaptive = false)
                @test !any(isnan, sol.u[end])
                sol = solve(prob_oop, alg; dt = dt, adaptive = false)
                @test !any(isnan, sol.u[end])
                sol = solve(prob_oop_2, alg; dt = dt, adaptive = false)
                @test !any(isnan, sol.u[end])
            end
        end

        # Check that approximations, and thus the Patankar weights,
        # remain positive to avoid division by zero.
        @testset "Positvity check" begin
            # For this problem u[1] decreases montonically to 0 very fast.
            # We perform 10^5 steps and check that u[end] does not contain any NaNs
            u0 = [0.9, 0.1]
            tspan = (0.0, 100.0)
            p = 1000.0
            function prod!(P, u, p, t)
                λ = p
                fill!(P, zero(eltype(P)))
                P[2, 1] = λ * u[1]
            end
            function dest!(D, u, p, t)
                fill!(D, zero(eltype(D)))
            end
            function prod(u, p, t)
                P = similar(u, (length(u), length(u)))
                prod!(P, u, p, t)
                return P
            end
            function dest(u, p, t)
                d = similar(u)
                dest!(d, u, p, t)
                return d
            end

            prob_ip = ConservativePDSProblem(prod!, u0, tspan, p)
            prob_ip_2 = PDSProblem(prod!, dest!, u0, tspan, p)
            prob_oop = ConservativePDSProblem(prod, u0, tspan, p)
            prob_oop_2 = PDSProblem(prod, dest, u0, tspan, p)

            algs = (MPE(), MPRK22(0.5), MPRK22(1.0), MPRK22(2.0),
                    MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75), MPRK43II(0.5),
                    MPRK43II(2.0 / 3.0), SSPMPRK22(0.5, 1.0), SSPMPRK43())

            dt = 1e-3
            for alg in algs
                sol1 = solve(prob_ip, alg; dt = dt, adaptive = false)
                @test !any(isnan, sol1.u[end])
                sol2 = solve(prob_ip_2, alg; dt = dt, adaptive = false)
                @test !any(isnan, sol2.u[end])
                sol3 = solve(prob_oop, alg; dt = dt, adaptive = false)
                @test !any(isnan, sol3.u[end])
                sol4 = solve(prob_oop_2, alg; dt = dt, adaptive = false)
                @test !any(isnan, sol4.u[end])
                @test sol1.u ≈ sol2.u ≈ sol3.u ≈ sol4.u
            end
        end
    end

    # Here we check that the implemented schemes can solve the predefined PDS
    # (at least for specific parameters)
    @testset "PDS problem library (adaptive schemes)" begin
        algs = (MPRK22(0.5), MPRK22(1.0), MPRK43I(1.0, 0.5), MPRK43I(0.5, 0.75),
                MPRK43II(2.0 / 3.0), MPRK43II(0.5), SSPMPRK22(0.5, 1.0))
        probs = (prob_pds_linmod, prob_pds_linmod_inplace, prob_pds_nonlinmod,
                 prob_pds_robertson, prob_pds_bertolazzi, prob_pds_brusselator,
                 prob_pds_npzd,
                 prob_pds_sir, prob_pds_stratreac)
        @testset "$alg" for alg in algs
            @testset "$i" for (i, prob) in enumerate(probs)
                if prob == prob_pds_stratreac && alg == SSPMPRK22(0.5, 1.0)
                    #TODO: SSPMPRK22(0.5, 1.0) is unstable for prob_pds_stratreac. 
                    #Need to figure out if this is a problem of the algorithm or not.
                    break
                elseif prob == prob_pds_stratreac && alg == MPRK43I(0.5, 0.75)
                    # Not successful on Julia 1.9
                    break
                end
                sol = solve(prob, alg)
                @test Int(sol.retcode) == 1
            end
        end
    end

    # Here we check that the implemented schemes can solve the predefined PDS.
    @testset "PDS problem library (non-adaptive schemes)" begin
        algs = (MPE(), SSPMPRK43())
        #prob_pds_robertson not included
        probs = (prob_pds_linmod, prob_pds_linmod_inplace, prob_pds_nonlinmod,
                 prob_pds_bertolazzi, prob_pds_brusselator,
                 prob_pds_npzd, prob_pds_sir, prob_pds_stratreac)
        @testset "$alg" for alg in algs
            @testset "$prob" for prob in probs
                tspan = prob.tspan
                dt = (tspan[2] - tspan[1]) / 10
                sol = solve(prob, alg; dt = dt)
                @test Int(sol.retcode) == 1
            end
        end
    end

    #=
    # TODO: Do we want to keep the examples and test them or do we want
    #       to switch to real docs/tutorials instead?
    @testset "Examples" begin
        # dummy test to make sure the testset errors if the process
        # errors out
        @test true

        if VERSION >= v"1.10"
            cmd = Base.julia_cmd()
            examples_dir = abspath(joinpath(pkgdir(PositiveIntegrators), "examples"))
            examples = ["01_example_proddest.jl",
                "02_example_mpe.jl",
                "03_example_mprk22.jl",
                "04_example_problemlibrary.jl"]

            @testset "Example $ex" for ex in examples
                @info "Testing examples" ex
                example = joinpath(examples_dir, ex)
                @test isfile(example)
                @time run(`$cmd --project=$(examples_dir) $(example)`)
            end
        end
    end
    =#
end;
