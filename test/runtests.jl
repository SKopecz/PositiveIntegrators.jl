using Test
using LinearAlgebra
using SparseArrays
using Statistics: mean

using StaticArrays: MVector

using OrdinaryDiffEq
using PositiveIntegrators

using LinearSolve: RFLUFactorization, LUFactorization

using Aqua: Aqua

"""
    experimental_order_of_convergence(prob, alg, dts, test_times)

Solve `prob` with `alg` and fixed time steps taken from `dts`, and compute
the mean error at the times `test_times`.
Return the associated experimental order of convergence.
"""
function experimental_order_of_convergence(prob, alg, dts, test_times)
    @assert length(dts) > 1
    errors = zeros(eltype(dts), length(dts))
    analytic = t -> prob.f.analytic(prob.u0, prob.p, t)

    for (i, dt) in enumerate(dts)
        sol = solve(prob, alg; dt = dt, adaptive = false)
        errors[i] = mean(test_times) do t
            norm(sol(t) - analytic(t))
        end
    end

    return experimental_order_of_convergence(errors, dts)
end

"""
    experimental_order_of_convergence(prob, alg, dts)

Solve `prob` with `alg` and fixed time steps taken from `dts`, and compute
the mean error at the final time.
Return the associated experimental order of convergence.
"""
function experimental_order_of_convergence(prob, alg, dts)
    @assert length(dts) > 1
    errors = zeros(eltype(dts), length(dts))
    analytic = t -> prob.f.analytic(prob.u0, prob.p, t)

    for (i, dt) in enumerate(dts)
        sol = solve(prob, alg; dt = dt, adaptive = false, save_everystep = false)
        errors[i] = norm(sol.u[end] - analytic(sol.t[end]))
    end

    return experimental_order_of_convergence(errors, dts)
end

"""
    experimental_order_of_convergence(errors, dts)

Compute the experimental order of convergence for given `errors` and
time step sizes `dts`.
"""
function experimental_order_of_convergence(errors, dts)
    Base.require_one_based_indexing(errors, dts)
    @assert length(errors) == length(dts)
    orders = zeros(eltype(errors), length(errors) - 1)

    for i in eachindex(orders)
        orders[i] = log(errors[i] / errors[i + 1]) / log(dts[i] / dts[i + 1])
    end

    return mean(orders)
end

const prob_pds_linmod_array = ConservativePDSProblem(prob_pds_linmod.f,
                                                     Array(prob_pds_linmod.u0),
                                                     prob_pds_linmod.tspan)
const prob_pds_linmod_mvector = ConservativePDSProblem(prob_pds_linmod_inplace.f,
                                                       MVector(prob_pds_linmod.u0),
                                                       prob_pds_linmod.tspan)

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

    @testset "MPE" begin
        @testset "Linear model problem" begin
            # For this linear model problem, the modified Patankar-Euler
            # method is equivalent to the implicit Euler method.

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

            dt = 0.25
            sol_MPE_op = solve(prob_op, MPE(); dt)
            sol_IE_op = solve(prob_op, ImplicitEuler(autodiff = false);
                              dt, adaptive = false)
            @test sol_MPE_op.t ≈ sol_IE_op.t
            @test sol_MPE_op.u ≈ sol_IE_op.u

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

            dt = 0.25
            sol_MPE_ip = solve(prob_ip, MPE(); dt)
            sol_IE_ip = solve(prob_ip, ImplicitEuler(autodiff = false);
                              dt, adaptive = false)
            @test sol_MPE_ip.t ≈ sol_IE_ip.t
            @test sol_MPE_ip.u ≈ sol_IE_ip.u

            # Check different linear solvers
            dt = 0.25
            sol1 = solve(prob_ip, MPE(); dt)
            sol2 = solve(prob_ip, MPE(linsolve = RFLUFactorization()); dt)
            sol3 = solve(prob_ip, MPE(linsolve = LUFactorization()); dt)
            @test sol1.t ≈ sol2.t
            @test sol1.t ≈ sol3.t
            @test sol1.u ≈ sol2.u
            @test sol1.u ≈ sol3.u
        end

        @testset "Different matrix types" begin
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

                sol_tridiagonal_ip = solve(prob_tridiagonal_ip, MPE();
                                           dt, adaptive = false)
                sol_tridiagonal_op = solve(prob_tridiagonal_op, MPE();
                                           dt, adaptive = false)
                sol_dense_ip = solve(prob_dense_ip, MPE();
                                     dt, adaptive = false)
                sol_dense_op = solve(prob_dense_op, MPE();
                                     dt, adaptive = false)
                sol_sparse_ip = solve(prob_sparse_ip, MPE();
                                      dt, adaptive = false)
                sol_sparse_op = solve(prob_sparse_op, MPE();
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

        @testset "Convergence tests" begin
            alg = MPE()
            dts = 0.5 .^ (6:11)
            problems = (prob_pds_linmod, prob_pds_linmod_array,
                        prob_pds_linmod_mvector, prob_pds_linmod_inplace)
            for prob in problems
                eoc = experimental_order_of_convergence(prob, alg, dts)
                @test isapprox(eoc, PositiveIntegrators.alg_order(alg); atol = 0.2)

                test_times = [
                    0.123456789, 1 / pi, exp(-1),
                    1.23456789, 1 + 1 / pi, 1 + exp(-1),
                ]
                eoc = experimental_order_of_convergence(prob, alg, dts, test_times)
                @test isapprox(eoc, PositiveIntegrators.alg_order(alg); atol = 0.2)
            end
        end
    end

    @testset "MPRK22" begin
        @testset "Different matrix types" begin
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

                sol_tridiagonal_ip = solve(prob_tridiagonal_ip, MPRK22(1.0); dt,
                                           adaptive = false)
                sol_tridiagonal_op = solve(prob_tridiagonal_op, MPRK22(1.0); dt,
                                           adaptive = false)
                sol_dense_ip = solve(prob_dense_ip, MPRK22(1.0); dt, adaptive = false)
                sol_dense_op = solve(prob_dense_op, MPRK22(1.0); dt, adaptive = false)
                sol_sparse_ip = solve(prob_sparse_ip, MPRK22(1.0); dt, adaptive = false)
                sol_sparse_op = solve(prob_sparse_op, MPRK22(1.0); dt, adaptive = false)

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

        @testset "Convergence tests" begin
            dts = 0.5 .^ (6:11)
            problems = (prob_pds_linmod, prob_pds_linmod_array,
                        prob_pds_linmod_mvector, prob_pds_linmod_inplace)
            for alpha in (0.5, 1.0, 2.0), prob in problems
                alg = MPRK22(alpha)
                eoc = experimental_order_of_convergence(prob, alg, dts)
                @test isapprox(eoc, PositiveIntegrators.alg_order(alg); atol = 0.2)

                test_times = [
                    0.123456789, 1 / pi, exp(-1),
                    1.23456789, 1 + 1 / pi, 1 + exp(-1),
                ]
                eoc = experimental_order_of_convergence(prob, alg, dts, test_times)
                @test isapprox(eoc, PositiveIntegrators.alg_order(alg); atol = 0.2)
            end
        end
    end

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
end
