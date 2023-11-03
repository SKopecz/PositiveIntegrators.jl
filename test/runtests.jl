using Test
using OrdinaryDiffEq
using PositiveIntegrators

using LinearSolve: RFLUFactorization, LUFactorization

using Aqua: Aqua

@testset "PositiveIntegrators.jl tests" begin
    @testset "Aqua.jl" begin
        # We do not test ambiguities since we get a lot of 
        # false positives from dependencies
        Aqua.test_all(PositiveIntegrators;
                      ambiguities = false,)
    end

    @testset "ProdDestODEProblem" begin
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
            linmod_PDS_op = ProdDestODEProblem(linmodP, linmodD, u0, tspan)

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
            linmod_PDS_ip = ProdDestODEProblem(linmodP!, linmodD!, u0, tspan)

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
            PD_op = ProdDestFunction(linmodP, linmodD;
                                     analytic = f_analytic)
            prob_op = ProdDestODEProblem(PD_op, u0, tspan, p)

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
            PD_ip = ProdDestFunction(linmodP!, linmodD!;
                                     p_prototype = zeros(2, 2),
                                     d_prototype = zeros(2, 1),
                                     analytic = f_analytic)
            prob_ip = ProdDestODEProblem(PD_ip, u0, tspan, p)

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
    end

    # TODO: Do we want to keep the examples and test them or do we want
    #       to switch to real docs/tutorials instead?
    @testset "Examples" begin
        # dummy test to make sure the testset errors if the process
        # errors out
        @test true

        cmd = Base.julia_cmd()
        examples_dir = abspath(joinpath(pkgdir(PositiveIntegrators), "examples"))
        examples = ["01_example_proddest.jl",
            "02_example_mpe.jl",
            "03_example_mprk22.jl"]

        @testset "Example $ex" for ex in examples
            @info "Testing examples" ex
            example = joinpath(examples_dir, ex)
            @test isfile(example)
            @time run(`$cmd --project=$(examples_dir) $(example)`)
        end
    end
end
