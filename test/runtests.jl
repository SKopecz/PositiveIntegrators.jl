using Test
using PositiveIntegrators

using Aqua: Aqua

@testset "PositiveIntegrators.jl tests" begin
    @testset "Aqua.jl" begin
        Aqua.test_all(PositiveIntegrators;
            ambiguities = false, # a lot of false positives from dependencies
        )
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
