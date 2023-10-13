using Test
using PositiveIntegrators

@testset "PositiveIntegrators.jl tests" begin


    # TODO: Do we want to keep the examples and test them or do we want
    #       to switch to real docs/tutorials instead?
    @testset "Examples" begin
        # dummy test to make sure the testset errors if the process
        # errors out
        @test true

        cmd = Base.julia_cmd()
        examples_dir = abspath(pkgdir(PositiveIntegrators, "examples"))

        run(`$cmd --project=$(examples_dir) $(joinpath(examples_dir, "01_example_proddest.jl"))`)

        run(`$cmd --project=$(examples_dir) $(joinpath(examples_dir, "02_example_mpe.jl"))`)

        run(`$cmd --project=$(examples_dir) $(joinpath(examples_dir, "03_example_mprk22.jl"))`)
    end

    @test true
end
