using Documenter
using Pkg: Pkg
using PositiveIntegrators

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(PositiveIntegrators,
  :DocTestSetup, :(using PositiveIntegrators); recursive=true)

# Copy some files from the top level directory to the docs and modify them
# as necessary
open(joinpath(@__DIR__, "src", "license.md"), "w") do io
  # Point to source license file
  println(io, """
  ```@meta
  EditURL = "https://github.com/SKopecz/PositiveIntegrators.jl/blob/main/LICENSE.md"
  ```
  """)
  # Write the modified contents
  println(io, "# License")
  println(io, "")
  for line in eachline(joinpath(dirname(@__DIR__), "LICENSE"))
    line = replace(line, "[LICENSE.md](LICENSE.md)" => "[License](@ref)")
    println(io, "> ", line)
  end
end

open(joinpath(@__DIR__, "src", "contributing.md"), "w") do io
  # Point to source license file
  println(io, """
  ```@meta
  EditURL = "https://github.com/SKopecz/PositiveIntegrators.jl/blob/main/CONTRIBUTING.md"
  ```
  """)
  # Write the modified contents
  println(io, "# Contributing")
  println(io, "")
  for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
    line = replace(line, "[LICENSE.md](LICENSE.md)" => "[License](@ref)")
    println(io, "> ", line)
  end
end

# Make documentation
makedocs(
  modules = [PositiveIntegrators],
  sitename="PositiveIntegrators.jl",
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    canonical = "https://SKopecz.github.io/PositiveIntegrators.jl/stable"
  ),
  # Explicitly specify documentation structure
  pages = [
    "Home" => "index.md",
    "Introduction" => "introduction.md",
    "Tutorials" => [
      "tutorials/constant_linear_advection.md",
      "tutorials/advection_diffusion.md",
      "tutorials/variable_linear_advection.md",
      "tutorials/wave_equation.md",
      "tutorials/kdv.md",
    ],
    "Automatic differentiation (AD)" => "ad.md",
    "Applications & references" => "applications.md",
    "Benchmarks" => "benchmarks.md",
    "API reference" => "api_reference.md",
    "Contributing" => "contributing.md",
    "License" => "license.md"
  ]
)

deploydocs(
  repo = "github.com/SKopecz/PositiveIntegrators.jl",
  devbranch = "main",
  push_preview = true
)
