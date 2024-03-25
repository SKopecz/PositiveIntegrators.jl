# PositiveIntegrators.jl

<!-- [![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SKopecz.github.io/PositiveIntegrators.jl/stable) -->
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SKopecz.github.io/PositiveIntegrators.jl/dev)
[![Build Status](https://github.com/SKopecz/PositiveIntegrators.jl/workflows/CI/badge.svg)](https://github.com/SKopecz/PositiveIntegrators.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/SKopecz/PositiveIntegrators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SKopecz/PositiveIntegrators.jl)
[![Coveralls](https://coveralls.io/repos/github/SKopecz/PositiveIntegrators.jl/badge.svg?branch=main)](https://coveralls.io/github/SKopecz/PositiveIntegrators.jl?branch=main)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10868393.svg)](https://doi.org/10.5281/zenodo.10868393)

Over the last two decades several approaches have been suggested to numerically
preserve the positivity of positive ordinary differential equation (ODE) systems.
This [Julia](https://julialang.org) package provides efficient implementations
of various positive time integration schemes, allowing a fair comparison of the
different schemes. The package extends [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
by
* adding a new problem type for production-destruction systems
* adding the algorithms of first and second order modified Patankar-Runge-Kutta (MPRK) schemes


## Referencing

If you use
[PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl)
for your research, please cite it using the bibtex entry
```bibtex
@misc{PositiveIntegrators.jl,
  title={{PositiveIntegrators.jl}: {A} {J}ulia library of positivity-preserving
         time integration methods},
  author={Kopecz, Stefan and Ranocha, Hendrik and contributors},
  year={2023},
  doi={10.5281/zenodo.10868393},
  url={https://github.com/SKopecz/PositiveIntegrators.jl}
}
```


## License and contributing

This project is licensed under the MIT license (see [License](@ref)).
Since it is an open-source project, we are very happy to accept contributions
from the community. Please refer to the section [Contributing](@ref) for more
details.
