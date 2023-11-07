# PositiveIntegrators.jl

The [Julia]() library
[PositiveIntegrators.jl](https://github.com/ranocha/PositiveIntegrators.jl)
provides several time integration methods developed to preserve the positivity
of numerical solutions.

TODO: More introduction etc.


## Installation

TODO: PositiveIntegrators.jl has to be registered - up to now, it is not!

[PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl)
is a registered Julia package. Thus, you can install it from the Julia REPL via
```julia
julia> using Pkg; Pkg.add("PositiveIntegrators")
```

If you want to update PositiveIntegrators.jl, you can use
```julia
julia> using Pkg; Pkg.update("PositiveIntegrators")
```
As usual, if you want to update PositiveIntegrators.jl and all other
packages in your current project, you can execute
```julia
julia> using Pkg; Pkg.update()
```


## Basic examples

### Modified Patankar-Runge-Kutta schemes

Modified Patankar-Runge-Kutta (MPRK) schemes are unconditionally positive and conservative time integration schemes for the solution of positive and conservative ODE systems. The formulation of these methods is based on the representation of the ODE system as a so-called production-destruction system (PDS).

#### Production-destruction systems (PDS)

The formulation of MPRK schemes requires the ODE system to be represented as a production-destruction system (PDS). A PDS takes the general form
```math
    y_i'(t) = \sum_{j=1}^N \bigl(p_{ij}(t,\boldsymbol y) - d_{ij}(t,\boldsymbol y)\bigr),\quad i=1,\dots,N,
```
where ``\boldsymbol y=(y_1,\dots,y_n)^T`` is the vector of unknowns and the production terms ``p_{ij}(t,\boldsymbol y)`` as well as the destruction terms ``d_{ij}(t,\boldsymbol y)`` must be positive for all ``i,j=1,\dots,N``. The meaning behind ``p_{ij}`` and ``d_{ij}`` is as follows:
* ``p_{ij}`` with ``i\ne j`` represents the sum of all positive terms which 
  appear in equation ``i`` with a positive sign and in equation ``j`` with a negative sign.
* ``d_{ij}`` with ``i\ne j`` represents the sum of all positive terms which 
  appear in equation ``i`` with a negative sign and in equation ``j`` with a positive sign.
* ``p_{ii}`` represents the sum of all positive terms  which appear in   
  equation ``i`` and don't have a negative counterpart.
* ``d_{ii}`` represents the sum of all negative terms which appear in   
  equation ``i`` and don't have a positive counterpart.

Please note that the above naming convention leads to ``p_{ij} = d_{ji}`` for ``i≠ j``.

To illustrate the indexing, we consider the ficticious ODE system
```math
\begin{aligned}
y_1' &= y_1y_3^2 + 1-e^{-y_3} - y_1 y_2 - y_1, \\
y_2' &= y_1y_2 + y_2 + y_1^2,\\
y_3'&=-y_1y_3^2-(1-e^{-y_3}).
\end{aligned}
```
Under the assumptions ``y_1,y_2,y_3>0``, the above naming scheme results in
```math
\begin{aligned}
p_{13}(t,\boldsymbol y) &= d_{31}(t,\boldsymbol y) = y_1y_3^2 + 1-e^{-y_3},\\
p_{21}(t,\boldsymbol y) &= d_{21}(t,\boldsymbol y) = y_1 y_2,\\
d_{11}(t,\boldsymbol y) &= y_1,\\
p_{22}(t,\boldsymbol y) &= y_2 + y_1^2,
\end{aligned}
```
where the missing production and destruction terms are set to zero.

One specific example of a PDS are the Lotka–Volterra equations
```math
x' = α x - β x y,\quad
y' = β x y - γ y,
```
with 
```math
p_{11}(x,y) = α x,\quad
p_{21}(x,y) = d_{12}(x, y) = β x y,\quad
d_{22}(x,y) = γ y,
```
where we assume ``x,y>0`` as well as ``\alpha,\beta,\gamma>0``.

In terms of implementation, a PDS is completely described by the square matrix ``(p_{ij})_{i,j=1,\dots,N}`` and the vector ``(d_{ii})_{i=1,\dots,N}``. 

#### Conservative production-destruction systems

A PDS with 
```math
  p_{ii}(t,\boldsymbol y)=d_{ii}(t,\boldsymbol y)=0
``` 
for ``i=1,\dots,N`` is called conservative. In this case we have

```math
\frac{d}{dt}\sum_{i=1}^N y_i=\sum_{i=1}^N y_i' = \sum_{\mathclap{i,j=1,\, i≠ j}}^N \bigl(p_{ij}(t,\boldsymbol y) - d_{ij}(t,\boldsymbol y)\bigr)= \sum_{\mathclap{i,j=1,\, i≠ j}}^N \bigl(p_{ij}(t,\boldsymbol y) - p_{ji}(t,\boldsymbol y)\bigr) = 0.
```

As a consequence the sum of state variables remains constant over time, i.e.
```math 
\sum_{i=1}^N y_i(t) = \sum_{i=1}^N y_i(0) 
```
for all times ``t>0``.

A specific example of a conservative PDS is the SIR model
```math
S' = -\frac{β S I}{N},\quad I'= \frac{β S I}{N} - γ I,\quad R'=γ I,
```
with
```math
p_{21}(S,I,R) = d_{12}(S,I,R) = \frac{β S I}{N},\quad p_{32}(S,I,R) = d_{23}(S,I,R) = γ I
```
and ``S,I,R>0``, ``N=S+I+R`` and ``\beta,\gamma>0``.

In terms of implementation, a conservative PDS is completely described by the square matrix ``(p_{ij})_{i,j=1,\dots,N}``. There is no need for an additional vector to store destruction terms since we have ``d_{ij} = p_{ji}``. 

## Referencing

If you use
[PositiveIntegrators.jl](https://github.com/ranocha/PositiveIntegrators.jl)
for your research, please cite it using the bibtex entry
```bibtex
@misc{PositiveIntegrators.jl,
  title={{PositiveIntegrators.jl}: {A} {J}ulia library of positivity-preserving
         time integration methods},
  author={Kopecz, Stefan and Ranocha, Hendrik and contributors},
  year={2023},
  doi={TODO},
  url={https://github.com/SKopecz/PositiveIntegrators.jl}
}
```


## License and contributing

This project is licensed under the MIT license (see [License](@ref)).
Since it is an open-source project, we are very happy to accept contributions
from the community. Please refer to the section [Contributing](@ref) for more
details.
