# PositiveIntegrators.jl

The [Julia](https://julialang.org) library
[PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl)
provides several time integration methods developed to preserve the positivity
of numerical solutions.


## Installation

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

Modified Patankar-Runge-Kutta (MPRK) schemes are unconditionally positive and conservative time integration schemes for the solution of positive and conservative ODE systems. The application of these methods is based on the representation of the ODE system as a so-called production-destruction system (PDS).

#### Production-destruction systems (PDS)

The application of MPRK schemes requires the ODE system to be represented as a production-destruction system (PDS). A PDS takes the general form
```math
    u_i'(t) = \sum_{j=1}^N \bigl(p_{ij}(t,\boldsymbol u) - d_{ij}(t,\boldsymbol u)\bigr),\quad i=1,\dots,N,
```
where ``\boldsymbol u=(u_1,\dots,u_n)^T`` is the vector of unknowns and both production terms ``p_{ij}(t,\boldsymbol u)`` and destruction terms ``d_{ij}(t,\boldsymbol u)`` must be nonnegative for all ``i,j=1,\dots,N``. The meaning behind ``p_{ij}`` and ``d_{ij}`` is as follows:
* ``p_{ij}`` with ``i\ne j`` represents the sum of all nonnegative terms which
  appear in equation ``i`` with a positive sign and in equation ``j`` with a negative sign.
* ``d_{ij}`` with ``i\ne j`` represents the sum of all nonnegative terms which
  appear in equation ``i`` with a negative sign and in equation ``j`` with a positive sign.
* ``p_{ii}`` represents the sum of all nonnegative terms  which appear in
  equation ``i`` and don't have a negative counterpart in one of the other equations.
* ``d_{ii}`` represents the sum of all negative terms which appear in
  equation ``i`` and don't have a positive counterpart in one of the other equations.

This naming convention leads to ``p_{ij} = d_{ji}`` for ``i≠ j`` and therefore a PDS is completely defined by the production matrix ``\mathbf{P}=(p_{ij})_{i,j=1,\dots,N}`` and the destruction vector ``\mathbf{d}=(d_{ii})_{i=1,\dots,N}``.

As an example we consider the Lotka-Volterra model
```math
\begin{aligned}
u_1' &= 2u_1-u_1u_2,\\
u_2' &= u_1u_2-u_2,
\end{aligned}
```
which always has positive solutions if positive initial values are supplied.
Assuming ``u_1,u_2>0``, the above naming scheme results in
```math
\begin{aligned}
p_{11}(u_1,u_2) &= 2u_1,\\
p_{21}(u_1,u_2) &= u_1u_2 = d_{12}(u_1,u_2) ,\\
d_{22}(u_1,u_2) &= u_2,
\end{aligned}
```
where all remaining production and destruction terms are zero.
Consequently the production matrix ``\mathbf P`` and destruction vector ``\mathbf d`` are
```math
\mathbf P(u_1,u_2) = \begin{pmatrix}2u_1 & 0\\ u_1u_2 & 0\end{pmatrix},\quad \mathbf d(u_1,u_2) = \begin{pmatrix}0\\ u_2\end{pmatrix}.
```

```@setup LotkaVolterra
import Pkg; Pkg.add("OrdinaryDiffEq");  Pkg.add("Plots")
```
To solve this PDS together with initial values ``u_1(0)=u_2(0)=2`` on the time domain ``(0,10)``, we first need to create a `PDSProblem`.
```@example LotkaVolterra
using PositiveIntegrators # load PDSProblem

P(u, p, t) = [2*u[1]  0.0; u[1]*u[2]  0.0] # Production matrix
d(u, p, t) = [0.0; u[2]] # Destruction vector

u0 = [2.0; 2.0] # initial values
tspan = (0.0, 10.0) # time span

# Create PDS
prob = PDSProblem(P, d, u0, tspan)
nothing #hide
```
Now that the problem has been created, we can solve it with any of the methods of [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). Here we use the method `Tsit5()`. Please note that [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl) currently only provides methods for positive and conservative PDS, see below.

```@example LotkaVolterra
using OrdinaryDiffEq  #load Tsit5

sol = solve(prob, Tsit5())
nothing # hide
```
Finally, we can use [Plots.jl](https://docs.juliaplots.org/stable/) to visualize the solution.
```@example LotkaVolterra
using Plots

plot(sol)
```

#### Conservative production-destruction systems

A PDS with the additional property
```math
  p_{ii}(t,\boldsymbol y)=d_{ii}(t,\boldsymbol y)=0
```
for ``i=1,\dots,N`` is called conservative. In this case we have
``p_{ij}=d_{ji}`` for all ``i,j=1,\dots,N``, which leads to
```math
\frac{d}{dt}\sum_{i=1}^N y_i=\sum_{i=1}^N y_i' = \sum_{\mathclap{i,j=1}}^N \bigl(p_{ij}(t,\boldsymbol y) - d_{ij}(t,\boldsymbol y)\bigr)= \sum_{\mathclap{i,j=1}}^N \bigl(p_{ij}(t,\boldsymbol y) - p_{ji}(t,\boldsymbol y)\bigr) = 0.
```
This shows that the sum of the state variables of a conservative PDS remains constant over time, i.e.
```math
\sum_{i=1}^N y_i(t) = \sum_{i=1}^N y_i(0)
```
for all times ``t>0``.
Moreover, a conservative PDS is completely defined by the square matrix ``\mathbf P=(p_{ij})_{i,j=1,\dots,N}``. There is no need to store an additional vector of destruction terms since ``d_{ij} = p_{ji}`` for all ``i,j=1,\dots,N``.

One specific example of a conservative PDS is the SIR model
```math
S' = -\frac{β S I}{N},\quad I'= \frac{β S I}{N} - γ I,\quad R'=γ I,
```
with ``N=S+I+R`` and ``\beta,\gamma>0``. Assuming ``S,I,R>0`` the production and destruction terms are given by
```math
p_{21}(S,I,R) = d_{12}(S,I,R) = \frac{β S I}{N},\quad p_{32}(S,I,R) = d_{23}(S,I,R) = γ I,
```
where the remaining production and destruction terms are zero.
The corresponding production matrix ``\mathbf P`` is
```math
\mathbf P(S,I,R) = \begin{pmatrix}0 & 0 & 0\\ \frac{β S I}{N} & 0 & 0\\ 0 & γ I & 0\end{pmatrix}.
```

The following example shows how to implement the above SIR model with ``\beta=0.4, \gamma=0.04``, initial conditions ``S(0)=997, I(0)=3, R(0)=0`` and time domain ``(0, 100)`` using `ConservativePDSProblem` from [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl).

```@setup SIR
import Pkg; Pkg.add("OrdinaryDiffEq");
```

```@example SIR
using PositiveIntegrators

# Out-of-place implementation of the P matrix for the SIR model
function P(u, p, t)
  S, I, R = u

  β = 0.4
  γ = 0.04
  N = 1000.0

  P = zeros(3,3)
  P[2,1] = β*S*I/N
  P[3,2] = γ*I
  return P
end

u0 = [997.0; 3.0; 0.0]; # initial values
tspan = (0.0, 100.0); # time span

# Create SIR problem
prob = ConservativePDSProblem(P, u0, tspan)
nothing # hide
```
Since the SIR model is not only conservative but also positive, we can use any MPRK scheme from [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl) to solve it. Here we use `MPRK22(1.0)`.
Please note that any method from [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) can be used as well, but might possibly generate negative approximations.

```@example SIR
sol = solve(prob, MPRK22(1.0))
nothing # hide
```
Finally, we can use [Plots.jl](https://docs.juliaplots.org/stable/) to visualize the solution.
```@example SIR
using Plots

plot(sol, legend=:right)
```


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
  doi={10.5281/zenodo.10868393},
  url={https://github.com/SKopecz/PositiveIntegrators.jl}
}
```


## License and contributing

This project is licensed under the MIT license (see [License](@ref)).
Since it is an open-source project, we are very happy to accept contributions
from the community. Please refer to the section [Contributing](@ref) for more
details.
