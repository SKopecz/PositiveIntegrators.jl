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

Modified Patankar-Runge-Kutta (MPRK) schemes are unconditionally positive and conservative time integration schemes for the solution of positive and conservative ODE systems. The application of these methods is based on the representation of the ODE system as a so-called production-destruction system (PDS).

#### Production-destruction systems (PDS)

The application of MPRK schemes requires the ODE system to be represented as a production-destruction system (PDS). A PDS takes the general form
```math
    u_i'(t) = \sum_{j=1}^N \bigl(p_{ij}(t,\boldsymbol u) - d_{ij}(t,\boldsymbol u)\bigr),\quad i=1,\dots,N,
```
where ``\boldsymbol u=(u_1,\dots,u_n)^T`` is the vector of unknowns and both production terms ``p_{ij}(t,\boldsymbol u)`` and destruction terms ``d_{ij}(t,\boldsymbol u)`` must be positive for all ``i,j=1,\dots,N``. The meaning behind ``p_{ij}`` and ``d_{ij}`` is as follows:
* ``p_{ij}`` with ``i\ne j`` represents the sum of all positive terms which 
  appear in equation ``i`` with a positive sign and in equation ``j`` with a negative sign.
* ``d_{ij}`` with ``i\ne j`` represents the sum of all positive terms which 
  appear in equation ``i`` with a negative sign and in equation ``j`` with a positive sign.
* ``p_{ii}`` represents the sum of all positive terms  which appear in   
  equation ``i`` and don't have a negative counterpart.
* ``d_{ii}`` represents the sum of all negative terms which appear in   
  equation ``i`` and don't have a positive counterpart.

Please note that the above naming convention leads to ``p_{ij} = d_{ji}`` for ``i≠ j``. Hence, a PDS is completely described by the production matrix ``\mathbf{P}=(p_{ij})_{i,j=1,\dots,N}`` and the destruction vector ``\mathbf{d}=(d_{ii})_{i=1,\dots,N}``. 

To illustrate the indexing, we consider the fictitious ODE system
```math
\begin{aligned}
u_1' &= u_1u_3^2 + 1-e^{-y_3} - u_1 u_2 - u_1, \\
u_2' &= u_1u_2 + u_2 + u_1^2,\\
u_3'&=-u_1u_3^2-(1-e^{-u_3}).
\end{aligned}
```
Under the assumptions ``u_1,u_2,u_3>0``, the above naming scheme results in
```math
\begin{aligned}
p_{13}(t,\boldsymbol u) &= d_{31}(t,\boldsymbol u) = u_1u_3^2 + 1-e^{-u_3},\\
p_{21}(t,\boldsymbol u) &= d_{21}(t,\boldsymbol u) = u_1 u_2,\\
d_{11}(t,\boldsymbol u) &= u_1,\\
p_{22}(t,\boldsymbol u) &= u_2 + u_1^2,
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
The corresponding production matrix ``\mathbf{P}`` and destruction vector ``\mathbf{d}`` are
```math
\mathbf{P}(x,y)=\begin{pmatrix}α x & 0\\β x y & 0\end{pmatrix},\quad \mathbf{d}(x,y)=\begin{pmatrix}0 \\γ y\end{pmatrix}.
```
The following example shows how to implement the Lotka-Volterra equations with ``α =2``, ``\beta=\frac{1}{2}`` and ``\gamma=1``.
```@setup LotkaVolterra
import Pkg; Pkg.add("OrdinaryDiffEq"); Pkg.add("Plots")
```

```@example LotkaVolterra
using PositiveIntegrators

# Out-of-place implementation of the matrix P for the Lotka-Volterra equations
function P(u, p, t)
  x, y = u
  α, β, γ = p

  P = zeros(2,2)
  P[1,1] = α*x
  P[2,1] = β*x*y
  return P
end

# Out-of-place implementation of the vector d for the Lotka-Volterra equations
function d(u, p, t)
  x, y = u
  α, β, γ = p

  d = zeros(2,1)
  d[1] = 0
  d[2] = γ*y
  return d
end

u0 = [10.0; 1.0]; # initial values
tspan = (0.0, 20.0); # time span
p = [2.0; 0.5; 1.0]; # α, β, γ

# Create Lotka-Volterra problem
LotkaVolterraProblem = PDSProblem(P, d, u0, tspan, p)
nothing # hide
```
All solvers of [OrdinaryDiffEq](https://docs.sciml.ai/OrdinaryDiffEq/stable/) can be used to solve the `PDSProblem`.
```@example LotkaVolterra
using OrdinaryDiffEq 
LotkaVolterraSol = solve(LotkaVolterraProblem, Tsit5(), reltol=1e-8, abstol=1e-8)
nothing # hide
```
Finally, we can plot the numerical solution.
```@example LotkaVolterra
using Plots
plot(LotkaVolterraSol)
savefig("LotkaVolterraPlot.svg"); nothing # hide
```
![](LotkaVolterraPlot.svg)

#### Conservative production-destruction systems

A PDS with the additional property
```math
  p_{ii}(t,\boldsymbol y)=d_{ii}(t,\boldsymbol y)=0
``` 
for ``i=1,\dots,N`` is called conservative. In this case we have

```math
\frac{d}{dt}\sum_{i=1}^N y_i=\sum_{i=1}^N y_i' = \sum_{\mathclap{i,j=1,\, i≠ j}}^N \bigl(p_{ij}(t,\boldsymbol y) - d_{ij}(t,\boldsymbol y)\bigr)= \sum_{\mathclap{i,j=1,\, i≠ j}}^N \bigl(p_{ij}(t,\boldsymbol y) - p_{ji}(t,\boldsymbol y)\bigr) = 0
```

and consequently the sum of state variables remains constant over time, i.e.
```math 
\sum_{i=1}^N y_i(t) = \sum_{i=1}^N y_i(0) 
```
for all times ``t>0``.

A specific example of a conservative PDS is the SIR model
```math
S' = -\frac{β S I}{N},\quad I'= \frac{β S I}{N} - γ I,\quad R'=γ I,
```
with ``N=S+I+R`` and ``\beta,\gamma>0``. Assuming ``S,I,R>0`` the production and destruction terms are given by
```math
p_{21}(S,I,R) = d_{12}(S,I,R) = \frac{β S I}{N},\quad p_{32}(S,I,R) = d_{23}(S,I,R) = γ I
```
and all missing production and destruction terms are zero.

In terms of implementation, a conservative PDS is completely described by the square matrix ``P=(p_{ij})_{i,j=1,\dots,N}``. There is no need for an additional vector to store the destruction terms since ``d_{ij} = p_{ji}`` for all ``i,j=1,\dots,N``. 

The following example shows how to implement the above SIR model with ``\beta=0.4, \gamma=0.04``, initial conditions ``S(0)=997, I(0)=3, R(0)=0`` and time domain ``(0, 100)``.

```@setup SIR
import Pkg; Pkg.add("OrdinaryDiffEq"); Pkg.add("Plots")
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
SIRProblem = ConservativePDSProblem(P, u0, tspan)
nothing # hide
```
`ConservativePDSProblem` is implemented as an `OrdinaryDiffEq.ODEProblem` and hence all solvers of [OrdinaryDiffEq](https://docs.sciml.ai/OrdinaryDiffEq/stable/) can be used to solve a `ConservativePDSProblem`. For instance, the SIR model from above can be solved with the method `Tsit5()` as follows.

```@example SIR
using OrdinaryDiffEq 

SIRSol = solve(SIRProblem,Tsit5())
nothing # hide
```
Finally, we can plot the numerical solution.
```@example SIR
using Plots

plot(SIRSol,legend=:right)
savefig("SIRplot.svg"); nothing # hide
```
![](SIRplot.svg)

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
