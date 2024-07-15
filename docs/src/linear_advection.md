# Tutorial: Solution of the linear advection equation

This tutorial is about the efficient solution of production-destruction systems (PDS) with a large number of differential equations. 
We will explore several ways to represent such large systems and assess their efficiency. 

## Definition of the production-destruction system

One example of the occurrence of a PDS with a large number of equations is the space discretization of a partial differential equation. In this tutorial we want to solve the linear advection equation

```math
\partial_t u(t,x)=-a\partial_x u(t,x),\quad u(0,x)=u_0(x)
```

with ``a>0``, ``t≥ 0``, ``x\in[0,1]`` and periodic boundary conditions. To keep things as simple as possible, we 
discretize the space domain as ``0=x_0<x_1\dots <x_{N-1}<x_N=1`` with ``x_i = i Δ x`` for ``i=0,\dots,N`` and ``Δx=1/N``. An upwind discretization of the spatial derivative yields the ODE system

```math
\begin{aligned}
&\partial_t u_1(t) =-\frac{a}{Δx}\bigl(u_1(t)-u_{N}(t)\bigr),\\
&\partial_t u_i(t) =-\frac{a}{Δx}\bigl(u_i(t)-u_{i-1}(t)\bigr),\quad i=2,\dots,N,
\end{aligned}
```

where ``u_i(t)`` is an approximation of ``u(t,x_i)`` for ``i=1,\dots, N``.
This system can also be written as ``\partial_t \mathbf u(t)=\mathbf A\mathbf u(t)`` with ``\mathbf u(t)=(u_1(t),\dots,u_N(t))`` and 

```math
\mathbf A= \frac{a}{Δ x}\begin{bmatrix}-1&0&\dots&0&1\\1&-1&\ddots&&0\\0&\ddots&\ddots&\ddots&\vdots\\ \vdots&\ddots&\ddots&\ddots&0\\0&\dots&0&1&-1\end{bmatrix}.
```

In particular the matrix ``\mathbf A`` shows that there is a single production term and a single destruction term per equation. 
Furthermore, the system is conservative as ``\mathbf A`` has column sum zero.
To be precise, the production matrix ``\mathbf P = (p_{i,j})`` of this conservative PDS is given by

```math
\begin{aligned}
&p_{1,N}(t,\mathbf u(t)) = \frac{a}{Δ x}u_N(t),\\
&p_{i,i-1}(t,\mathbf u(t)) = \frac{a}{Δ x}u_{i-1}(t),\quad i=2,\dots,N.
\end{aligned}
```

Since the PDS is conservative, we have ``d_{i,j}=p_{j,i}`` and the system is fully determined by the production matrix ``\mathbf P``.

## Solution of the production-destruction system

Now we are ready to define a `ConservativePDSProblem` and to solve this problem with a method of [PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl) or [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/). In the following we use ``a=1``, ``N=1000`` and the time domain ``t\in[0,1]``. Moreover, we choose the step function

```math
u_0(x)=\begin{cases}1, & 0.4 ≤ x ≤ 0.6,\\ 0,& \text{elsewhere}\end{cases}
```

as initial condition. Due to the periodic boundary conditions and the transport velocity ``a=1``, the solution at time ``t=1`` is identical to the initial distribution, i.e. ``u(1,x) = u_0(x)``.

```@example LinearAdvection
N = 1000 # number of subintervals
dx = 1/N # mesh width
x = LinRange(dx, 1.0, N) # discretization points x_1,...,x_N = x_0
u0 = @. 0.0 + (0.4 ≤ x ≤ 0.6) * 1.0 # initial solution
tspan = (0.0, 1.0) # time domain

nothing #hide
```

As mentioned above, we will try different approaches to solve this PDS and compare their efficiency. These are
1. an in-place implementation with a dense matrix,
2. an in-place implementation with a sparse matrix.

### Standard in-place implementation

```@example LinearAdvection
using PositiveIntegrators # load ConservativePDSProblem

function lin_adv_P!(P, u, p, t)
    P .= 0.0
    N = length(u)
    dx = 1 / N
    P[1, N] = u[N] / dx
    for i in 2:N
        P[i, i - 1] = u[i - 1] / dx
    end
    return nothing
end

prob = ConservativePDSProblem(lin_adv_P!, u0, tspan) # create the PDS

sol = solve(prob, MPRK43I(1.0, 0.5); save_everystep = false)

nothing #hide
```

```@example LinearAdvection
using Plots

plot(x, u0; label = "u0", xguide = "x", yguide = "u")
plot!(x, last(sol.u); label = "u")
```

### Using sparse matrices

TODO: Some text

```@example LinearAdvection
using SparseArrays
p_prototype = spdiagm(-1 => ones(eltype(u0), N - 1),
                      N - 1 => ones(eltype(u0), 1))
prob_sparse = ConservativePDSProblem(lin_adv_P!, u0, tspan; p_prototype=p_prototype)

sol_sparse = solve(prob_sparse, MPRK43I(1.0, 0.5); save_everystep = false)

nothing #hide
```

```@example LinearAdvection
plot(x,u0)
plot!(x, last(sol_sparse.u))
```

```@example LinearAdvection
using BenchmarkTools
@benchmark solve(prob, MPRK43I(1.0, 0.5); save_everystep = false)
```

```@example LinearAdvection
@benchmark solve(prob_sparse, MPRK43I(1.0, 0.5); save_everystep = false)
```