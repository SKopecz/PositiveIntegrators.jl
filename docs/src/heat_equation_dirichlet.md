# [Tutorial: Solution of the heat equation with Dirichlet boundary conditions](@id tutorial-heat-equation-dirichlet)

We continue the previous tutorial on
[solving the heat equation with Neumann boundary conditions](@ref tutorial-heat-equation-neumann)
by looking at Dirichlet boundary conditions instead, resulting in a non-conservative
production-destruction system.


## Definition of the (non-conservative) production-destruction system

Consider the heat equation

```math
\partial_t u(t,x) = \mu \partial_x^2 u(t,x),\quad u(0,x)=u_0(x),
```

with ``μ ≥ 0``, ``t≥ 0``, ``x\in[0,1]``, and homogeneous Dirichlet boundary conditions.
We use again a finite volume discretization, i.e., we split the domain ``[0, 1]`` into
``N`` uniform cells of width ``\Delta x = 1 / N``. As degrees of freedom, we use
the mean values of ``u(t)`` in each cell approximated by the point value ``u_i(t)``
in the center of cell ``i``. Finally, we use the classical central finite difference
discretization of the Laplacian with homogeneous Dirichlet boundary conditions,
resulting in the ODE

```math
\partial_t u(t) = L u(t),
\quad
L = \frac{\mu}{\Delta x^2} \begin{pmatrix}
    -2 & 1 \\
    1 & -2 & 1 \\
    & \ddots & \ddots & \ddots \\
    && 1 & -2 & 1 \\
    &&& 1 & -2
\end{pmatrix}.
```

The system can be written as a non-conservative PDS with production terms

```math
\begin{aligned}
&p_{i,i-1}(t,\mathbf u(t)) = \frac{\mu}{\Delta x^2} u_{i-1}(t),\quad i=2,\dots,N, \\
&p_{i,i+1}(t,\mathbf u(t)) = \frac{\mu}{\Delta x^2} u_{i+1}(t),\quad i=1,\dots,N-1,
\end{aligned}
```

and destruction terms ``d_{i,j} = p_{j,i}`` for ``i \ne j`` as well as the
non-conservative destruction terms

```math
\begin{aligned}
d_{1,1}(t,\mathbf u(t)) &= \frac{\mu}{\Delta x^2} u_{1}(t), \\
d_{N,N}(t,\mathbf u(t)) &= \frac{\mu}{\Delta x^2} u_{N}(t).
\end{aligned}
```


## Solution of the non-conservative production-destruction system

Now we are ready to define a [`PDSProblem`](@ref) and to solve this
problem with a method of
[PositiveIntegrators.jl](https://github.com/SKopecz/PositiveIntegrators.jl) or
[OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/).
In the following we use ``N = 100`` nodes and the time domain ``t \in [0,1]``.
Moreover, we choose the initial condition

```math
u_0(x) = \sin(\pi x)^2.
```

```@example HeatEquationDirichlet
x_boundaries = range(0, 1, length = 101)
x = x_boundaries[1:end-1] .+ step(x_boundaries) / 2
u0 = @. sinpi(x)^2 # initial solution
tspan = (0.0, 1.0) # time domain

nothing #hide
```

We will choose three different matrix types for the production terms and
the resulting linear systems:

1. standard dense matrices (default)
2. sparse matrices (from SparseArrays.jl)
3. tridiagonal matrices (from LinearAlgebra.jl)


### Standard dense matrices

```@example HeatEquationDirichlet
using PositiveIntegrators # load ConservativePDSProblem

function heat_eq_P!(P, u, μ, t)
    fill!(P, 0)
    N = length(u)
    Δx = 1 / N
    μ_Δx2 = μ / Δx^2

    let i = 1
        # Dirichlet boundary condition
        P[i, i + 1] = u[i + 1] * μ_Δx2
    end

    for i in 2:(length(u) - 1)
        # interior stencil
        P[i, i - 1] = u[i - 1] * μ_Δx2
        P[i, i + 1] = u[i + 1] * μ_Δx2
    end

    let i = length(u)
        # Dirichlet boundary condition
        P[i, i - 1] = u[i - 1] * μ_Δx2
    end

    return nothing
end

function heat_eq_D!(D, u, μ, t)
    fill!(D, 0)
    N = length(u)
    Δx = 1 / N
    μ_Δx2 = μ / Δx^2

    # Dirichlet boundary condition
    D[begin] = u[begin] * μ_Δx2
    D[end] = u[end] * μ_Δx2

    return nothing
end

μ = 1.0e-2
prob = PDSProblem(heat_eq_P!, heat_eq_D!, u0, tspan, μ) # create the PDS

sol = solve(prob, MPRK22(1.0); save_everystep = false)

nothing #hide
```

```@example HeatEquationDirichlet
using Plots

plot(x, u0; label = "u0", xguide = "x", yguide = "u")
plot!(x, last(sol.u); label = "u")
```


### Sparse matrices

To use different matrix types for the production terms and linear systems,
you can use the keyword argument `p_prototype` of
[`ConservativePDSProblem`](@ref) and [`PDSProblem`](@ref).

```@example HeatEquationDirichlet
using SparseArrays
p_prototype = spdiagm(-1 => ones(eltype(u0), length(u0) - 1),
                      +1 => ones(eltype(u0), length(u0) - 1))
prob_sparse = PDSProblem(heat_eq_P!, heat_eq_D!, u0, tspan, μ;
                         p_prototype = p_prototype)

sol_sparse = solve(prob_sparse, MPRK22(1.0); save_everystep = false)

nothing #hide
```

```@example HeatEquationDirichlet
plot(x,u0; label = "u0", xguide = "x", yguide = "u")
plot!(x, last(sol_sparse.u); label = "u")
```


### Tridiagonal matrices

The sparse matrices used in this case have a very special structure
since they are in fact tridiagonal matrices. Thus, we can also use
the special matrix type `Tridiagonal` from the standard library
`LinearAlgebra`.

```@example HeatEquationDirichlet
using LinearAlgebra
p_prototype = Tridiagonal(ones(eltype(u0), length(u0) - 1),
                          ones(eltype(u0), length(u0)),
                          ones(eltype(u0), length(u0) - 1))
prob_tridiagonal = PDSProblem(heat_eq_P!, heat_eq_D!, u0, tspan, μ;
                              p_prototype = p_prototype)

sol_tridiagonal = solve(prob_tridiagonal, MPRK22(1.0); save_everystep = false)

nothing #hide
```

```@example HeatEquationDirichlet
plot(x, u0; label = "u0", xguide = "x", yguide = "u")
plot!(x, last(sol_tridiagonal.u); label = "u")
```



### Performance comparison

Finally, we use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
to compare the performance of the different implementations.

```@example HeatEquationDirichlet
using BenchmarkTools
@benchmark solve(prob, MPRK22(1.0); save_everystep = false)
```

```@example HeatEquationDirichlet
@benchmark solve(prob_sparse, MPRK22(1.0); save_everystep = false)
```

By default, we use an LU factorization for the linear systems. At the time of
writing, Julia uses
[SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl)
defaulting to UMFPACK from SuiteSparse in this case. However, the linear
systems do not necessarily have the structure for which UMFPACK is optimized
 for. Thus, it is often possible to gain performance by switching to KLU
 instead.

```@example HeatEquationDirichlet
using LinearSolve
@benchmark solve(prob_sparse, MPRK22(1.0; linsolve = KLUFactorization()); save_everystep = false)
```

```@example HeatEquationDirichlet
@benchmark solve(prob_tridiagonal, MPRK22(1.0); save_everystep = false)
```


## Package versions

These results were obtained using the following versions.
```@example HeatEquationDirichlet
using InteractiveUtils
versioninfo()
println()

using Pkg
Pkg.status(["PositiveIntegrators", "SparseArrays", "KLU", "LinearSolve", "OrdinaryDiffEq"],
           mode=PKGMODE_MANIFEST)
nothing # hide
```
