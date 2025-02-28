# [Experimental convergence order of MPRK schemes](@id convergence_mprk)

In this tutorial, we check that the implemented MPRK schemes have the expected order of convergence. 

## Conservative production-destruction systems

First, we consider conservative production-destruction systems (PDS). To investigate the convergence order, we define the non-autonomous test problem 

```math
\begin{aligned}
u_1' &= \cos(\pi t)^2 u_2 - \sin(2\pi t)^2 u_1, & u_1(0)&=0.9, \\
u_2' & = \sin(2\pi t)^2 u_1 - \cos(\pi t)^2 u_2, & u_2(0)&=0.1,
\end{aligned}
```
for ``0≤ t≤ 1``.
The PDS is conservative since the sum of the right-hand side terms equals zero. 
An implementation of the problem is given next.


```@example eoc
using PositiveIntegrators

# define problem
P(u, p, t) = [0.0 cos.(π * t) .^ 2 * u[2]; sin.(2 * π * t) .^ 2 * u[1] 0.0]
prob = ConservativePDSProblem(P, [0.9; 0.1], (0.0, 1.0))

nothing # hide
```

To use `analyticless_test_convergence` from [DiffEqDevTools.jl](https://github.com/SciML/DiffEqDevTools.jl), we need to pick a solver to compute the reference solution and specify tolerances.
Since the problem is not stiff, we use the high-order explicit solver `Vern9()` from [OrdinaryDiffEqVerner.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/).
Moreover, we choose time step sizes to investigate the convergence behavior. 

```@example eoc
using OrdinaryDiffEqVerner
using DiffEqDevTools # load analyticless_test_convergence

# solver and tolerances to compute reference solution
test_setup = Dict(:alg => Vern9(), :reltol => 1e-14, :abstol => 1e-14)

# choose step sizes
dts = 0.5 .^ (5:10)

nothing # hide
```

### Second order MPRK schemes

First, we test several second order MPRK schemes.

```@example eoc
# select schemes
algs2 = [MPRK22(0.5); MPRK22(2.0 / 3.0); MPRK22(1.0); SSPMPRK22(0.5, 1.0)]
labels2 = ["MPRK22(0.5)"; "MPRK22(2.0/3.0)"; "MPRK22(1.0)"; "SSPMPRK22(0.5, 1.0)"]

#compute errors and experimental order of convergence
err_eoc = []
for i in eachindex(algs2)
     sim = analyticless_test_convergence(dts, prob, algs2[i], test_setup)

     err = sim.errors[:l∞]
     eoc = [NaN; -log2.(err[2:end] ./ err[1:(end - 1)])]

     push!(err_eoc, tuple.(err, eoc))
end
```

Next, we print a table with the computed data.
The table lists the errors obtained with the respective time step size ``Δ t`` as well as the estimated order of convergence in parentheses.

```@example eoc
using Printf: @sprintf
using PrettyTables: pretty_table

# gather data for table
data = hcat(dts, reduce(hcat,err_eoc))

# print table
formatter = (v, i, j) ->  (j>1) ? (@sprintf "%5.2e (%4.2f) " v[1] v[2]) : (@sprintf "%5.2e " v)
pretty_table(data, formatters = formatter, header = ["Δt"; labels2])                  
```

The table shows that all schemes converge as expected.

### Third-order MPRK schemes

In this section, we proceed as above, but consider third-order MPRK schemes instead.

```@example eoc
# select 3rd order schemes
algs3 = [MPRK43I(1.0, 0.5); MPRK43I(0.5, 0.75); MPRK43II(0.5); MPRK43II(2.0 / 3.0); 
         SSPMPRK43()]
labels3 = ["MPRK43I(1.0,0.5)"; "MPRK43I(0.5, 0.75)"; "MPRK43II(0.5)"; "MPRK43II(2.0/3.0)";
          "SSPMPRK43()"]

# compute errors and experimental order of convergence
err_eoc = []
for i in eachindex(algs3)
     sim = analyticless_test_convergence(dts, prob, algs3[i], test_setup)

     err = sim.errors[:l∞]
     eoc = [NaN; -log2.(err[2:end] ./ err[1:(end - 1)])]

     push!(err_eoc, tuple.(err, eoc))
end

# gather data for table
data = hcat(dts, reduce(hcat,err_eoc))

# print table
formatter = (v, i, j) ->  (j>1) ? (@sprintf "%5.2e (%4.2f) " v[1] v[2]) : (@sprintf "%5.2e " v)
pretty_table(data, formatters = formatter, header = ["Δt"; labels3])  
```

As above, the table shows that all schemes converge as expected.

## Non-conservative PDS

In this section we consider the non-autonomous but non-conservative test problem 

```math
\begin{aligned}
u_1' &= \cos(\pi t)^2 u_2 - \sin(2\pi t)^2 u_1 - \cos(2\pi t)^2 u_1, & u_1(0)&=0.9,\\
u_2' & = \sin(2\pi t)^2 u_1 - \cos(\pi t)^2 u_2 - \sin(\pi t)^2 u_2, & u_2(0)&=0.1,
\end{aligned}
```

for ``0≤ t≤ 1``.
Since the sum of the right-hand side terms does not cancel, the PDS is indeed non-conservative.
Hence, we need to use [`PDSProblem`](@ref) for its implementation.

```@example eoc
# choose problem
P(u, p, t) = [0.0 cos.(π * t) .^ 2 * u[2]; sin.(2 * π * t) .^ 2 * u[1] 0.0]
D(u, p, t) = [cos.(2 * π * t) .^ 2 * u[1]; sin.(π * t) .^ 2 * u[2]]
prob = PDSProblem(P, D, [0.9; 0.1], (0.0, 1.0))

nothing # hide
```

The following sections will show that the selected MPRK schemes show the expected convergence order also for this non-conservative PDS.

### Second-order MPRK schemes

```@example eoc
# compute errors and experimental order of convergence
err_eoc = []
for i in eachindex(algs2)
     sim = analyticless_test_convergence(dts, prob, algs2[i], test_setup)

     err = sim.errors[:l∞]
     eoc = [NaN; -log2.(err[2:end] ./ err[1:(end - 1)])]

     push!(err_eoc, tuple.(err, eoc))
end

# gather data for table
data = hcat(dts, reduce(hcat,err_eoc))

# print table
formatter = (v, i, j) ->  (j>1) ? (@sprintf "%5.2e (%4.2f) " v[1] v[2]) : (@sprintf "%5.2e " v)
pretty_table(data, formatters = formatter, header = ["Δt"; labels2])                  
```

### Third-order MPRK schemes

```@example eoc
# compute errors and experimental order of convergence
err_eoc = []
for i in eachindex(algs3)
     sim = analyticless_test_convergence(dts, prob, algs3[i], test_setup)

     err = sim.errors[:l∞]
     eoc = [NaN; -log2.(err[2:end] ./ err[1:(end - 1)])]

     push!(err_eoc, tuple.(err, eoc))
end

# gather data for table
data = hcat(dts, reduce(hcat,err_eoc))

# print table
formatter = (v, i, j) ->  (j>1) ? (@sprintf "%5.2e (%4.2f) " v[1] v[2]) : (@sprintf "%5.2e " v)
pretty_table(data, formatters = formatter, header = ["Δt"; labels3])  
```