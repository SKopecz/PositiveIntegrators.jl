# [Experimental convergence order of MPRK schemes](@id convergence_mprk)

In this tutorial we check that the implemented MPRK schemes have the expected order of convergence. 

## Conservative PDS

First, we consider conservative PDS and define an academic non-autonomous test problem.

```@example eoc
using PositiveIntegrators

# choose problem
P(u, p, t) = [0.0 cos.(π * t) .^ 2 * u[2]; sin.(2 * π * t) .^ 2 * u[1] 0.0]
prob = ConservativePDSProblem(P, [0.9; 0.1], (0.0, 10.0))

nothing # hide output
```

To use `analyticless_test_convergence` from `DiffEqDevTools` we need pick a solver to compute the reference solution and specify tolerances.
Moreover, we need to choose the different time step sizes.

```@example eoc
using OrdinaryDiffEq
using DiffEqDevTools # load analyticless_test_convergence

# solver and tolerances to compute reference solution
test_setup = Dict(:alg => Vern9(), :reltol => 1e-14, :abstol => 1e-14)

# choose step sizes
dts = 0.5 .^ (5:10)
```

### Second order MPRK schemes

First, we test the second order MPRK schemes.

```@example eoc
# select schemes
algs2 = [MPRK22(0.5); MPRK22(2.0 / 3.0); MPRK22(1.0); SSPMPRK22(0.5, 1.0)]
names2 = ["MPRK22(0.5)"; "MPRK22(2.0/3.0)"; "MPRK22(1.0)"; "SSPMPRK22(0.5, 1.0)"]

#compute errors and experimental order of convergence
err_eoc = Vector{Any}[]
for i in eachindex(algs2)
     sim = analyticless_test_convergence(dts, prob, algs2[i], test_setup)

     err = sim.errors[:l∞]
     eoc = [NaN; -log2.(err[2:end] ./ err[1:(end - 1)])]

     push!(err_eoc, tuple.(err, eoc))
end
```

Finally, we print a table with the computed data. The table lists the used time step sizes in the first column. The following columns contain the error obtaind with the respective time step size as well as the estimated order of convergence in parenthesis.

```@example eoc
using Printf # load @sprintf
using PrettyTables # load pretty_table

# gather data for table
data = hcat(dts, reduce(hcat,err_eoc))

# print table
formatter = (v, i, j) ->  (j>1) ? (@sprintf "%5.2e (%4.2f) " v[1] v[2]) : (@sprintf "%5.2e " v)
pretty_table(data, formatters = formatter, header = ["Δt"; names2])                  
```

### Third order MPRK schemes

In this section, we proceed as above, but consider third order schemes instead.

```@example eoc
# select 3rd order schemes
algs3 = [MPRK43I(1.0, 0.5); MPRK43I(0.5, 0.75); MPRK43II(0.5); MPRK43II(2.0 / 3.0); SSPMPRK43()]
names3 = ["MPRK43I(1.0,0.5)"; "MPRK43I(0.5, 0.75)"; "MPRK43II(0.5)"; "MPRK43II(2.0/3.0)"; "SSPMPRK43()"]

#compute errors and experimental order of convergence
err_eoc = Vector{Any}[]
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
pretty_table(data, formatters = formatter, header = ["Δt"; names3])  
```

## Non-conservative PDS

Next, we consider a non-conservative non-autonomous PDS as a test problem. 

```@example eoc

# choose problem
P(u, p, t) = [0.0 cos.(π * t) .^ 2 * u[2]; sin.(2 * π * t) .^ 2 * u[1] 0.0]
D(u, p, t) = [cos.(2 * π * t) .^ 2 * u[1]; sin.(π * t) .^ 2 * u[2]]
prob = PDSProblem(P, D, [0.9; 0.1], (0.0, 10.0))

nothing # hide output
```

### Second order MPRK schemes

```@example eoc
#compute errors and experimental order of convergence
err_eoc = Vector{Any}[]
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
pretty_table(data, formatters = formatter, header = ["Δt"; names2])                  
```

### Third order MPRK schemes

```@example eoc
#compute errors and experimental order of convergence
err_eoc = Vector{Any}[]
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
pretty_table(data, formatters = formatter, header = ["Δt"; names3])  
```