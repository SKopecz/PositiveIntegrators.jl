# [Experimental convergence order of MPRK schemes](@id convergence_mprk)

In this tutorial we check that the implemented MPRK schemes have the expected order of convergence. 
As a test problem we choose [`prob_pds_linmod`](@ref).

```@example eoc
using PositiveIntegrators
using Printf # load @sprintf
using DiffEqDevTools # load test_convergence
using PrettyTables # load pretty_table

# choose problem
prob = prob_pds_linmod

# choose step sizes
dts = 0.5 .^ (5:10)

nothing # hide output
```

## Second order MPRK schemes

First, we test the second order MPRK schemes.

```@example eoc
# select schemes
algs = [MPRK22(0.5); MPRK22(2.0 / 3.0); MPRK22(1.0); SSPMPRK22(0.5, 1.0)]
names = ["MPRK22(0.5)"; "MPRK22(2.0/3.0)"; "MPRK22(1.0)"; "SSPMPRK22(0.5, 1.0)"]

#compute errors and experimental order of convergence
err_eoc = Vector{Any}[]
for i in eachindex(algs)
     sim = test_convergence(dts, prob, algs[i])

     err = sim.errors[:l∞]
     eoc = [NaN; -log2.(err[2:end] ./ err[1:(end - 1)])]

     push!(err_eoc, tuple.(err, eoc))
end

# gather data for table
data = hcat(dts, reduce(hcat,err_eoc))

# print table
formatter = (v, i, j) ->  (j>1) ? (@sprintf "%5.2e (%4.2f) " v[1] v[2]) : (@sprintf "%5.2e " v)
pretty_table(data, formatters = formatter, header = ["Δt"; names])                  
```

The above table lists the used time step sizes in the first column. The following columns contain the error obtaind with the respective time step size as well as the estimated order of convergence in parenthesis.

## Third order MPRK schemes


```@example eoc
# select 3rd order schemes
algs = [MPRK43I(1.0, 0.5); MPRK43I(0.5, 0.75); MPRK43II(0.5); MPRK43II(2.0 / 3.0); SSPMPRK43()]
names = ["MPRK43I(1.0,0.5)"; "MPRK43I(0.5, 0.75)"; "MPRK43II(0.5)"; "MPRK43II(2.0/3.0)"; "SSPMPRK43()"]

#compute errors and experimental order of convergence
err_eoc = Vector{Any}[]
for i in eachindex(algs)
     sim = test_convergence(dts, prob, algs[i])

     err = sim.errors[:l∞]
     eoc = [NaN; -log2.(err[2:end] ./ err[1:(end - 1)])]

     push!(err_eoc, tuple.(err, eoc))
end

# gather data for table
data = hcat(dts, reduce(hcat,err_eoc))

# print table
formatter = (v, i, j) ->  (j>1) ? (@sprintf "%5.2e (%4.2f) " v[1] v[2]) : (@sprintf "%5.2e " v)
pretty_table(data, formatters = formatter, header = ["Δt"; names])  
```

The above table lists the used time step sizes in the first column. The following columns contain the error obtaind with the respective time step size as well as the estimated order of convergence in parenthesis.