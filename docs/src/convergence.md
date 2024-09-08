# [Experimental convergence order of MPRK schemes](@id convergence_mprk)

## Second order MPRK schemes

```@example convergence
using PositiveIntegrators

# choose schemes
algs = [MPRK22(0.5)
        MPRK22(2.0 / 3.0)
        MPRK22(1.0)
        SSPMPRK22(0.5, 1.0)]

names = ["MPRK22(0.5)"
         "MPRK22(2.0/3.0)"
         "MPRK22(1.0)"
         "SSPMPRK22(0.5, 1.0)"]

prob = prob_pds_linmod

nothing #hide
```
```@example convergence
using DiffEqDevTools

dts = 0.5 .^ (5:10)
err = Vector{Vector{Float64}}(undef, length(algs))
eoc = Vector{Vector{Float64}}(undef, length(algs))

#compute errors and experimental order of convergence
for i in eachindex(algs)
    sim = test_convergence(dts, prob, algs[i])
    sims[i] = sim
    err[i] = sim.errors[:l∞]
    eoc[i] = -log2.(err[i][2:end] ./ err[i][1:(end - 1)])
end
```
```@example convergence
using PrettyTables

# collect data and create headers
data = dts
header = ["Δt"]
subheader = [""]
for i in eachindex(algs)
    data = [data err[i] [NaN; eoc[i]]]
    header = [header names[i] names[i]]
    subheader = [subheader "error" "order"]
end

# print table
pretty_table(data, header = (header, subheader),
             formatters = (ft_printf("%5.4e", [1, 2, 4, 6, 8]),
                           ft_printf("%5.4f", [3, 5, 7, 9])))
```

## Third order MPRK schemes

```@example convergence

#select schemes
algs = [MPRK43I(1.0, 0.5)
        MPRK43I(0.5, 0.75)
        MPRK43II(0.5)
        MPRK43II(2.0 / 3.0)
        SSPMPRK43()]

names = ["MPRK43I(1.0,0.5)"
         "MPRK43I(0.5, 0.75)"
         "MPRK43II(0.5)"
         "MPRK43II(2.0/3.0)"
         "SSPMPRK43()"]

#compute errors and experimental order of convergence
sims = Vector{ConvergenceSimulation}(undef, length(algs))
err = Vector{Vector{Float64}}(undef, length(algs))
p = Vector{Vector{Float64}}(undef, length(algs))
for i in eachindex(algs)
    sim = test_convergence(dts, prob, algs[i])
    sims[i] = sim
    err[i] = sim.errors[:l∞]
    p[i] = -log2.(err[i][2:end] ./ err[i][1:(end - 1)])
end

# collect data and create headers
data = dts
header = ["Δt"]
subheader = [""]
for i in eachindex(algs)
    data = [data err[i] [NaN; p[i]]]
    header = [header names[i] names[i]]
    subheader = [subheader "error" "order"]
end

pretty_table(data, header = (header, subheader),
             formatters = (ft_printf("%5.4e", [1, 2, 4, 6, 8, 10]),
                           ft_printf("%5.4f", [3, 5, 7, 9, 11])))
```