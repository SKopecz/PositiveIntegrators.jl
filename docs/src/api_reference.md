# PositiveIntegrators.jl API

```@meta
CurrentModule = PositiveIntegrators
```

## Problem types

```@docs
ConservativePDSProblem
PDSProblem
```

## Example problems

```@docs
prob_pds_bertolazzi
prob_pds_brusselator
prob_pds_linmod
prob_pds_linmod_inplace
prob_pds_nonlinmod
prob_pds_npzd
prob_pds_robertson
prob_pds_sir
prob_pds_stratreac
```

## Algorithms

```@docs
MPE
MPRK22
SSPMPRK22
MPRK43I
MPRK43II
SSPMPRK43
```

## Auxiliary functions

```@docs
isnegative
isnonnegative
```
