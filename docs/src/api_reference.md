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
prob_pds_minmapk
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
MPDeC
```

## Auxiliary functions

```@docs
isnegative
isnonnegative
rel_max_error_tend
rel_max_error_overall
rel_l1_error_tend
rel_l2_error_tend
work_precision_adaptive
work_precision_adaptive!
work_precision_fixed
work_precision_fixed!
```
