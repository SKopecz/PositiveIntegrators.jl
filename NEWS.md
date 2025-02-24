# Changelog

PositiveIntegrators.jl.jl follows the interpretation of
[semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file
for human readability.


## Changes in the v0.2 lifecycle

### Changed

- The minimum required Julia version was updated to v1.10


## Changes when updating to v0.2 from v0.1.x

#### Removed

- The optional keyword argument `d_prototype` has been removed from `PDSProblem`


## Changes in the v0.1 lifecycle

### Added

- MPRK methods `MPRK43I` and `MPRK43II` of Kopecz and Meister
- SSP MPRK methods `SSPMPRK22` and `SSPMPRK43` of Huang and Shu


## Initial release v0.1.0

- Production-destruction problems `PDSProblem` and `ConservativePDSProblem`,
  including conversions to standard `ODEProblem`s from OrdinaryDiffEq.jl
- Some default problems such as `prob_pds_bertolazzi`, `prob_pds_brusselator`,
  `prob_pds_linmod`, `prob_pds_nonlinmod`, `prob_pds_npzd`, `prob_pds_robertson`,
  `prob_pds_sir`, `prob_pds_stratreac`
- Modified Patankar methods `MPE` and `MPRK22`
