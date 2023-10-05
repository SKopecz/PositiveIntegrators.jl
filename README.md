# PositiveIntegrators.jl
Over the last two decades several approaches have been suggested to numerically preserve the positivity of positive ODE systems. This package provides efficient implementations of various positive time integration schemes, allowing a fair comparison of the different schemes. The package extends OrdinaryDiffEq.jl by
* adding a new problem type for production-destruction systems
* adding the algorithms of first and second order modified Patankar-Runge-Kutta (MPRK) schemes



