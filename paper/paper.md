---
title: 'PositiveIntegrators.jl: A Julia library of positivity-preserving time integration methods'
tags:
  - Julia
  - TODO
authors:
  - name: Stefan Kopecz
    orcid: 0000-0003-0593-450X
    affiliation: 1
  - name: Joshua Lampert
    orcid: 0009-0007-0971-6709
    affiliation: 2
  - name: Hendrik Ranocha
    orcid: 0000-0002-3456-2277
    affiliation: 3
affiliations:
 - name: Institute of Mathematics, University of Kassel, Germany
   index: 1
 - name: Department of Mathematics, University of Hamburg, Germany
   index: 2
 - name: Institute of Mathematics, Johannes Gutenberg University Mainz, Germany
   index: 3
date: 20 February 2025
bibliography: paper.bib
---

# Summary

Many systems of ordinary differential equations that model real-life applications have positive solutions, and it is quite natural to require that numerical solutions of such systems also remain positive. Unfortunately, positivity is a property that standard time integration schemes, such as Runge–Kutta methods, Rosenbrock methods, or linear multistep methods, do not preserve. Consequently, various new, unconditionally positive schemes have been introduced in recent years. Unfortunately, these new methods are not widely available, making them inaccessible to most users and limiting their comparability within the scientific community.

We introduce PositiveIntegrators.jl, a Julia package that provides efficient implementations of various positive time integration schemes, making these methods usable and comparable.


# Statement of need

Positivity-preserving time integration methods are helpful or even necessary to obtain meaningful solutions of specific ordinary differential equations. The only standard scheme with which unconditional positivity can be achieved is the implicit Euler method. However, this is only first-order accurate and, in addition, the preservation of positivity within the nonlinear iterations poses a problem. Another strategy for preserving positivity used in existing open source or commercial packages is to set negative solution components that are accepted by the step size control to zero, which can have a negative impact on possible conservations properties of the ODE system. Other approaches in the literature are additional projections after the calculation of negative solutions or it is tried to reduce the time step until a non-negative solution is calculated. Finally, SSP Runge-Kutta methods can also be used, although the preservation of positivity is again subject to step size limitations. Other approaches, especially modified Patankar--Runge--Kutta methods, are not yet available in software packages. We make these methods available to make them usable for users and comparable for researchers.


# Features

The package is fully compatible with `DifferentialEquations.jl` and therefore many features that are available there can be used directly.
It offers implementations of conservative and non-conservative production-destruction systems, including conversions to standard `ODEProblem`s from `DifferentialEquations.jl`. 
Production-destruction systems are the building blocks for solving differential equations with MPRK schemes. 

The package provides several MPRK methods to solve production-destruction systems:
- The MPRK methods `MPE`, `MPRK22`, `MPRK43I` and `MPRK43II` of Kopecz and Meister are based on the classical formulation of Runge--Kutta schemes and have accuracies from first to third order.
- The MPRK methods `SSPMPRK22` and `SSPMPRK43` of Huang, Zhao and Shu are based on the SSP formulation of Runge--Kutta schemes and are of second or third order. 
- The `MPDeC` methods of Öffner and Torlo combine the deferred correction approach with the idea of MPRK schemes to obtain schemes of arbitrary order. In the package methods from second up to 10th order are implemented.

In addition, the methods mentioned above have been extended so that non-conservative and non-autonomous production-destruction systems can be solved as well. Furthermore, adaptive step size control is available for most schemes.

# Related research and software

## Research

The first MPRK schemes were introduced in @burchard2003. These are the first order scheme `MPE` and a second order scheme based on Heun's method. To avoid the restriction to Heun's method, the second order `MPRK22` schemes were developed in @kopeczmeister2018order2. The techniques developed therein, also enabled a generalization to third order schemes and thus the introduction of `MPRK43I` and `MPRK43II` schemes in @kopeczmeister2018order3.

All aforementioned schemes were derived from the classical formulation of Runge-Kutta schemes. Using the Shu-Osher formulation instead lead to the introduction of the second order schemes `SSPMPRK22` in @huang2019order2 and the third order scheme `SSPMPRK43` in @huang2019order3.

Starting from a low order scheme, the deferred correction approach can be used to increase the scheme's approximation order iteratively. In @OeffnerTorlo2020 deferred correction was combined with the MPRK idea to devise MPRK schemes of arbitrary order. These are implemented as `MPDeC` schemes. 

The implemented schemes were originally introduced for conservative production-destruction systems only. An extension to non-conservative production-destruction-systems was presented in @benzmeister2015. We implemented a modification of this algorithm, by treating the non-conservative production and destruction terms separately, weighting the destruction terms and leaving the production terms unweighted.

Readers interested in additional theoretical background and further properties of the implemented schemes are referred to the following papers: @kopeczmeister2019, @izgin2022stability1, @izgin2022stability2, @huang2023, @torlo2022, @izginoeffner2023

## Software

Existing software libraries do not have a strong focus on unconditional positivity and, to the authors' knowledge, there is no other software library which offers MPRK schemes. 
A common strategy to obtain nonnegative solutions used in the `PositiveDomain` callback of `Differentialequtions.jl` or the commercial package `Matlab` is described by @Shampine2005. In this approach negative components of approximate solutions that have been accepted by the adaptive time stepping algorithm are set to zero.
Another possibility is to reduce the chosen time step size beyond accuracy considerations until a non-negative approximation is calculated. This can be achieved in `DifferentialEquations.jl` using the solver option `isoutofdomain`.

We also mention that some papers on MPRK schemes offer supplementary codes. However, these are mainly small scripts for the reproduction of results shown in the papers and are not intended as software libraries.


TODO

@bartel2024structure


# Acknowledgements

Hendrik Ranocha was supported by
the German Research Foundation (Deutsche Forschungsgemeinschaft DFG, project number 513301895) and
the Daimler und Benz Stiftung (Daimler and Benz foundation, project number 32-10/22).


# References
