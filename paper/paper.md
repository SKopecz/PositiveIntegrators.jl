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

We introduce PositiveIntegrators.jl, a Julia package that provides efficient implementations of various time integration schemes for the solution of positive ordinary differential equations, making these methods accessible for users and comparable for researchers. Currently, the package provides MPRK, SSP-MPRK, and MPDeC schemes, all of which are unconditionally positive and also preserve the conservation property when applied to a conservative system.
The package is fully compatible with DifferentialEquations.jl, which allows a direct comparison between the provided schemes and standard methods.


# Statement of need

Many systems of ordinary differential equations that model real-life applications have positive solutions, and it is quite natural to require that numerical solutions of such systems also remain positive.
For some of these systems unconditionally positivity-preserving time integration methods are helpful or even necessary to obtain meaningful solutions. 

Unfortunately, positivity is a property that almost all standard time integration schemes, such as Runge–Kutta methods, Rosenbrock methods, or linear multistep methods, do not preserve.
In particular, higher-order general linear methods cannot preserve positivity unconditionally [@bolley1978conservation].
The only standard scheme with which unconditional positivity can be achieved is the implicit Euler method
(assuming that the nonlinear systems are solved exactly). However, this is only first-order accurate and, in addition, the preservation of positivity within the nonlinear iteration process poses a problem. 
Another strategy for preserving positivity used in existing open source or commercial packages (like MATLAB) is to set negative solution components that are accepted by the step size control to zero. Unfortunately, this can have a negative impact on possible conservation properties. Further approaches in the literature include projections in between time steps [@sandu2001positive; @nusslein2021positivity], if a negative solution was computed, or it is tried to reduce the time step size as long as a nonnegative solution is calculated. Finally, strong stability preserving (SSP) methods can also be used to preserve positivity, but this is again subject to step size limitations [@gottlieb2011strong]. 

Consequently, various new, unconditionally positive schemes have been introduced in recent years, see @burchard2003, @Bruggeman2007, @Broekhuizen2008, @Formaggia2011, @Ortleb2017, @kopeczmeister2018order2, @kopeczmeister2018order3, @huang2019order2, @huang2019order3, @OeffnerTorlo2020, @Martiradonna2020, @Avila2020, @Avila2021, @Blanes2022, @Zhu2024, @Izzo2025, and @Izgin2025. Among these, most of the literature is devoted to modified Patankar--Runge--Kutta (MPRK) methods.

Unfortunately, these new methods are not yet available in software packages, making them inaccessible to most users and limiting their comparability within the scientific community. PositiveIntegrators.jl aims at making these methods available and thus usable and comparable.


# Features

PositiveIntegrators.jl is written in Julia [@bezanson2017julia] and makes use of its strengths for scientific computing, e.g., ease of use and performance.
The package is fully compatible with DifferentialEquations.jl [@rackauckas2017differentialequations] and therefore many features that are available there can be used directly. In particular, this allows a direct comparison of the provided methods and standard schemes. Moreover, it integrates well with the Julia ecosystem, e.g., by making it simple to visualize numerical solutions using dense output in Plots.jl [@christ2023plots].

The package offers implementations of conservative as well as non-conservative production-destruction systems (PDS), which are the building blocks for the solution of differential equations with MPRK schemes. Furthermore, conversions of these PDS to standard `ODEProblem`s from DifferentialEquations.jl are provided.

Currently, the package contains the following methods:

- The MPRK methods `MPE`, `MPRK22`, `MPRK43I`, and `MPRK43II` of @kopeczmeister2018order2 and @kopeczmeister2018order3 are based on the classical formulation of Runge--Kutta schemes and have accuracies from first to third order.
- The MPRK methods `SSPMPRK22` and `SSPMPRK43` of @huang2019order2 and @huang2019order3 are based on the SSP formulation of Runge--Kutta schemes and are of second and third order, respectively. 
- The `MPDeC` methods of @OeffnerTorlo2020 combine the deferred correction approach with the idea of MPRK schemes to obtain schemes of arbitrary order. In the package methods from second up to 10th order are implemented.

In addition, all implemented methods have been extended so that non-conservative and non-autonomous PDS can be solved as well. Furthermore, adaptive step size control is available for almost all schemes.

# Related research and software

The first MPRK methods were introduced by @burchard2003. These are the first-order scheme `MPE` and a second-order scheme based on Heun's method. To avoid the restriction to Heun's method, the second-order `MPRK22` schemes were developed by @kopeczmeister2018order2. The techniques developed therein also enabled a generalization to third-order schemes and thus the introduction of `MPRK43I` and `MPRK43II` methods by @kopeczmeister2018order3.

The aforementioned schemes were derived from the classical formulation of Runge-Kutta methods. Using the Shu-Osher formulation instead lead to the introduction of the second-order schemes `SSPMPRK22` by @huang2019order2 and the third-order scheme `SSPMPRK43` by @huang2019order3.

Starting from a low-order method, the deferred correction approach can be used to increase the method's approximation order iteratively. @OeffnerTorlo2020 combined deferred correction with the MPRK idea to devise MPRK schemes of arbitrary order. These are implemented as `MPDeC` schemes for orders 2 up to 10.

The implemented methods were originally introduced for conservative production-destruction systems only. An extension to non-conservative production-destruction systems was presented by @benzmeister2015. We implemented a modification of this algorithm, by treating the non-conservative production and destruction terms separately, weighting the destruction terms and leaving the production terms unweighted.

Readers interested in additional theoretical background, further properties of the implemented schemes, and some applications are referred to the publications of @kopeczmeister2019, @izgin2022stability1, @izgin2022stability2, @huang2023, @torlo2022, and @izginoeffner2023. PositiveIntegrators.jl was successfully applied in the work of @bartel2024structure to solve Fokker-Planck equations, ensuring the positivity of the unknown quantities.

Existing software libraries do not have a strong focus on unconditional positivity and, to the authors' knowledge, there is no other software library which offers MPRK schemes. 
A common strategy to obtain nonnegative solutions used in the `PositiveDomain` callback of DifferentialEquations.jl or the commercial package MATLAB is described by @Shampine2005. In this approach negative components of approximate solutions that have been accepted by the adaptive time stepping algorithm are set to zero.
Another possibility is to reduce the chosen time step size beyond accuracy considerations until a nonnegative approximation is calculated. This can be achieved in DifferentialEquations.jl using the solver option `isoutofdomain`.

We also mention that some papers on MPRK schemes offer supplementary codes. However, these are mainly small scripts for the reproduction of results shown in the papers and are not intended as software libraries.


# Acknowledgements

JL acknowledges the support by the Deutsche Forschungsgemeinschaft (DFG)
within the Research Training Group GRK 2583 "Modeling, Simulation and
Optimization of Fluid Dynamic Applications".
HR was supported by
the German Research Foundation (Deutsche Forschungsgemeinschaft DFG, project number 513301895) and
the Daimler und Benz Stiftung (Daimler and Benz foundation, project number 32-10/22).


# References
