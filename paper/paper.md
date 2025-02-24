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

Many systems of ordinary differential equations that model real life applications possess positive solutions and it is quite natural to demand that numerical solutions of such systems should be positive as well. Unfortunately, positivty is a property that is not preserved by standard time integration schemes such as Runge--Kutta schemes, Rosenbrock methods or linear multistep methods. Therefore, various new, unconditionally positive schemes have been introduced in recent years. Unfortunately, these new methods are not freely available and therefore not applicable for most users and also not comparable within the scientific community.

Here we describe `PostiveIntegrators.jl` a julia package that provides efficient implementations of various positive time integration schemes and thus makes these methods usable and comparable.

# Statement of need

TODO

@bartel2024structure


# Features

TODO


# Related research and software

TODO


# Acknowledgements

Hendrik Ranocha was supported by
the German Research Foundation (Deutsche Forschungsgemeinschaft DFG, project number 513301895) and
the Daimler und Benz Stiftung (Daimler and Benz foundation, project number 32-10/22).


# References
