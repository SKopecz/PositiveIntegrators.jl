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

TODO


# Features

TODO


# Related research and software

TODO

@burchard2003
@kopeczmeister2018order2
@kopeczmeister2018order3
@huang2019order2
@huang2019order3
@benzmeister2015

@kopeczmeister2019
@izgin2022stability1
@izgin2022stability2
@huang2023
@torlo2022

@bartel2024structure


# Acknowledgements

Hendrik Ranocha was supported by
the German Research Foundation (Deutsche Forschungsgemeinschaft DFG, project number 513301895) and
the Daimler und Benz Stiftung (Daimler and Benz foundation, project number 32-10/22).


# References
