module PositiveIntegrators

# 1. Load dependencies
using LinearAlgebra: LinearAlgebra, Tridiagonal, I, diag, mul!
using Statistics: median

using SparseArrays: SparseArrays, AbstractSparseMatrix,
                    issparse, nonzeros, nzrange, rowvals, spdiagm
using StaticArrays: SVector, SMatrix, StaticArray, @SVector, @SMatrix, MMatrix

using FastBroadcast: @..
using MuladdMacro: @muladd
using SimpleUnPack: @unpack

using Reexport: @reexport

@reexport using SciMLBase: ODEProblem, init, solve

using SciMLBase: AbstractODEFunction, NullParameters, FullSpecialize,
                 isinplace

# TODO: Check imports and using statements below, reduce if possible
using OrdinaryDiffEqCore: OrdinaryDiffEqCore, OrdinaryDiffEqAlgorithm, ODESolution

using SymbolicIndexingInterface: SymbolicIndexingInterface

using LinearSolve: LinearSolve, LinearProblem, LUFactorization, solve!

import SciMLBase: interp_summary

using OrdinaryDiffEqCore: @cache,
                          OrdinaryDiffEqAdaptiveAlgorithm,
                          OrdinaryDiffEqConstantCache, OrdinaryDiffEqMutableCache,
                          False,
                          _vec
import OrdinaryDiffEqCore: alg_order, isfsal,
                           calculate_residuals, calculate_residuals!,
                           alg_cache, get_tmp_cache,
                           initialize!, perform_step!,
                           _ode_interpolant, _ode_interpolant!,
                           get_fsalfirstlast

using RecipesBase: @recipe

# 2. Export functionality defining the public API
export PDSFunction, PDSProblem
export ConservativePDSFunction, ConservativePDSProblem

export MPE, MPRK22, MPRK43I, MPRK43II
export SSPMPRK22, SSPMPRK43
export MPDeC

export prob_pds_linmod, prob_pds_linmod_inplace, prob_pds_nonlinmod,
       prob_pds_robertson, prob_pds_brusselator, prob_pds_sir,
       prob_pds_bertolazzi, prob_pds_npzd, prob_pds_stratreac, prob_pds_minmapk

export isnegative, isnonnegative
export work_precision_adaptive, work_precision_adaptive!, work_precision_fixed,
       work_precision_fixed!
export rel_max_error_overall, rel_max_error_tend, rel_l1_error_tend, rel_l2_error_tend

# 3. Load source code

# production-destruction systems
include("proddest.jl")

# modified Patankar-Runge-Kutta (MPRK) methods
include("mprk.jl")

# modified Patankar-Runge-Kutta based on the SSP formulation of RK methods (SSPMPRK)
include("sspmprk.jl")

# MPDeC methods
include("mpdec.jl")

# interpolation for dense output
include("interpolation.jl")

# predefined PDS problems
include("PDSProblemLibrary.jl")

# additional auxiliary functions
include("utilities.jl")

end # module
