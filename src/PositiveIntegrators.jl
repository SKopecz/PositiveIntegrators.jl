module PositiveIntegrators

# 1. Load dependencies
using LinearAlgebra: LinearAlgebra, Tridiagonal, I, diag, diagind, mul!
using SparseArrays: SparseArrays, AbstractSparseMatrix
using StaticArrays: SVector, MVector, SMatrix, StaticArray, @SVector, @SMatrix

using FastBroadcast: @..
using Kwonly: @add_kwonly
using MuladdMacro: @muladd
using SimpleUnPack: @unpack

using Reexport: @reexport

@reexport using SciMLBase: ODEFunction, ODEProblem, init, solve

using SciMLBase: AbstractODEFunction, NullParameters, FullSpecialize, NoSpecialize,
                 isinplace

# TODO: Check imports and using statements below, reduce if possible
using OrdinaryDiffEq: OrdinaryDiffEq, OrdinaryDiffEqAlgorithm

using SymbolicIndexingInterface

using LinearSolve: LinearSolve, LinearProblem, LUFactorization

using SciMLBase: DEFAULT_OBSERVED
import SciMLBase: interp_summary,
                  __has_mass_matrix, __has_analytic, __has_tgrad,
                  __has_jac, __has_jvp, __has_vjp, __has_jac_prototype,
                  __has_sparsity, __has_Wfact, __has_Wfact_t,
                  __has_paramjac, __has_syms, __has_indepsym, __has_paramsyms,
                  __has_observed, __has_colorvec, __has_sys

using OrdinaryDiffEq: @cache,
                      DEFAULT_PRECS,
                      OrdinaryDiffEqAdaptiveAlgorithm,
                      OrdinaryDiffEqConstantCache, OrdinaryDiffEqMutableCache,
                      False,
                      recursivefill!, _vec, wrapprecs, dolinsolve
import OrdinaryDiffEq: alg_order, isfsal,
                       calculate_residuals, calculate_residuals!,
                       alg_cache, initialize!, perform_step!,
                       _ode_interpolant, _ode_interpolant!

# 2. Export functionality defining the public API
export PDSFunction, PDSProblem
export ConservativePDSFunction, ConservativePDSProblem

export MPE, MPRK22

export prob_pds_linmod, prob_pds_linmod_inplace, prob_pds_nonlinmod,
       prob_pds_robertson, prob_pds_brusselator, prob_pds_sir,
       prob_pds_bertolazzi, prob_pds_npzd, prob_pds_stratreac

# 3. Load source code

# production-destruction systems
include("proddest.jl")

# modified Patankar-Runge-Kutta (MPRK) methods
include("mprk.jl")

# predefined PDS problems
include("PDSProblemLibrary.jl")

end # module
