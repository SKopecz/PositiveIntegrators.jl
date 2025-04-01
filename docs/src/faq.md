# Troubleshooting and frequently asked questions

## Sparse matrices

You can use sparse matrices for the linear systems arising in
[PositiveIntegrators.jl](https://github.com/NumericalMathematics/PositiveIntegrators.jl),
as described, e.g., in the [tutorial on linear advection](@ref tutorial-linear-advection).
However, you need to make sure that you do not change the sparsity pattern
of the production term matrix since we assume that the structural nonzeros
are kept fixed. This is a [known issue](https://github.com/JuliaSparse/SparseArrays.jl/issues/190).
For example, you should avoid something like

```@repl
using SparseArrays
p = spdiagm(0 => ones(4), 1 => zeros(3))
p .= 2 * p
```

Instead, you should be able to use a pattern like the following, where the function `nonzeros` is used to modify the values of a sparse matrix.

```@repl
using SparseArrays
p = spdiagm(0 => ones(4), 1 => zeros(3))
for j in axes(p, 2)
    for idx in nzrange(p, j)
        i = rowvals(p)[idx]
        nonzeros(p)[idx] = 10 * i + j # value p[i, j]
    end
end; p
```
