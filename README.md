# SphericalHarmonicArrays

![CI](https://github.com/jishnub/SphericalHarmonicArrays.jl/workflows/CI/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/jishnub/SphericalHarmonicArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jishnub/SphericalHarmonicArrays.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jishnub.github.io/SphericalHarmonicArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/SphericalHarmonicArrays.jl/dev)

Arrays to store spherical-harmonic coefficients, that may be indexed by modes as well as array indices. The type used for this is named `SHArray`, and the aliases `SHVector` and `SHMatrix` are exported for covenience. `SHArray` is a wrapper around an underlying parent array, usually dense, that is indexed according to iterators that are specified while constructing the type. The arrays may have mixed axes, where certain axes are indexed using spherical harmonic modes whereas the others are indexed like the parent array.

## Getting Started

### Installing

```julia
julia> ]
pkg> add SphericalHarmonicArrays

julia> using SphericalHarmonicArrays
```

## Usage

This package uses iterators from [SphericalHarmonicModes.jl](https://github.com/jishnub/SphericalHarmonicModes.jl) for indexing. All the iterators available in the package may be used as axes. Take a look at that package to understand more about the iterators being used here.

## SHVector

An `SHVector` is a 1D array indexed using spherical harmonic modes. 

### Creating an SHVector

The simplest constructor assigns the array automatically based on the number of modes specified.

```julia
julia> modes=LM(1:1)
Spherical harmonic modes with l increasing faster than m
(l_min = 1, l_max = 1, m_min = -1, m_max = 1)

julia> SHVector{Float64}(modes)
3-element SHArray(::Array{Float64,1}, (LM(1:1, -1:1),)):
 0.0
 0.0
 0.0
```

The parent array may be preallocated.

```julia
julia> v = ones(3);

julia> shv = SHVector(v,modes)
3-element SHArray(::Array{Float64,1}, (LM(1:1, -1:1),)):
 1.0
 1.0
 1.0

julia> v[1] = 4 # update the parent array
4

julia> shv # updated as well
3-element SHArray{Float64,1,Array{Float64,1},Tuple{LM},1}:
 4.0
 1.0
 1.0
```

### Indexing

An `SHVector` may be indexed either linearly as a normal vector, or using the modes that are stored in the array. Linear indexing is faster as this simply passes the indices to the parent, so this is what should be used if all the indices of the array are being iterated over. Modes need to be specified as a tuple of integers, eg. `(l,m)`, corresponding to the type of axis iterator that was used to create the `SHVector`.

```julia
julia> v = [1,2,3]; shv=SHVector(v, LM(1:1))
3-element SHArray{Int64,1,Array{Int64,1},Tuple{LM},1}:
 1
 2
 3

julia> shv[2]
2

julia> shv[(1,0)] # indexed using (l,m)
2

julia> @btime $shv[2] # Linear indexing
  2.026 ns (0 allocations: 0 bytes)
2

julia> mode=(1,0); @btime $shv[$mode] # Indexing using modes
  9.774 ns (0 allocations: 0 bytes)
2

julia> @. shv = 56 # broadcasting works as expected
3-element Array{Int64,1}:
 56
 56
 56

julia> shv[(1,-1)] = 6 # can set indices using modes, in this case it's specificed as an (l,m) pair
6

julia> shv
3-element SHArray{Int64,1,Array{Int64,1},Tuple{LM},1}:
  6
 56
 56
```

Note that in this specific case it is more efficient to define the mode range as 
`LM(SingleValuedRange(1))`. Using this, we obtain 

```julia
julia> shv = SHVector(v, LM(SingleValuedRange(1)));

julia> mode=(1,0); @btime $shv[$mode]
  3.734 ns (0 allocations: 0 bytes)
2
```

Indexing operations are significantly more performant for arrays constructed using these special ranges.

## SHMatrix

An `SHMatrix` is a 2D array with both axes storing spherical harmonic coefficients.

### Creating an SHMatrix

The constructors are similar to those of `SHVector`.

```julia
julia> SHMatrix{Float64}(LM(1:2, 1:1),LM(1:1))
2×3 SHArray(::Array{Float64,2}, (LM(1:2, 1:1), LM(1:1, -1:1))):
 0.0  0.0  0.0
 0.0  0.0  0.0

# may combine different iterators as axes
julia> SHMatrix{Float64}(LM(1:2, 1:1),ML(1:1))
2×3 SHArray(::Array{Float64,2}, (LM(1:2, 1:1), ML(1:1, -1:1))):
 0.0  0.0  0.0
 0.0  0.0  0.0
```

### Indexing

The matrix elements may be accessed with a combination of mode indices or the index style of the parent array.

```julia
julia> shm = SHMatrix{Float64}(LM(1:2,1:1), LM(1:1))
2×3 SHArray(::Array{Float64,2}, (LM(1:2, 1:1), LM(1:1, -1:1))):
 0.0  0.0  0.0
 0.0  0.0  0.0

# Linear indexing works if the parent array supports it
julia> for i in eachindex(shm)
       shm[i]=i
       end

julia> shm
2×3 SHArray(::Array{Float64,2}, (LM(1:2, 1:1), LM(1:1, -1:1))):
 1.0  3.0  5.0
 2.0  4.0  6.0

# Both axes may be indexed using modes
julia> shm[(1,1),(1,0)]
3.0

# Any axis may be indexed using the corresponding mode
julia> shm[(1,1), 2]
3.0 + 0.0im

# May use any combination of Cartesian and fancy mode indexing
julia> shm[1,2] == shm[(1,1),2] == shm[1,(1,0)] ==shm[(1,1),(1,0)]
true

julia> mode1=(1,1);mode2=(1,0); @btime $shm[$mode1,$mode2] # twice as expensive as SHVector
  19.026 ns (0 allocations: 0 bytes)
3.0 + 0.0im
```

## SHArray

This is the most general type of an array with any axis possibly being indexed using a collection of modes. This differs from the previously described aliases as some (or all) axes need not be indexed using modes.

### Creating an SHArray

```julia
julia> sha = SHArray(zeros(1),(LM(1:1,0:0),))
1-element SHArray(::Array{Float64,1}, (LM(1:1, 0:0),)):
 0.0

# SHVector is an alias for a 1D SHArray that is indexed with modes
julia> sha isa SHVector
true

julia> SHArray{Float64}((LM(1:1,0:0),1:2)) # supports OffsetArrays
1×2 SHArray(OffsetArray(::Array{Float64,2}, 1:1, 1:2), (LM(1:1, 0:0), 1:2)) with indices 1:1×1:2:
 0.0  0.0
```

The arrays may have mixed axes, where some store spherical harmonic modes and some don't.

```julia
julia> sha = SHArray{Float64}((LM(1:1,0:0),-1:1,ML(0:1,0:0)))
1×3×2 SHArray(OffsetArray(::Array{Float64,3}, 1:1, -1:1, 1:2), (LM(1:1, 0:0), -1:1, ML(0:1, 0:0))) with indices 1:1×-1:1×1:2:
[:, :, 1] =
 0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0
```

### Indexing

Indexing is similar to `SHVector` and `SHMatrix`.

```julia
julia> SHArray{Float64}((1:1, LM(1:1,0:1)))
1×2 SHArray(OffsetArray(::Array{Float64,2}, 1:1, 1:2), (1:1, LM(1:1, 0:1))) with indices 1:1×1:2:
 0.0  0.0

julia> sha[1,(1,0)] = 4 # first index
4

julia> sha[1,2] = 5 # second index
5

julia> sha
1×2 SHArray(OffsetArray(::Array{Float64,2}, 1:1, 1:2), (1:1, LM(1:1, 0:1))) with indices 1:1×1:2:
 4.0  5.0
```

## Broadcasting

`SHArray`s retain information about their modes upon broadcasting. If multiple `SHArray`s are involved in a broadcast operation, the result has the same axes as the one with the most dimensions. The dimensions being broadcasted over, if indexed with modes, have to exactly match for all the `SHArray`s involved in the operation.

```julia
julia> s = SHMatrix{Float64}(LM(1:1,0:0),LM(1:1,-1:0)); s .= 4
1×2 SHArray(::Array{Float64,2}, (LM(1:1, 0:0), LM(1:1, -1:0))):
 4.0  4.0

julia> s + s
1×2 SHArray(::Array{Float64,2}, (LM(1:1, 0:0), LM(1:1, -1:0))):
 8.0  8.0

julia> s .* s
1×2 SHArray(::Array{Float64,2}, (LM(1:1, 0:0), LM(1:1, -1:0))):
 16.0  16.0

julia> sv = SHVector{Float64}(first(SphericalHarmonicArrays.modes(s))); sv .= 6;

julia> s .* sv # Leading dimensions of s and sv are the same
1×2 SHArray(::Array{Float64,2}, (LM(1:1, 0:0), LM(1:1, -1:0))):
 24.0  24.0
```

Broadcasting operations might be slow, so watch out for performance drops.