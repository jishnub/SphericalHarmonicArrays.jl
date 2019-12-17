# SphericalHarmonicArrays

[![Build Status](https://travis-ci.com/jishnub/SphericalHarmonicArrays.jl.svg?branch=master)](https://travis-ci.com/jishnub/SphericalHarmonicArrays.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jishnub/SphericalHarmonicArrays.jl?svg=true)](https://ci.appveyor.com/project/jishnub/SphericalHarmonicArrays-jl)
[![Codecov](https://codecov.io/gh/jishnub/SphericalHarmonicArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jishnub/SphericalHarmonicArrays.jl)

Arrays to store spherical-harmonic coefficients, that may be indexed by modes as well as array indices. The type used for this is named `SHArray`, and aliases such as `SHVector` and `SHMatrix` are exported for covenience. `SHArray` is a wrapper around an underlying parent array, usually dense, that is indexed according to iterators that are specified while constructing the type. The arrays may have mixed axes, where certain axes are indexed using spherical harmonic modes whereas the others are indexed like the parent array.

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

julia> SHVector(modes) # will assign an array to store the correct number of modes, 3 in this case
3-element SHArray{Complex{Float64},1,Array{Complex{Float64},1},Tuple{LM},1}:
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
```

The default element type of the parent array is `ComplexF64`, however this may be specified in the constructor.

```julia
julia> SHVector{Int}(modes)
3-element SHArray{Int64,1,Array{Int64,1},Tuple{LM},1}:
 0
 0
 0
```

The parent array may be preallocated.

```julia
julia> v=ones(3);

julia> shv=SHVector(v,modes)
3-element SHArray{Float64,1,Array{Float64,1},Tuple{LM},1}:
 1.0
 1.0
 1.0

julia> v[1]=4 # update the parent array
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
julia> v=[1,2,3];shv=SHVector(v,LM(1:1))
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

julia> shv[(1,-1)]=6 # can set indices using modes, in this case it's specificed as an (l,m) pair
6

julia> shv
3-element SHArray{Int64,1,Array{Int64,1},Tuple{LM},1}:
  6
 56
 56
```

## SHMatrix

An `SHMatrix` is a 2D array with both axes storing spherical harmonic coefficients.

### Creating an SHMatrix

The constructors are similar to those of `SHVector`.

```julia
julia> SHMatrix(LM(1:2,1:1),LM(1:1)) # need to speficy two axes
2×3 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,LM},2}:
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im

# may combine different iterators as axes
julia> SHMatrix(LM(1:2,1:1),ML(1:1))
2×3 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,ML},2}:
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im
```

### Indexing

The matrix elements may be accessed with a combination of mode indices or the index style of the parent array.

```julia
julia> shm=SHMatrix(LM(1:2,1:1),LM(1:1))
2×3 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,LM},2}:
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im

# Linear indexing works if the parent array supports it
julia> for i in eachindex(shm)
       shm[i]=i
       end

julia> shm
2×3 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,LM},2}:
 1.0+0.0im  3.0+0.0im  5.0+0.0im
 2.0+0.0im  4.0+0.0im  6.0+0.0im

# Both axes may be indexed using modes
julia> shm[(1,1),(1,0)]
3.0 + 0.0im

# Any axis may be indexed using the corresponding mode
julia> shm[(1,1),2]
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
julia> sha = SHArray(zeros(1),LM(1:1,0:0))
1-element SHArray{Float64,1,Array{Float64,1},Tuple{LM},1}:
 0.0

# SHVector is an alias for a 1D SHArray that is indexed with modes
julia> sha isa SHVector
true

julia> SHArray(LM(1:1,0:0),1:2) # returns an OffsetArray
1×2 SHArray{Complex{Float64},2,OffsetArrays.OffsetArray{Complex{Float64},2,Array{Complex{Float64},2}},Tuple{LM,UnitRange{Int64}},1} with indices 1:1×1:2:
 0.0  0.0
```

If no parent array is specified and the first axis is a subtype of `AbstractArray`, the modes necessarily need to be passed as a `Tuple` to avoid ambiguity. This is because it's otherwise unclear if the first argument is the parent array or the first axis.

```julia
julia> SHArray((1:1,LM(1:1,0:1))) # second axis stores modes
1×2 SHArray{Complex{Float64},2,OffsetArrays.OffsetArray{Complex{Float64},2,Array{Complex{Float64},2}},Tuple{UnitRange{Int64},LM},1} with indices 1:1×1:2:
 0.0  0.0

julia> sha = SHArray(LM(1:1,0:0),ML(1:1,0:1)) # both axes stores modes
1×2 SHArray{Float64,2,Array{Float64,2},Tuple{LM,ML},2}:
 0.0  0.0

# SHMatrix is an alias for a 2D SHArray with both axes indexed with modes
julia> sha isa SHMatrix 
true
```

It is also possible to create an empty wrapper around an array. This is essentially equivalent to an array and is retained for completeness.
```julia
julia> SHArray(zeros(1,2)) # no modes, equivalent to an Array
1×2 SHArray{Float64,2,Array{Float64,2},Tuple{Base.OneTo{Int64},Base.OneTo{Int64}},0}:
 0.0  0.0
```

The arrays may have mixed axes, where some store spherical harmonic modes and some don't.

```julia
julia> sha = SHArray(LM(1:1,0:0),1:2,ML(0:1,0:0)) # mixed axes
1×2×2 SHArray{Complex{Float64},3,OffsetArrays.OffsetArray{Complex{Float64},3,Array{Complex{Float64},3}},Tuple{LM,UnitRange{Int64},ML},2} with indices 1:1×1:2×1:2:
[:, :, 1] =
 0.0  0.0

[:, :, 2] =
 0.0  0.0
```

### Indexing

Indexing is similar to `SHVector` and `SHMatrix`. The performance depends on the number of axes being indexed with modes.

```julia
julia> sha = SHArray((1:1, LM(1:1,0:1)))
1×2 SHArray{Complex{Float64},2,OffsetArrays.OffsetArray{Complex{Float64},2,Array{Complex{Float64},2}},Tuple{UnitRange{Int64},LM},1} with indices 1:1×1:2:
 0.0+0.0im  0.0+0.0im

julia> sha[1,(1,0)]=4 # first index
4

julia> sha[1,2]=5 # second index
5

julia> sha
1×2 SHArray{Complex{Float64},2,OffsetArrays.OffsetArray{Complex{Float64},2,Array{Complex{Float64},2}},Tuple{UnitRange{Int64},LM},1} with indices 1:1×1:2:
 4.0+0.0im  5.0+0.0im

julia> sha = SHArray(LM(1:1,0:0),1:2,ML(0:1,0:0));

julia> mode1=(1,0);mode2=(1,0); @btime $sha[$mode1,1,$mode2]
  19.842 ns (0 allocations: 0 bytes)
0.0 + 0.0im
```

## Broadcasting

`SHArray`s retain information about their modes upon broadcasting. If multiple `SHArray`s are involved in a broadcast operation, the result has the same axes as the one with the most dimensions. The dimensions being broadcasted over, if indexed with modes, have to exactly match for all the `SHArray`s involved in the operation.

```julia
julia> s = SHMatrix(LM(1:1,0:0),LM(1:1,-1:0));s .= 4
1×2 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,LM},2}:
 4.0+0.0im  4.0+0.0im

julia> s + s
1×2 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,LM},2}:
 8.0+0.0im  8.0+0.0im

julia> s .* s
1×2 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,LM},2}:
 16.0+0.0im  16.0+0.0im

julia> sv = SHVector(first(modes(s)));sv .= 6;

julia> s .* sv # Leading dimensions of s and sv are the same
1×2 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,LM},2}:
 24.0+0.0im  24.0+0.0im
```

Broadcasting operations might be slow, so watch out for performance drops.

```julia
julia> sm = SHMatrix(LM(1:1,0:0),LM(1:1,-1:0));a = zeros(size(sm));oa = zeros(map(UnitRange,axes(sm)));

# Arrays are the fastest
julia> @btime @. $a + $a;
  37.537 ns (1 allocation: 96 bytes)

 # OffsetArrays are less performant
julia> @btime @. $oa + $oa;
  87.452 ns (4 allocations: 224 bytes)

# SHMatrices are comparable
julia> @btime @. $sm + $sm;
  80.524 ns (3 allocations: 240 bytes)

julia> sa = SHArray(zeros(1:1,1:2),(LM(1:1,0:0),LM(1:1,-1:0)));

# SHArrays that use an OffsetArray as the parent are slower
julia> @btime @. $sa + $sa;
  159.311 ns (7 allocations: 400 bytes)

# We may operate on the underlying array to regain performance, if the axes permit this.
julia> @btime parent(parent($sa)) .+ parent(parent($sa));
  37.295 ns (1 allocation: 96 bytes)
```
