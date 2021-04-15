module SphericalHarmonicArrays
using OffsetArrays
import Base: tail, @propagate_inbounds
using SphericalHarmonicModes
import SphericalHarmonicModes: ModeRange

export SHArray

include("errors.jl")

const RangeOrInteger = Union{ModeRange, AbstractUnitRange, Integer}
const OneBasedAxisType = Union{ModeRange, Integer, Base.OneTo}
const DimOrInd = Union{Integer, AbstractUnitRange}
const RangeOrModeRange = Union{AbstractUnitRange, ModeRange}
const ArrayInitializer = Union{UndefInitializer, Missing, Nothing}

# Methods from Unrolled.jl
type_length(tup::Type{T}) where {T<:Tuple} = length(tup.parameters)
function _unrolled_filter(f, tup)
    :($([Expr(:(...), :(f(tup[$i]) ? (tup[$i],) : ()))
         for i in 1:type_length(tup)]...),)
end
function _unrolled_filterdims(f, tup)
    :($([Expr(:(...), :(f(tup[$i]) ? ($i,) : ()))
         for i in 1:type_length(tup)]...),)
end
@generated unrolled_filter(f, tup) = _unrolled_filter(f, tup)
@generated unrolled_filterdims(f, tup) = _unrolled_filterdims(f, tup)

"""
    SHArray(arr::AbstractArray{T,N}, modes::NTuple{N, Union{AbstractUnitRange, SphericalHarmonicModes.ModeRange}}) where {T,N}

Create a wrapper around an array such that certain dimensions may be indexed using
`Tuple` of spherical harmonic modes. Often the indices would be spherical harmonic
degrees `(l,m)`. The argument `modes` dictates the map between spherical harmonic modes and
the indices of the parent array. The indices of `modes` that are of a `ModeRange` type
correspond to the dimensions of the array that enable indexing with a `Tuple` of
spherical harmonic degrees.

Use the iterators provided by `SphericalHarmonicModes` in `modes` to generate the map between
the indices of the array and `Tuple`s of spherical harmonic degrees.
The resulting map would be of the form
`s[collect(modes[1])[i], collect(modes[2])[j], ...] == s[i,j, ...]`.

# Examples

```jldoctest; setup=:(using SphericalHarmonicModes)
julia> import SphericalHarmonicModes: LM

julia> s = SHArray(reshape(1:4, 2, 2), (LM(0:1, 0:0), 1:2))
2×2 SHArray(reshape(::UnitRange{Int64}, 2, 2), (LM(0:1, 0:0), 1:2)):
 1  3
 2  4

julia> collect(LM(0:1, 0:0))
2-element Vector{Tuple{Int64, Int64}}:
 (0, 0)
 (1, 0)

julia> s[(1,0), 1]
2
```
"""
struct SHArray{T, N, AA<:AbstractArray{T,N}, TM<:NTuple{N,RangeOrInteger}} <: AbstractArray{T, N}
    parent :: AA
    modes :: TM

    function SHArray{T,N,AA,TM}(arr::AA, modes::TM) where {T, N, AA<:AbstractArray{T,N}, TM<:NTuple{N,RangeOrInteger}}
        map(checkaxes, axes(arr), map(moderangeaxes, modes))
        new{T,N,AA,TM}(arr, modes)
    end
end

moderangeaxes(m::ModeRange) = axes(m, 1)
moderangeaxes(m) = to_axis(m) # convert size to axes
moderangelength(m::ModeRange) = length(m)
moderangelength(m) = m
to_axis(n::Integer) = Base.OneTo(n)
to_axis(n) = n
_uncolon(ax, m::Colon) = ax
_uncolon(ax, m) = m

function SHArray(arr::AbstractArray{T,N}, modes::NTuple{N,Union{Colon, RangeOrInteger}}) where {T,N}
    modes_uncolon = map(_uncolon, axes(arr), modes)
    SHArray{T, N, typeof(arr), typeof(modes_uncolon)}(arr, modes_uncolon)
end
SHArray(arr::AbstractArray{T,N}, modes::Vararg{Union{Colon, RangeOrInteger},N}) where {T,N} =
    SHArray(arr, modes)

"""
    SHArray{T}(init, modes::NTuple{N, Union{Integer, AbstractUnitRange{<:Integer}, SphericalHarmonicModes.ModeRange}}) where {T,N}
    SHArray{T,N}(init, modes::NTuple{N, Union{Integer, AbstractUnitRange{<:Integer}, SphericalHarmonicModes.ModeRange}}) where {T,N}
    SHArray{T}(init, modes::Vararg{N, Union{Integer, AbstractUnitRange{<:Integer}, SphericalHarmonicModes.ModeRange}}) where {T,N}
    SHArray{T,N}(init, modes::Vararg{N, Union{Integer, AbstractUnitRange{<:Integer}, SphericalHarmonicModes.ModeRange}}) where {T,N}

Return an `SHArray` wrapper around a parent array of the appropriate size with `N` dimensions and elements of type `T`,
such that the indices of `modes` that are of a `ModeRange` type may be indexed using
`Tuples` of spherical harmonic degrees.
The elements of the parent array are set according to the initializer `init`.

# Examples

```jldoctest; setup=:(using SphericalHarmonicModes)
julia> import SphericalHarmonicModes: LM

julia> SHArray{Missing}(undef, (LM(0:1, 0:0), 2))
2×2 SHArray(::$(Array{Missing,2}), (LM(0:1, 0:0), Base.OneTo(2))):
 missing  missing
 missing  missing
```
"""
function SHArray{T}(init::ArrayInitializer, modes::NTuple{N,RangeOrInteger}) where {T,N}
    SHArray{T,N}(init, modes)
end
function SHArray{T,N}(::UndefInitializer, modes::NTuple{N,RangeOrInteger}) where {T,N}
    arr = similar(Array{T,N}, map(moderangeaxes, modes))
    SHArray(arr, map(to_axis, modes))
end
function SHArray{T,N}(init::ArrayInitializer, modes::NTuple{N,RangeOrInteger}) where {T,N}
    S = SHArray{T,N}(undef, modes)
    fill!(parent(S), init)
    S
end

SHArray{T,N}(init::ArrayInitializer, modes::Vararg{RangeOrInteger,N}) where {T,N} = SHArray{T,N}(init, modes)
SHArray{T}(init::ArrayInitializer, modes::Vararg{RangeOrInteger,N}) where {T,N} = SHArray{T}(init, modes)

for (DT, f) in ((:zeros, :zero), (:ones, :one))
    @eval function Base.$DT(::Type{T}, modes::NTuple{N,RangeOrInteger}) where {T,N}
        s = SHArray{T,N}(undef, modes)
        fill!(parent(s), $f(T))
        s
    end
    @eval Base.$DT(::Type{T}, modes::Vararg{RangeOrInteger}) where {T} = $DT(T, modes)
    @eval Base.$DT(modes::Tuple{Vararg{RangeOrInteger}}) = $DT(Float64, modes)
    @eval Base.$DT(modes::Vararg{RangeOrInteger}) = $DT(Float64, modes)
end

# Convenience constructors
const SHArrayAllModeRange{T, N, AA<:AbstractArray{T,N}, TM<:NTuple{N,ModeRange}} = SHArray{T, N, AA, TM}
const SHVector{T, AA<:AbstractVector{T}, M<:Tuple{ModeRange}} = SHArrayAllModeRange{T, 1, AA, M}

# Add methods to Base functions
Base.parent(s::SHArray) = s.parent

Base.similar(arr::SHArray) = SHArray(similar(parent(arr)), modes(arr))
Base.similar(arr::SHArray, T::Type) = SHArray(similar(parent(arr), T), modes(arr))
Base.similar(arr::AbstractArray, modes::RangeOrInteger...) = similar(arr, eltype(arr), modes)
Base.similar(arr::AbstractArray, T::Type, modes::RangeOrInteger...) = similar(arr, T, modes)
Base.similar(arr::AbstractArray, T::Type, modes::Tuple{Vararg{RangeOrInteger}}) = SHArray(similar(parent(arr), T, map(moderangelength, modes)), modes)

Base.dataids(A::SHArray) = Base.dataids(parent(A)) # needed for fast broadcasting
function Broadcast.broadcast_unalias(dest::SHArray, src::SHArray)
    parent(dest) === parent(src) ? src : Broadcast.unalias(dest, src)
end

# Accessor methods
modes(s::SHArray) = s.modes

"""
    SphericalHarmonicArrays.shmodes(arr::SHArray)

Returns a `Tuple` containing the elements of `arr.modes` that are of a `SphericalHarmonicModes.ModeRange` type.
This is type-stable even if `arr.modes` contains inhomogeneous types.

# Examples
```jldoctest; setup=:(using SphericalHarmonicModes)
julia> import SphericalHarmonicModes: LM

julia> sa = SHArray{ComplexF64}(undef, (2,2));

julia> SphericalHarmonicArrays.shmodes(sa)
()

julia> sa = zeros(ComplexF64, (LM(1:1, 0:0), 2));

julia> SphericalHarmonicArrays.shmodes(sa)
(LM(1:1, 0:0),)

julia> sa = zeros(ComplexF64, (LM(1:1, 0:0), LM(2:2)));

julia> SphericalHarmonicArrays.shmodes(sa)
(LM(1:1, 0:0), LM(2:2, -2:2))
```

See also: [`shdims`](@ref)
"""
shmodes(b::SHArrayAllModeRange) = modes(b)
shmodes(b::SHArray) = unrolled_filter(x->isa(x, ModeRange), modes(b))

"""
    shdims(arr::SHArray)

Return the dimensions of `arr` that may be indexed using `ModeRange`s.
This is type-stable even if `arr.modes` contains inhomogeneous types.

# Examples
```jldoctest; setup=:(using SphericalHarmonicModes)
julia> import SphericalHarmonicModes: LM

julia> s = zeros(Float64, (LM(1:1,0:1), LM(0:0)));

julia> SphericalHarmonicArrays.shdims(s)
(1, 2)

julia> s = zeros(Float64, (LM(1:1,0:1), 2));

julia> SphericalHarmonicArrays.shdims(s)
(1,)
```

See also: [`shmodes`](@ref)
"""
shdims(b::SHArrayAllModeRange{<:Any,N}) where {N} = ntuple(identity, Val(N))
shdims(b::SHArray) = unrolled_filterdims(x->isa(x, ModeRange), modes(b))

Base.size(s::SHArray) = size(parent(s))
Base.axes(s::SHArray) = axes(parent(s))

# Indexing

Base.IndexStyle(::Type{SA}) where {SA<:SHArray} = IndexStyle(parenttype(SA))
parenttype(::Type{<:SHArray{<:Any,<:Any,AA}}) where {AA} = AA
parenttype(A::SHArray) = parenttype(typeof(A))

const ModeRangeIndexType = Union{Tuple{Integer,Integer},ModeRange,
                            Tuple{AbstractUnitRange{<:Integer},
                            AbstractUnitRange{<:Integer}}}

# use modes instead of axes for indexing in general, so that we may convert tuples of modes to indices
@inline Base.to_indices(s::SHArray, I::Tuple) = to_indices(s, modes(s), I)
@inline Base.to_indices(s::SHVector, I::Tuple{ModeRangeIndexType}) = to_indices(s, modes(s), I)

# methods to avoid ambiguity
@inline Base.to_indices(A::SHArray, I::Tuple{Vararg{Int}}) = I
@inline Base.to_indices(A::SHArray, I::Tuple{Vararg{Integer}}) = to_indices(parent(A), I)
@inline Base.to_indices(A::SHArray, ::Tuple{}) = ()
@inline Base.to_indices(s::SHArray, I::Tuple{Any}) = to_indices(parent(s), I)

@inline Base.to_indices(s::SHArray, I::Tuple{Vararg{Union{Integer, CartesianIndex}}}) =
    to_indices(parent(s), I)

@inline function Base.to_indices(s::SHArray, m::Tuple{ModeRange,Vararg{Any}},
    I::Tuple{ModeRangeIndexType,Vararg{Any}})
    (modeindex(first(m), first(I)), to_indices(s, tail(m), tail(I))...)
end

@inline function Base.uncolon(inds::Tuple{ModeRange,Vararg{Any}}, I::Tuple{Colon, Vararg{Any}})
    Base.uncolon(axes(first(inds)), I)
end

# getindex
@propagate_inbounds function Base.getindex(s::SHArray, I...)
    parent(s)[to_indices(s, I)...]
end

# setindex
@propagate_inbounds function Base.setindex!(s::SHArray, val, I...)
    parent(s)[to_indices(s, I)...] = val
    s
end

# Broadcasting
Base.BroadcastStyle(::Type{<:SHArray}) = Broadcast.ArrayStyle{SHArray}()

function assert_modes_same(a,b)
    a == b || throw(ModeMismatchError(a,b))
end
@inline function assert_leading_modes_compatible(a::Tuple, b::Tuple)
    assert_modes_same(first(a) ,first(b))
    assert_leading_modes_compatible(tail(a), tail(b))
end
@inline assert_leading_modes_compatible(a::Tuple{},b::Tuple) = nothing
@inline assert_leading_modes_compatible(a::Tuple,b::Tuple{}) = nothing
@inline assert_leading_modes_compatible(a::Tuple{},b::Tuple{}) = nothing

function modes_larger(A::SHArray, B::SHArray)
    modes_A = modes(A)
    modes_out = modes_A
    (A === B) && return modes_out

    modes_B = modes(B)

    assert_leading_modes_compatible(modes_A, modes_B)

    if ndims(A) < ndims(B)
        modes_out = modes_B
    end

    return modes_out
end

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SHArray}},
    ::Type{ElType}) where ElType

    A = find_sharray(bc)
    modes_out = modes(A)

    # Check to make sure that all the SHArrays that are
    # being broadcasted over have compatible axes
    for B in Broadcast.flatten(bc).args
        if B isa SHArray
            (A === B) && continue
            modes_out = modes_larger(A,B)
        end
    end
    SHArray(similar(Array{ElType}, axes(bc)), modes_out)
end

find_sharray(bc::Base.Broadcast.Broadcasted) = find_sharray(bc.args)
find_sharray(args::Tuple) = find_sharray(find_sharray(args[1]), tail(args))
find_sharray(x) = x
find_sharray(a::SHArray, rest) = a
find_sharray(::Any, rest) = find_sharray(rest)

# show

function Base.showarg(io::IO, sa::SHArray, toplevel)
    print(io, "SHArray(")
    Base.showarg(io, parent(sa), false)
    print(io, ", ")
    print(io, modes(sa))
    print(io, ")")
end

function Base.replace_in_print_matrix(A::SHArray, i::Integer, j::Integer, s::AbstractString)
    Base.replace_in_print_matrix(parent(A), i, j, s)
end

end # module
