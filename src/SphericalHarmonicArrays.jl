module SphericalHarmonicArrays
using Reexport, OffsetArrays
import Base: tail, @propagate_inbounds

@reexport using SphericalHarmonicModes
import SphericalHarmonicModes: ModeRange, modeindex, 
l_range, m_range,l₁_range,l₂_range

export SHArray, SHVector, SHMatrix
export modes,shmodes, shdims

include("errors.jl")

const AxisType = Union{ModeRange,AbstractUnitRange}
const ArrayInitializer = Union{UndefInitializer, Missing, Nothing}

"""
	SHArray(arr::AbstractArray{T,N}, modes::Tuple, shdims::NTuple{NSH, Int}) where {T,N,NSH}

Create a wrapper around an array such that the dimensions specified by `shdims` may be 
indexed using a `Tuple` representing a collection of spherical harmonic degrees. Often this 
would be a spherical harmonic mode represented by `(l,m)`.

Use the iterators provided by `SphericalHarmonicModes` in `modes` to generate the map between 
the indices of the array and `Tuple`s of spherical harmonic degrees.

# Examples

```jldoctest
julia> s = SHArray(reshape(1:4, 2, 2), (LM(0:1,0:0), 2))
2×2 SHArray{Int64,2,Base.ReshapedArray{Int64,2,UnitRange{Int64},Tuple{}},Tuple{LM,Base.OneTo{Int64}},1}:
 1  3
 2  4

julia> s.modes[1] |> collect
2-element Array{Tuple{Int64,Int64},1}:
 (0, 0)
 (1, 0)

julia> s[(1,0), 1]
2
```

See also: [`SHVector`](@ref), [`SHMatrix`](@ref)
"""
struct SHArray{T,N,AA<:AbstractArray{T,N},
	TM<:Tuple{Vararg{AxisType,N}},NSH} <: AbstractArray{T,N}

	parent :: AA
	modes :: TM
	shdims :: NTuple{NSH,Int}

	function SHArray{T,N,AA,TM,NSH}(arr::AA, modes::TM, shdims::NTuple{NSH,Int}) where {T,N,AA,TM,NSH}

		NSH > N && throw(MismatchedDimsError(NSH,N))

		for (dim,mode) in enumerate(modes)
			if dim in shdims && !isa(mode,ModeRange)
				throw(UnexpectedAxisTypeError(dim,ModeRange,typeof(mode)))
			elseif !(dim in shdims) && mode isa ModeRange
				throw(UnexpectedAxisTypeError(dim,typeof(axes(arr,dim)),typeof(mode)))
			end 
			if size(arr,dim) != length(mode)
				if dim in shdims
					throw(SizeMismatchArrayModeError(dim,size(arr,dim),length(mode)))
				else
					throw(SizeMismatchError(dim,size(arr,dim),length(mode)))
				end
			end
		end
		new{T,N,AA,TM,NSH}(arr,modes,shdims)
	end
end

# Constructors
to_axis(n::Int) = Base.OneTo(n)
to_axis(n) = n

function SHArray(arr::AbstractArray{T,N},modes::NTuple{N,AxisType},
	shdims::NTuple{N,Int}) where {T,N}
	
	SHArray{T,N,typeof(arr),typeof(modes),N}(arr,modes,shdims)
end

function SHArray(arr::AbstractArray{T,N},modes::NTuple{N,AxisType},
	shdims::NTuple{NSH,Int}) where {T,N,NSH}

	SHArray{T,N,typeof(arr),typeof(modes),NSH}(arr,modes,shdims)
end

function SHArray(arr::AbstractArray{T,N},modes::Tuple{Vararg{AxisType,NSH}},
	shdims::NTuple{NSH,Int}) where {T,N,NSH}

	allmodes = Vector{AxisType}(undef,N)
	allmodes .= axes(arr)
	for ind in 1:NSH
		d,m = shdims[ind],modes[ind]
		allmodes[d] = m
	end
	modes = Tuple(allmodes)

	SHArray(arr,modes,shdims)
end

function SHArray(arr::AbstractArray{<:Any,N}, modes::NTuple{N,Union{Int,AxisType}}) where {N}
	shdims = Tuple(dim for dim in eachindex(modes) if modes[dim] isa ModeRange)
	SHArray(arr, map(to_axis, modes), shdims)
end

SHArray(arr::AbstractArray) = SHArray(arr,axes(arr))
SHArray(arr::AbstractVector, mode::ModeRange) = SHArray(arr,(mode,))
SHArray(arr::AbstractVector, mode::ModeRange,shdims::Tuple{Int}) = SHArray(arr,(mode,),shdims)

# These constructors allocate an array of an appropriate size
@inline moderangeaxes(m::ModeRange) = axes(m,1)
@inline moderangeaxes(m) = m

"""
	SHArray{T}(modes::Tuple, [shdims::Tuple])

Return an `SHArray` wrapper around a parent array of the appropriate size with elements of type `T`, 
such that the dimensions specified by `shdims` may be indexed using `Tuple`s of 
spherical harmonic degrees. The elements of the parent array are set to zero by default.

	SHArray(modes::Tuple, [shdims::Tuple])

Return an `SHArray` of the appropriate size and element type `ComplexF64`.

# Examples

```jldoctest
julia> s = SHArray((LM(0:1,0:0), 2))
2×2 SHArray{Complex{Float64},2,Array{Complex{Float64},2},Tuple{LM,Base.OneTo{Int64}},1}:
 0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im
```
"""
function SHArray{T}(modes::Tuple{Vararg{AxisType}},shdims::Tuple{Vararg{Int}}) where {T}
	ax = Tuple(moderangeaxes(m) for m in modes)
	SHArray(zeros(T,ax), map(to_axis, modes), shdims)
end
function SHArray(modes::Tuple{Vararg{AxisType}},shdims::Tuple{Vararg{Int}})
	SHArray{ComplexF64}(modes,shdims)
end

function SHArray{T}(mode::ModeRange,shdims::Tuple{Int}) where {T}
	SHArray(zeros(T,moderangeaxes(mode)),(mode,),shdims)
end
SHArray(mode::ModeRange,shdims::Tuple{Int}) = SHArray{ComplexF64}(mode,shdims)

function SHArray{T}(modes::NTuple{N,Any}) where {T,N}
	ax = ntuple(i->moderangeaxes(modes[i]), Val(N))
	SHArray(zeros(T,ax), map(to_axis, modes))
end
SHArray(modes::Tuple) = SHArray{ComplexF64}(modes)

SHArray{T}(modes::Vararg{Union{AxisType,Int}}) where {T} = SHArray{T}(modes)
SHArray(modes::Union{Int,AxisType}, args::Union{Int,AbstractUnitRange}...) = SHArray((modes,args...))

# undef, missing and nothing initializers

function SHArray{T,N}(init::ArrayInitializer,
	modes::Tuple{Vararg{AxisType,N}},args...) where {T,N}
	
	arr = OffsetArray{T,N}(init,map(moderangeaxes,modes))
	SHArray(arr,modes,args...)
end

function SHArray{T}(init::ArrayInitializer,
	modes::Tuple{Vararg{AxisType,N}},args...) where {T,N}
	SHArray{T,N}(init,modes,args...)
end

function SHArray{T,N}(init::ArrayInitializer,modes::Vararg{AxisType,N}) where {T,N}
	arr = OffsetArray{T,N}(init,map(moderangeaxes,modes))
	SHArray(arr,modes)
end

function SHArray{T}(init::ArrayInitializer,modes::Vararg{AxisType,N}) where {T,N}
	SHArray{T,N}(init,modes)
end
SHArray(init::ArrayInitializer,args...) = SHArray{ComplexF64}(init,args...)

# Convenience constructors
const SHArrayOneAxis{T,N,AA,M} = SHArray{T,N,AA,M,1}
const SHArrayOnlyFirstAxis{T,N,AA,M<:Tuple{ModeRange,Vararg{<:AbstractUnitRange}}} = SHArrayOneAxis{T,N,AA,M}
const SHVector{T,AA,M<:Tuple{ModeRange}} = SHArrayOnlyFirstAxis{T,1,AA,M}

"""
	SHVector(arr::AbstractVector, mode::ModeRange)

Return a wrapper that maps the indices of the parent vector to a `Tuple` of spherical harmonic degrees.
Often the `Tuple` would be a pair of spherical harmonic modes `(l,m)`.

# Examples

```jldoctest
julia> s = SHVector(1:3, LM(1:1))
3-element SHArray{Int64,1,UnitRange{Int64},Tuple{LM},1}:
 1
 2
 3

julia> s.modes[1] |> collect
3-element Array{Tuple{Int64,Int64},1}:
 (1, -1)
 (1, 0)
 (1, 1)

julia> s[(1,0)]
2
```

See also: [`SHMatrix`](@ref), [`SHArray`](@ref)
"""
SHVector(arr::AbstractVector,mode::ModeRange) = SHArray(arr,(mode,),(1,))
SHVector(arr::AbstractVector,modes::Tuple{ModeRange}) = SHArray(arr,modes,(1,))

# Automatically allocate a vector of an appropriate size
SHVector{T}(mode::ModeRange) where {T} = SHArray(zeros(T,moderangeaxes(mode)),(mode,),(1,))
SHVector(mode::ModeRange) = SHArray{ComplexF64}(mode)

# undef, missing and nothing initializers
function SHVector{T}(init::ArrayInitializer,modes::Tuple{ModeRange}) where {T}
	arr = Vector{T}(init,length(modes[1]))
	SHVector(arr,modes)
end
function SHVector{T}(init::ArrayInitializer,modes::ModeRange) where {T}
	arr = Vector{T}(init,length(modes))
	SHVector(arr,modes)
end
SHVector(init::ArrayInitializer,modes) = SHVector{ComplexF64}(init,modes)

const SHMatrix{T,AA<:AbstractMatrix{T},M<:Tuple{ModeRange,ModeRange}} = SHArray{T,2,AA,M,2}

"""
	SHMatrix(arr::AbstractMatrix, modes::NTuple{2,ModeRange})
	SHMatrix(arr::AbstractMatrix, ax1::ModeRange, ax2::ModeRange)

Return a wrapper that maps the indices of the parent matrix along each axis 
to a `Tuple` of spherical harmonic degrees.
Often these `Tuple`s would be pairs of spherical harmonic modes `(l,m)`.

# Examples

```jldoctest
julia> s = SHMatrix(reshape(1:4, 2,2), (LM(0:1, 0:0), LM(1:2,0:0)))
2×2 SHArray{Int64,2,Base.ReshapedArray{Int64,2,UnitRange{Int64},Tuple{}},Tuple{LM,LM},2}:
 1  3
 2  4

julia> s[(1,0),(2,0)]
4
```

See also: [`SHVector`](@ref), [`SHArray`](@ref)
"""
SHMatrix(arr::AbstractMatrix, modes::NTuple{2,ModeRange}) = SHArray(arr,modes,(1,2))
SHMatrix(arr::AbstractMatrix, modes::Vararg{ModeRange,2}) = SHArray(arr,modes,(1,2))
SHMatrix{T}(modes::NTuple{2,ModeRange}) where {T} = 
	SHArray(zeros(T,map(moderangeaxes,modes)),modes,(1,2))
SHMatrix{T}(modes::Vararg{ModeRange,2}) where {T} = SHMatrix{T}(modes)
SHMatrix(modes::NTuple{2, ModeRange}) = SHMatrix{ComplexF64}(modes)
SHMatrix(modes::Vararg{ModeRange,2}) = SHMatrix(modes)

# undef, missing and nothing initializers
function SHMatrix{T}(init::ArrayInitializer,modes::Tuple{ModeRange,ModeRange}) where {T}
	arr = Matrix{T}(init,map(length,modes))
	SHMatrix(arr,modes)
end
function SHMatrix{T}(init::ArrayInitializer,modes::Vararg{ModeRange,2}) where {T}
	arr = Matrix{T}(init,map(length,modes))
	SHMatrix(arr,modes)
end
SHMatrix(init::ArrayInitializer,modes) = SHMatrix{ComplexF64}(init,modes)

# Add methods to Base functions

@inline Base.parent(s::SHArray) = s.parent
Base.similar(arr::T) where {T<:SHArray} = T(similar(parent(arr)),modes(arr),shdims(arr))
Base.dataids(A::SHArray) = Base.dataids(parent(A)) # needed for fast broadcasting
function Broadcast.broadcast_unalias(dest::SHArray, src::SHArray)
	parent(dest) === parent(src) ? src : Broadcast.unalias(dest, src)
end


# Accessor methods
@inline modes(s::SHArray) = s.modes
@inline shdims(s::SHArray) = s.shdims
@inline shmodes(b::SHArrayOnlyFirstAxis) = first(modes(b))

firstshmode(t::Tuple{AbstractUnitRange,Vararg{<:Any}}) = firstshmode(Base.tail(t))
firstshmode(t::Tuple{ModeRange,Vararg{<:Any}}) = first(t)

@inline shmodes(b::SHArrayOneAxis) = firstshmode(modes(b))
@inline shmodes(b::SHArray) = Tuple(modes(b)[i] for i in shdims(b))

@inline Base.size(s::SHArray) = size(parent(s))
@inline Base.size(s::SHArray,d) = size(parent(s),d)
@inline Base.axes(s::SHArray) = axes(parent(s))
@inline Base.axes(s::SHArray,d) = axes(parent(s),d)

# Indexing 

Base.IndexStyle(::Type{SA}) where {SA<:SHArray} = IndexStyle(parenttype(SA))
parenttype(::Type{<:SHArray{<:Any,<:Any,AA}}) where {AA} = AA
parenttype(A::SHArray) = parenttype(typeof(A))

const ModeRangeIndexType = Union{Tuple{Integer,Integer},ModeRange,
							Tuple{AbstractUnitRange{<:Integer},AbstractUnitRange{<:Integer}}}

@propagate_inbounds function Base.to_indices(s::SHArray, inds::Tuple)
	Base.to_indices(s, modes(s), inds)
end
@propagate_inbounds function Base.to_indices(s::SHArray, inds::Tuple{ModeRangeIndexType})
	Base.to_indices(s, modes(s), inds)
end
function Base.to_indices(s::SHArray, inds::Tuple{Any})
	Base.to_indices(s, (eachindex(IndexLinear(),s),), inds)
end
Base.to_indices(s::SHArray, inds::Tuple{Vararg{Union{Integer, CartesianIndex}}}) = 
	Base.to_indices(s, axes(s), inds)

Base.to_indices(s::SHArray, ::Tuple{}) = ()

function Base.uncolon(inds::Tuple{ModeRange,Vararg{Any}}, I::Tuple{Colon, Vararg{Any}})
	Base.Slice(Base.OneTo(length(first(inds))))
end

@propagate_inbounds function Base.to_indices(s::SHArray, m::Tuple{ModeRange,Vararg{Any}},
	inds::Tuple{ModeRangeIndexType,Vararg{Any}})

	(modeindex(first(m),first(inds)), to_indices(s, tail(m), tail(inds))...)
end

# throw an informative error if the axis is not indexed by a ModeRange
Base.to_indices(s::SHArray, m::Tuple{AxisType,Vararg{Any}},
	inds::Tuple{ModeRangeIndexType,Vararg{Any,N} where N}) = throw(NotAnSHAxisError())

# getindex
@propagate_inbounds function Base.getindex(s::SHArray, I...)
	parent(s)[to_indices(s, I)...]
end

# Linear indexing with one integer
@propagate_inbounds Base.getindex(s::SHArray, ind::Int) = parent(s)[ind]

# setindex
@propagate_inbounds function Base.setindex!(s::SHArray, val, I...)
	parent(s)[to_indices(s, I)...] = val
	s
end

# Linear indexing with one integer
@propagate_inbounds function Base.setindex!(s::SHArray, val, ind::Int)
	parent(s)[ind] = val
	s
end

# Broadcasting
Base.BroadcastStyle(::Type{<:SHArray}) = Broadcast.ArrayStyle{SHArray}()

@inline function assert_leading_modes_compatible(a::Tuple,b::Tuple)
	assert_modes_same(a[1],b[1])
	assert_leading_modes_compatible(Base.tail(a),Base.tail(b))
end
@inline assert_leading_modes_compatible(a::Tuple{},b::Tuple) = nothing
@inline assert_leading_modes_compatible(a::Tuple,b::Tuple{}) = nothing
@inline assert_leading_modes_compatible(a::Tuple{},b::Tuple{}) = nothing

function modes_shdims_larger(A::SHArray,B::SHArray)
	modes_A = modes(A)
    modes_out,shdims_out = modes_A,shdims(A)

    (A === B) && return modes_out,shdims_out

    modes_B = modes(B)

    assert_leading_modes_compatible(modes_A,modes_B)

    if ndims(A) < ndims(B)
		modes_out,shdims_out = modes_B,shdims(B)
	end

	return modes_out,shdims_out
end

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SHArray}}, 
	::Type{ElType}) where ElType

    A = find_sharray(bc)
    modes_out,shdims_out = modes(A),shdims(A)

    # Check to make sure that all the SHArrays that are
    # being broadcasted over have compatible axes
    for B in Broadcast.flatten(bc).args
    	if B isa SHArray
    		(A === B) && continue
    		modes_out,shdims_out = modes_shdims_larger(A,B)
    	end
    end
    SHArray(similar(Array{ElType}, axes(bc)), modes_out, shdims_out)
end

find_sharray(bc::Base.Broadcast.Broadcasted) = find_sharray(bc.args)
find_sharray(args::Tuple) = find_sharray(find_sharray(args[1]), Base.tail(args))
find_sharray(x) = x
find_sharray(a::SHArray, rest) = a
find_sharray(::Any, rest) = find_sharray(rest)

# Extend methods from SphericalHarmonicModes
modeindex(arr::SHArrayOneAxis,l,m) = modeindex(shmodes(arr),l,m)
modeindex(arr::SHArrayOneAxis,mode::Tuple) = modeindex(shmodes(arr),mode)
modeindex(::SHArrayOneAxis,::Colon,::Colon) = Colon()
modeindex(::SHArrayOneAxis,::Tuple{Colon,Colon}) = Colon()

l_range(arr::SHArrayOneAxis) = l_range(shmodes(arr))
l_range(arr::SHArrayOneAxis,m::Integer) = l_range(shmodes(arr),m)
m_range(arr::SHArrayOneAxis) = m_range(shmodes(arr))
m_range(arr::SHArrayOneAxis,l::Integer) = m_range(shmodes(arr),l)

l₁_range(arr::SHArrayOneAxis) = l₁_range(shmodes(arr))
l₂_range(arr::SHArrayOneAxis) = l₂_range(shmodes(arr))
l₂_range(arr::SHArrayOneAxis,l₁::Integer) = l₂_range(shmodes(arr),l₁)

end # module
