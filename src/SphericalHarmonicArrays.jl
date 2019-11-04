module SphericalHarmonicArrays
using Reexport
import Base: tail, @propagate_inbounds

@reexport using SphericalHarmonicModes
import SphericalHarmonicModes: ModeRange

export SHArray,SHVector, modes, SizeMismatchArrayModeError, 
SizeMismatchError, MismatchedDimsError, UnexpectedAxisTypeError,
NotAnSHAxisError

struct SizeMismatchArrayModeError <: Exception
	dim :: Int
	arraylength :: Int
	modelength :: Int
end

struct SizeMismatchError <: Exception
	dim :: Int
	arraylength :: Int
	axislength :: Int
end

struct MismatchedDimsError <: Exception
	M :: Int
	N :: Int
end

struct UnexpectedAxisTypeError <: Exception
	dim :: Int
	texp :: Type
	tgot :: Type
end

struct NotAnSHAxisError <: Exception end

Base.showerror(io::IO, e::SizeMismatchArrayModeError) = print(io,
	"Size of array does not match the number of modes along dimension $(e.dim). "*
	"Size of array is $(e.arraylength) whereas the number of modes is $(e.modelength)")

Base.showerror(io::IO, e::SizeMismatchError) = print(io,
	"Size of array does not match the size of axis specified along dimension $(e.dim). "*
	"Size of array is $(e.arraylength) whereas the size of axis is $(e.axislength)")

Base.showerror(io::IO, e::MismatchedDimsError) = print(io,
	"The array has $(e.N) dimensions but $(e.M) shperical harmonic axes specified")

Base.showerror(io::IO, e::UnexpectedAxisTypeError) = print(io,
	"Expected type $(e.text) along dimension $(e.dim) but got $(e.tgot)")

Base.showerror(io::IO, e::NotAnSHAxisError) = print(io,
	"Attempted to index into a non-SH axis with a mode Tuple")

const AxisType = Union{ModeRange,AbstractUnitRange}

struct SHArray{T,N,AA<:AbstractArray{T,N},TM<:Tuple{Vararg{<:AxisType,N}},NSH} <: AbstractArray{T,N}
	parent :: AA
	modes :: TM
	SHdims :: NTuple{NSH,Int}

	function SHArray{T,N,AA,TM,NSH}(arr,modes,SHdims) where {T,N,AA,TM,NSH}

		NSH <= N || throw(MismatchedDimsError(NSH,N))

		for (dim,mode) in enumerate(modes)
			if dim in SHdims && !isa(mode,ModeRange)
				throw(UnexpectedAxisTypeError(dim,ModeRange,typeof(mode)))
			elseif !(dim in SHdims) && mode isa ModeRange
				throw(UnexpectedAxisTypeError(dim,typeof(axes(arr,dim)),typeof(mode)))
			end 
			if size(arr,dim) != length(mode)
				if dim in SHdims
					throw(SizeMismatchArrayModeError(dim,size(arr,dim),length(mode)))
				else
					throw(SizeMismatchError(dim,size(arr,dim),length(mode)))
				end
			end
		end
		new{T,N,AA,TM,NSH}(arr,modes,SHdims)
	end
end

const SHVector{T,AA,M<:Tuple{ModeRange}} = SHArray{T,1,AA,M,1}

function SHArray(arr::AbstractArray{T,N},modes::Tuple{Vararg{<:AxisType,N}},
	SHdims::NTuple{N,Int}) where {T<:Number,N}
	SHArray{T,N,typeof(arr),typeof(modes),N}(arr,modes,SHdims)
end

function SHArray(arr::AbstractArray{T,N},modes::Tuple{Vararg{<:AxisType,N}},
	SHdims::NTuple{NSH,Int}) where {T<:Number,N,NSH}
	SHArray{T,N,typeof(arr),typeof(modes),NSH}(arr,modes,SHdims)
end

function SHArray(arr::AbstractArray{T,N},SHModes::Tuple{Vararg{<:AxisType,NSH}},
	SHdims::NTuple{NSH,Int}) where {T<:Number,N,NSH}
	allmodes = Vector{AxisType}(undef,N)
	allmodes .= axes(arr)
	for (d,m) in zip(SHdims,SHModes)
		allmodes[d] = m
	end
	modes = Tuple(allmodes)
	SHArray{T,N,typeof(arr),typeof(modes),length(SHdims)}(arr,modes,SHdims)
end

function SHArray(arr::AbstractArray,modes::Tuple)
	SHdims = Tuple(dim for dim in 1:ndims(arr) if modes[dim] isa ModeRange)
	SHArray(arr,modes,SHdims)
end

SHArray(arr::AbstractArray) = SHArray(arr,axes(arr))
SHArray(arr::AbstractVector,mode::ModeRange) = SHArray(arr,(mode,))
SHArray(arr::AbstractVector,mode::ModeRange,SHdims::Tuple{Int}) = SHArray(arr,(mode,),SHdims)

# These constructors allocate an array of an appropriate size
SHArray{T}(mode::ModeRange) where {T<:Number} = SHArray(zeros(T,length(mode)),(mode,))
SHArray(mode::ModeRange) = SHArray{ComplexF64}(mode)
function SHArray{T}(modes::Vararg{<:AxisType,N}) where {T<:Number,N}
	ax = Tuple(length(m) for m in modes)
	SHArray(zeros(T,ax),modes)
end
SHArray{T}(modes::Tuple{Vararg{<:AxisType}}) where {T<:Number} = SHArray{T}(modes...)
SHArray(modes::Vararg{<:AxisType}) = SHArray{ComplexF64}(modes)
SHArray(modes::Tuple{Vararg{<:AxisType}}) = SHArray{ComplexF64}(modes)
function SHArray{T}(modes::Tuple{Vararg{<:AxisType}},SHdims::Tuple{Vararg{Int}}) where {T<:Number}
	ax = Tuple(length(m) for m in modes)
	SHArray(zeros(T,ax),modes,SHdims)
end
SHArray(modes::Tuple{Vararg{<:AxisType}},SHdims::Tuple{Vararg{Int}}) = SHArray{ComplexF64}(modes,SHdims)
SHArray{T}(mode::ModeRange,SHdims::Tuple{Int}) where {T<:Number} = SHArray(zeros(T,length(mode)),(mode,),SHdims)
SHArray(mode::ModeRange,SHdims::Tuple{Int}) = SHArray{ComplexF64}(mode,SHdims)

# Convenience constructors
SHVector(arr::AbstractVector,modes::ModeRange) = SHArray(arr,(modes,),(1,))
SHVector(arr::AbstractVector,modes::Tuple{ModeRange}) = SHArray(arr,modes,(1,))
# Automatically allocate a vector of an appropriate size
SHVector{T}(mode::ModeRange) where {T<:Number} = SHArray(zeros(T,length(mode)),(mode,),(1,))
SHVector(mode::ModeRange) = SHArray{ComplexF64}(mode)

# Add to Base methods

@inline Base.parent(s::SHArray) = s.parent
@inline modes(s::SHArray) = s.modes

Base.size(s::SHArray) = size(parent(s))
Base.axes(s::SHArray) = axes(parent(s))

Base.fill!(s::SHArray,x) = fill!(parent(s),x)

# Indexing 

Base.IndexStyle(::Type{<:SHArray}) = IndexLinear()

# Convert modes to a linear index along each axis
@inline compute_flatinds(modes,inds::Integer) = inds
@inline function compute_flatinds(modes::Tuple{Vararg{Any,1}},inds::Tuple{Integer,Integer})
	modeindex(modes[1],inds)
end
@inline function compute_flatinds(::AbstractUnitRange,::Tuple{Integer,Integer})
	throw(NotAnSHAxisError())
end
@inline function compute_flatinds(modes::ModeRange,inds::Tuple{Integer,Integer})
	modeindex(modes,inds)
end
@inline function compute_flatinds(modes::Tuple,inds::Tuple{Vararg{Union{Integer,Tuple}}})
	(compute_flatinds(modes[1],inds[1]),compute_flatinds(tail(modes),tail(inds))...)
end
@inline compute_flatinds(::Tuple{},::Tuple{}) = ()
@inline compute_flatinds(s::SHArray,inds::Union{Integer,Tuple}) = compute_flatinds(modes(s),inds)


# getindex

@inline @propagate_inbounds function Base.getindex(s::SHArray{<:Number,N},
	inds::Vararg{Union{Integer,Tuple{Integer,Integer}},N}) where {N}

	parentinds = compute_flatinds(s,inds)
	@boundscheck checkbounds(s, parentinds...)
	@inbounds ret = parent(s)[parentinds...]
	ret
end

# These method passes the indices to the parent
@inline @propagate_inbounds function Base.getindex(s::SHArray{<:Number,N},
	inds::Vararg{Integer,N}) where {N}

	@boundscheck checkbounds(s, inds...)
	@inbounds ret = parent(s)[inds...]
	ret
end

@inline @propagate_inbounds function Base.getindex(s::SHArray,ind::Integer)
	@boundscheck checkbounds(s, ind)
	@inbounds ret = parent(s)[ind]
	ret
end

# setindex

@inline @propagate_inbounds function Base.setindex!(s::SHArray{<:Number,N},val,
	inds::Vararg{Union{Integer,Tuple{Integer,Integer}},N}) where {N}

	parentinds = compute_flatinds(s,inds)
	@boundscheck checkbounds(s, parentinds...)
	@inbounds parent(s)[parentinds...] = val
	val
end

# These methods pass the indices to the parent
@inline @propagate_inbounds function Base.setindex!(s::SHArray{<:Number,N},val,
	inds::Vararg{Integer,N}) where {N}

	@boundscheck checkbounds(s, inds...)
	@inbounds parent(s)[inds...] = val
	val
end

@inline @propagate_inbounds function Base.setindex!(s::SHArray,val,ind::Integer)
	@boundscheck checkbounds(s, ind)
	@inbounds parent(s)[ind] = val
	val
end

end # module
