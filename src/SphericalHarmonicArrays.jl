module SphericalHarmonicArrays
using Reexport, OffsetArrays
import Base: tail, @propagate_inbounds

@reexport using SphericalHarmonicModes
import SphericalHarmonicModes: ModeRange, modeindex, 
l_range, m_range,l₁_range,l₂_range

export SHArray, SHVector, SHMatrix
export modes,shmodes, shdims

struct SizeMismatchArrayModeError <: Exception
	dim :: Int
	arraylength :: Int
	modelength :: Int
end

Base.showerror(io::IO, e::SizeMismatchArrayModeError) = print(io,
	"Size of array does not match the number of modes along dimension $(e.dim). "*
	"Size of array is $(e.arraylength) whereas the number of modes is $(e.modelength)")

struct SizeMismatchError <: Exception
	dim :: Int
	arraylength :: Int
	axislength :: Int
end

Base.showerror(io::IO, e::SizeMismatchError) = print(io,
	"Size of array does not match the size of axis specified along dimension $(e.dim). "*
	"Size of array is $(e.arraylength) whereas the size of axis is $(e.axislength)")

struct MismatchedDimsError <: Exception
	M :: Int
	N :: Int
end

Base.showerror(io::IO, e::MismatchedDimsError) = print(io,
	"The array has $(e.N) dimensions but $(e.M) shperical harmonic axes specified")

struct UnexpectedAxisTypeError <: Exception
	dim :: Int
	texp :: Type
	tgot :: Type
end

Base.showerror(io::IO, e::UnexpectedAxisTypeError) = print(io,
	"Expected type $(e.texp) along dimension $(e.dim) but got $(e.tgot)")

struct NotAnSHAxisError <: Exception end

Base.showerror(io::IO, e::NotAnSHAxisError) = print(io,
	"Attempted to index into a non-SH axis with a mode Tuple")

const AxisType = Union{ModeRange,AbstractUnitRange}
const ArrayInitializer = Union{UndefInitializer, Missing, Nothing}

struct SHArray{T,N,AA<:AbstractArray{T,N},
	TM<:Tuple{Vararg{AxisType,N}},NSH} <: AbstractArray{T,N}

	parent :: AA
	modes :: TM
	shdims :: NTuple{NSH,Int}

	function SHArray{T,N,AA,TM,NSH}(arr,modes,shdims) where {T,N,AA,TM,NSH}

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

function SHArray(arr::AbstractArray{T,N},modes::Tuple{Vararg{AxisType,N}},
	shdims::NTuple{N,Int}) where {T,N}
	SHArray{T,N,typeof(arr),typeof(modes),N}(arr,modes,shdims)
end

function SHArray(arr::AbstractArray{T,N},modes::Tuple{Vararg{AxisType,N}},
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
	SHArray{T,N,typeof(arr),typeof(modes),length(shdims)}(arr,modes,shdims)
end

function SHArray(arr::AbstractArray{<:Any,N},modes::Tuple{Vararg{AxisType,N}}) where {N}
	shdims = Tuple(dim for dim in 1:N if modes[dim] isa ModeRange)
	SHArray(arr,modes,shdims)
end

SHArray(arr::AbstractArray) = SHArray(arr,axes(arr))
SHArray(arr::AbstractVector,mode::ModeRange) = SHArray(arr,(mode,))
SHArray(arr::AbstractVector,mode::ModeRange,shdims::Tuple{Int}) = SHArray(arr,(mode,),shdims)

# These constructors allocate an array of an appropriate size
@inline moderangeaxes(m::ModeRange) = axes(m,1)
@inline moderangeaxes(m::AxisType) = m

function SHArray{T}(modes::Tuple{Vararg{AxisType}},shdims::Tuple{Vararg{Int}}) where {T}
	ax = Tuple(moderangeaxes(m) for m in modes)
	SHArray(zeros(T,ax),modes,shdims)
end
function SHArray(modes::Tuple{Vararg{AxisType}},shdims::Tuple{Vararg{Int}})
	SHArray{ComplexF64}(modes,shdims)
end

function SHArray{T}(mode::ModeRange,shdims::Tuple{Int}) where {T}
	SHArray(zeros(T,moderangeaxes(mode)),(mode,),shdims)
end
SHArray(mode::ModeRange,shdims::Tuple{Int}) = SHArray{ComplexF64}(mode,shdims)

function SHArray{T}(modes::Tuple{Vararg{AxisType,N}}) where {T,N}
	ax = Tuple(moderangeaxes(m) for m in modes)
	SHArray(zeros(T,ax),modes)
end
SHArray(modes::Tuple{Vararg{AxisType}}) = SHArray{ComplexF64}(modes)

SHArray{T}(modes::Vararg{AxisType}) where {T} = SHArray{T}(modes)
SHArray(modes::Vararg{AxisType}) = SHArray{ComplexF64}(modes)

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

# Accessor methods
@inline shmodes(b::SHArrayOneAxis) = modes(b)[1]

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

SHMatrix(arr::AbstractMatrix,modes::Tuple{ModeRange,ModeRange}) = 
	SHArray(arr,modes,(1,2))
SHMatrix(arr::AbstractMatrix,modes::Vararg{ModeRange,2}) = 
	SHArray(arr,modes,(1,2))
SHMatrix{T}(modes::Tuple{ModeRange,ModeRange}) where {T} = 
	SHArray(zeros(T,map(moderangeaxes,modes)),modes,(1,2))
SHMatrix{T}(modes::Vararg{ModeRange,2}) where {T} = SHMatrix{T}(modes)
SHMatrix(modes::Tuple{ModeRange,ModeRange}) = SHMatrix{ComplexF64}(modes)
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
@inline modes(s::SHArray) = s.modes
@inline shdims(s::SHArray) = s.shdims

Base.size(s::SHArray) = size(parent(s))
Base.size(s::SHArray,d) = size(parent(s),d)
Base.axes(s::SHArray) = axes(parent(s))
Base.axes(s::SHArray,d) = axes(parent(s),d)

# Indexing 

Base.IndexStyle(::Type{SA}) where {SA<:SHArray} = IndexStyle(parenttype(SA))
parenttype(::Type{<:SHArray{<:Any,<:Any,AA}}) where {AA} = AA
parenttype(A::SHArray) = parenttype(typeof(A))

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

@inline @propagate_inbounds function Base.getindex(s::SHArray{<:Any,N},
	inds::Vararg{Union{Integer,Tuple{Integer,Integer}},N}) where {N}

	parentinds = compute_flatinds(s,inds)
	@boundscheck checkbounds(s, parentinds...)
	@inbounds ret = parent(s)[parentinds...]
	ret
end

# These method pass the indices to the parent
@inline @propagate_inbounds function Base.getindex(s::SHArray{<:Any,N},
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

@inline @propagate_inbounds function Base.setindex!(s::SHArray{<:Any,N},val,
	inds::Vararg{Union{Integer,Tuple{Integer,Integer}},N}) where {N}

	parentinds = compute_flatinds(s,inds)
	@boundscheck checkbounds(s, parentinds...)
	@inbounds parent(s)[parentinds...] = val
	val
end

# These methods pass the indices to the parent
@inline @propagate_inbounds function Base.setindex!(s::SHArray{<:Any,N},val,
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

## Other Base functions

Base.similar(arr::T) where {T<:SHArray} = T(similar(parent(arr)),modes(arr),shdims(arr))

# Extend methods from SphericalHarmonicModes
modeindex(arr::SHArrayOnlyFirstAxis,l,m) = modeindex(shmodes(arr),l,m)
modeindex(arr::SHArrayOnlyFirstAxis,mode::Tuple) = modeindex(shmodes(arr),mode)
modeindex(::SHArrayOnlyFirstAxis,::Colon,::Colon) = Colon()
modeindex(::SHArrayOnlyFirstAxis,::Tuple{Colon,Colon}) = Colon()

l_range(arr::SHArrayOnlyFirstAxis) = l_range(shmodes(arr))
l_range(arr::SHArrayOnlyFirstAxis,m::Integer) = l_range(shmodes(arr),m)
m_range(arr::SHArrayOnlyFirstAxis) = m_range(shmodes(arr))
m_range(arr::SHArrayOnlyFirstAxis,l::Integer) = m_range(shmodes(arr),l)

l₁_range(arr::SHArrayOnlyFirstAxis) = l₁_range(shmodes(arr))
l₂_range(arr::SHArrayOnlyFirstAxis) = l₂_range(shmodes(arr))
l₂_range(arr::SHArrayOnlyFirstAxis,l₁::Integer) = l₂_range(shmodes(arr),l₁)

end # module
