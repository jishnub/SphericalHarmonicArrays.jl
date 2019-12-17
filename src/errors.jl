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

struct ModeMismatchError{A,B} <: Exception
	mode1 :: A
	mode2 :: B
end

function Base.showerror(io::IO, e::ModeMismatchError{A,A}) where {A}
	print(io,"Modes $(e.mode1) and $(e.mode2) differ")
end
function Base.showerror(io::IO, e::ModeMismatchError{A,B}) where {A,B}
	print(io,"Modes are of different types: $A and $B")
end

function assert_modes_same(a,b)
	a == b || throw(ModeMismatchError(a,b))
end