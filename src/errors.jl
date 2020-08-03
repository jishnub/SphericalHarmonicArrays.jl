function checkaxes(ax_exp, ax_recv)
	ax_exp == ax_recv || throw_axismismatcherror(ax_exp, ax_recv)
end
function throw_axismismatcherror(ax_exp, ax_recv)
	throw(ArgumentError("expected an axis $ax_exp, received $ax_recv"))
end

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