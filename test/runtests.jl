using SphericalHarmonicArrays,OffsetArrays
import SphericalHarmonicArrays: SizeMismatchArrayModeError, 
SizeMismatchError, MismatchedDimsError, UnexpectedAxisTypeError,
NotAnSHAxisError

using Test

@testset "Constructors" begin
	mode = st(0:1,0:0)
	@testset "0dim" begin
	    @test SHArray(zeros()) == zeros()
	end
	@testset "normal" begin
	    arr = zeros(ComplexF64,length(mode))
	    @test SHArray(arr) == arr
	end
	@testset "one SH axis" begin
		arr = zeros(ComplexF64,length(mode))
		@testset "SHArray" begin
		    @test SHArray(mode) == arr
		    @test SHArray((mode,)) == arr
		    @test SHArray(mode,(1,)) == arr
		    @test SHArray((mode,),(1,)) == arr
		    @test SHArray(arr,mode) == arr
		    @test SHArray(arr,(mode,)) == arr
		    @test SHArray(arr,mode,(1,)) == arr
		    @test SHArray(arr,(mode,),(1,)) == arr
		end
		@testset "SHVector" begin
		    @test SHVector(mode) == arr
		    @test SHVector{ComplexF64}(mode) == arr
		    @test SHVector{Float64}(mode) == real(arr)
		    @test SHVector(arr,mode) == arr
		end
		@testset "BipolarVSH" begin
			arr = zeros(ComplexF64,length(mode),-1:1,-1:1)
		   	@test BipolarVSH(mode) == arr
		   	@test BipolarVSH(mode,-1:1) == arr
		   	@test BipolarVSH(mode,-1:1,-1:1) == arr
		    @test BipolarVSH{ComplexF64}(mode) == arr
		    @test BipolarVSH{Float64}(mode) == real(arr)
		    @test BipolarVSH(arr,mode) == arr
		    @test BipolarVSH(arr,(mode,1:3,1:3)) == arr

		   	arr = zeros(ComplexF64,length(mode),0:0,-1:1)
		   	@test BipolarVSH(mode,0:0) == arr
		   	@test BipolarVSH(mode,0:0,-1:1) == arr

		   	arr = zeros(ComplexF64,length(mode),0:0,0:0)
		   	@test BipolarVSH(mode,0:0,0:0) == arr

		end

	    @testset "Errors" begin
	    	@test_throws UnexpectedAxisTypeError SHArray(mode,(2,))
	    end
	end
    @testset "2D mixed axes" begin
    	@testset "first SH" begin
		    arr = zeros(ComplexF64,length(mode),2)
		    ax = (mode,1:2)
		    SHdims = (1,)
		    @test SHArray(ax) == arr
		    @test SHArray{ComplexF64}(ax) == arr
		    @test SHArray{Float64}(ax) == real(arr)
		    @test SHArray(ax,SHdims) == arr
		    @test SHArray{ComplexF64}(ax,SHdims) == arr
		    @test SHArray{Float64}(ax,SHdims) == real(arr)
		    @test SHArray(arr,ax) == arr
		    @test SHArray(arr,ax,SHdims) == arr

		    @testset "Errors" begin
		    	@test_throws MismatchedDimsError SHArray(ax,(1,2,3))
		    	@test_throws UnexpectedAxisTypeError SHArray(ax,(2,))
		    end
		end
		@testset "second SH" begin
			arr = zeros(ComplexF64,2,length(mode),)
			ax = (1:2,mode)
			SHdims = (2,)
		    @test SHArray(ax) == arr
		    @test SHArray{ComplexF64}(ax) == arr
		    @test SHArray{Float64}(ax) == real(arr)
		    @test SHArray(ax,SHdims) == arr
		    @test SHArray{ComplexF64}(ax,SHdims) == arr
		    @test SHArray{Float64}(ax,SHdims) == real(arr)
		    @test SHArray(arr,ax) == arr
		    @test SHArray(arr,ax,SHdims) == arr
		    
		    @testset "Errors" begin
		    	@test_throws MismatchedDimsError SHArray(ax,(1,2,3))
		    	@test_throws UnexpectedAxisTypeError SHArray(ax,(1,))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,(1:2,st(s_range(mode))),(2,))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,(1:2,st(s_range(mode))))
		    	@test_throws SizeMismatchError SHArray(arr,(1:3,st(s_range(mode))),(2,))
		    	@test_throws SizeMismatchError SHArray(arr,(1:3,st(s_range(mode))))
		    end
		end
		@testset "both SH" begin
			arr = zeros(ComplexF64,length(mode),length(mode))
			ax = (mode,mode)
			SHdims = (1,2)
		    @test SHArray(ax) == arr
		    @test SHArray{ComplexF64}(ax) == arr
		    @test SHArray{Float64}(ax) == real(arr)
		    @test SHArray(ax,SHdims) == arr
		    @test SHArray{ComplexF64}(ax,SHdims) == arr
		    @test SHArray{Float64}(ax,SHdims) == real(arr)
		    @test SHArray(arr,ax) == arr
		    @test SHArray(arr,ax,SHdims) == arr

		    @testset "Errors" begin
		    	@test_throws MismatchedDimsError SHArray(ax,(1,2,3))
		    	@test_throws UnexpectedAxisTypeError SHArray{Float64}(ax,(1,))
		    	@test_throws UnexpectedAxisTypeError SHArray{Float64}(ax,(2,))
		    	ax = (st(s_range(mode)),st(s_range(mode)))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,ax,(1,2))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,ax)
		    end
		end
	end
end

@testset "Indexing" begin
	mode = st(0:1,0:0)
	@testset "1D SH" begin
	    v = SHVector(mode)
	    @testset "getindex" begin
	    	@testset "linear" begin
		    	for i in eachindex(v)
			    	@test v[i] == 0
			    end
	    	end
	    	@testset "mode" begin
			    for m in mode
			    	@test v[m] == 0
			    end
	    	end
	    end
	    @testset "setindex" begin
	    	@testset "linear" begin
			    for i in eachindex(v)
			    	v[i] = i
			    	@test v[i] == i
			    end
	    	end
	    	@testset "mode" begin
			    for (i,m) in enumerate(mode)
			    	v[m] = i
			    	@test v[m] == i
			    end
	    	end
	    end
	end
	@testset "1D normal" begin
	    v = SHArray(zeros(length(mode)))
	    @testset "getindex" begin
	    	@testset "linear" begin
		    	for i in eachindex(v)
			    	@test v[i] == 0
			    end
			end
			@testset "mode" begin
			    for m in mode
			    	@test_throws NotAnSHAxisError v[m]
			    end
	    	end
	    end
	    @testset "setindex" begin
	    	@testset "linear" begin
			    for i in eachindex(v)
			    	v[i] = i
			    	@test v[i] == i
			    end
	    	end
	    	@testset "mode" begin
			    for (i,m) in enumerate(mode)
			    	@test_throws NotAnSHAxisError v[m] = i
			    end
	    	end
	    end
	end
	@testset "2D mixed axes" begin
		@testset "first SH" begin
		    v = SHArray((mode,1:2))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(v)
				    	@test v[i] == 0
				    end
		    	end
		    	@testset "mode" begin
				    for ind2 in axes(v,2), m in mode
				    	@test v[m,ind2] == 0
				    end
		    	end
		    end
		    @testset "setindex" begin
		    	@testset "linear" begin
				    for i in eachindex(v)
				    	v[i] = i
				    	@test v[i] == i
				    end
		    	end
		    	@testset "mode" begin
				    for ind2 in axes(v,2),(i,m) in enumerate(mode)
				    	v[m,ind2] = i
				    	@test v[m,ind2] == i
				    end
		    	end
		    end
		end
		@testset "second SH" begin
		    v = SHArray((1:2,mode))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(v)
				    	@test v[i] == 0
				    end
		    	end
		    	@testset "mode" begin
				    for (i,m) in enumerate(mode),ind1 in axes(v,1)
				    	@test v[ind1,m] == 0
				    end
		    	end
		    end
		    @testset "setindex" begin
		    	@testset "linear" begin
				    for i in eachindex(v)
				    	v[i] = i
				    	@test v[i] == i
				    end
		    	end
		    	@testset "mode" begin
				    for (i,m) in enumerate(mode),ind1 in axes(v,1)
				    	v[ind1,m] = i
				    	@test v[ind1,m] == i
				    end
		    	end
		    end
		end
		@testset "both SH" begin
		    v = SHArray((mode,mode))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(v)
				    	@test v[i] == 0
				    end
		    	end
		    	@testset "mode" begin
				    for m2 in mode,m1 in mode
				    	@test v[m1,m2] == 0
				    end
		    	end
		    end
		    @testset "setindex" begin
		    	@testset "linear" begin
				    for i in eachindex(v)
				    	v[i] = i
				    	@test v[i] == i
				    end
		    	end
		    	@testset "mode" begin
				    for (i,m2) in enumerate(mode),m1 in mode
				    	v[m1,m2] = i
				    	@test v[m1,m2] == i
				    end
		    	end
		    end
		end
	end
end