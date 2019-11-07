using SphericalHarmonicArrays,OffsetArrays
import SphericalHarmonicArrays: SizeMismatchArrayModeError, 
SizeMismatchError, MismatchedDimsError, UnexpectedAxisTypeError,
NotAnSHAxisError

using Test

@testset "Constructors" begin
	mode_st = st(0:1,0:0)
	mode_ts = ts(0:1,0:0)
	mode_s′s = s′s(0:2,2,0:2)
	@testset "0dim" begin
	    @test SHArray(zeros()) == zeros()
	end
	@testset "normal" begin
	    arr = zeros(ComplexF64,length(mode_st))
	    @test SHArray(arr) == arr
	end
	@testset "one SH axis" begin
		arr = zeros(ComplexF64,length(mode_st))
		@testset "SHArray" begin
		    @test SHArray(mode_st) == arr
		    @test SHArray((mode_st,)) == arr
		    @test SHArray(mode_st,(1,)) == arr
		    @test SHArray((mode_st,),(1,)) == arr
		    @test SHArray(arr,mode_st) == arr
		    @test SHArray(arr,(mode_st,)) == arr
		    @test SHArray(arr,mode_st,(1,)) == arr
		    @test SHArray(arr,(mode_st,),(1,)) == arr
		end
		@testset "SHVector" begin
		    @test SHVector(mode_st) == arr
		    @test SHVector{ComplexF64}(mode_st) == arr
		    @test SHVector{Float64}(mode_st) == real(arr)
		    @test SHVector(arr,mode_st) == arr
		end
		@testset "BipolarVSH" begin
			arr = zeros(ComplexF64,length(mode_st),-1:1,-1:1)
		   	@test BipolarVSH(mode_st) == arr
		   	@test BipolarVSH(mode_st,-1:1) == arr
		   	@test BipolarVSH(mode_st,-1:1,-1:1) == arr
		    @test BipolarVSH{ComplexF64}(mode_st) == arr
		    @test BipolarVSH{Float64}(mode_st) == real(arr)
		    @test BipolarVSH(arr,mode_st) == arr
		    @test BipolarVSH(arr,(mode_st,1:3,1:3)) == arr

		   	arr = zeros(ComplexF64,length(mode_st),0:0,-1:1)
		   	@test BipolarVSH(mode_st,0:0) == arr
		   	@test BipolarVSH(mode_st,0:0,-1:1) == arr

		   	arr = zeros(ComplexF64,length(mode_st),0:0,0:0)
		   	@test BipolarVSH(mode_st,0:0,0:0) == arr

		end

	    @testset "Errors" begin
	    	@test_throws UnexpectedAxisTypeError SHArray(mode_st,(2,))
	    end
	end
    @testset "2D mixed axes" begin
    	@testset "first SH" begin
		    arr = zeros(ComplexF64,length(mode_st),2)
		    ax = (mode_st,1:2)
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
			arr = zeros(ComplexF64,2,length(mode_st),)
			ax = (1:2,mode_st)
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
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,(1:2,st(s_range(mode_st))),(2,))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,(1:2,st(s_range(mode_st))))
		    	@test_throws SizeMismatchError SHArray(arr,(1:3,st(s_range(mode_st))),(2,))
		    	@test_throws SizeMismatchError SHArray(arr,(1:3,st(s_range(mode_st))))
		    end
		end
		@testset "both SH" begin
			arr = zeros(ComplexF64,length(mode_st),length(mode_st))
			ax = (mode_st,mode_st)
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
		    	ax = (st(s_range(mode_st)),st(s_range(mode_st)))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,ax,(1,2))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,ax)
		    end
		end
		@testset "SHMatrix" begin
			function testSHMatrix(mode1,mode2)
				arr = zeros(ComplexF64,length(mode1),length(mode2))
				@test SHMatrix(arr,mode1,mode2) isa SHMatrix
			    @test SHMatrix(mode1,mode2) isa SHMatrix
			    @test SHMatrix((mode1,mode2)) isa SHMatrix
			    @test SHMatrix(arr,mode1,mode2) == arr
			    @test SHMatrix(arr,(mode1,mode2)) == arr
			    @test SHMatrix(mode1,mode2) == arr
			    @test SHMatrix((mode1,mode2)) == arr
			    @test SHMatrix{Float64}(mode1,mode2) == real(arr)
			    @test SHMatrix{Float64}((mode1,mode2)) == real(arr)
			    @test SHdims(SHMatrix(arr,mode1,mode2)) == (1,2)
			    @test SHdims(SHMatrix(arr,(mode1,mode2))) == (1,2)
			    @test SHdims(SHMatrix(mode1,mode2)) == (1,2)
			    @test SHdims(SHMatrix((mode1,mode2))) == (1,2)
			    @test modes(SHMatrix(arr,mode1,mode2)) == (mode1,mode2)
			    @test modes(SHMatrix(arr,(mode1,mode2))) == (mode1,mode2)
			    @test modes(SHMatrix(mode1,mode2)) == (mode1,mode2)
			    @test modes(SHMatrix((mode1,mode2))) == (mode1,mode2)	
			end
			@testset "st st" begin
			    testSHMatrix(mode_st,mode_st)
			end
			@testset "st ts" begin
			    testSHMatrix(mode_st,mode_ts)
			end
			@testset "ts st" begin
			    testSHMatrix(mode_ts,mode_st)
			end
			@testset "ts ts" begin
			    testSHMatrix(mode_ts,mode_ts)
			end
		    @testset "st s′s" begin
			    testSHMatrix(mode_st,mode_s′s)
			end
			@testset "ts s′s" begin
			    testSHMatrix(mode_ts,mode_s′s)
			end
		    @testset "s′s st" begin
			    testSHMatrix(mode_s′s,mode_st)
			end
			@testset "s′s ts" begin
			    testSHMatrix(mode_s′s,mode_ts)
			end
			@testset "s′s s′s" begin
			    testSHMatrix(mode_s′s,mode_s′s)
			end
		end
	end
end

@testset "Indexing" begin
	mode_st = st(0:1,0:0)
	@testset "1D SH" begin
	    v = SHVector(mode_st)
	    @testset "getindex" begin
	    	@testset "linear" begin
		    	for i in eachindex(v)
			    	@test v[i] == 0
			    end
	    	end
	    	@testset "mode_st" begin
			    for m in mode_st
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
	    	@testset "mode_st" begin
			    for (i,m) in enumerate(mode_st)
			    	v[m] = i
			    	@test v[m] == i
			    end
	    	end
	    end
	end
	@testset "1D normal" begin
	    v = SHArray(zeros(length(mode_st)))
	    @testset "getindex" begin
	    	@testset "linear" begin
		    	for i in eachindex(v)
			    	@test v[i] == 0
			    end
			end
			@testset "mode_st" begin
			    for m in mode_st
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
	    	@testset "mode_st" begin
			    for (i,m) in enumerate(mode_st)
			    	@test_throws NotAnSHAxisError v[m] = i
			    end
	    	end
	    end
	end
	@testset "2D mixed axes" begin
		@testset "first SH" begin
		    v = SHArray((mode_st,1:2))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(v)
				    	@test v[i] == 0
				    end
		    	end
		    	@testset "mode_st" begin
				    for ind2 in axes(v,2), m in mode_st
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
		    	@testset "mode_st" begin
				    for ind2 in axes(v,2),(i,m) in enumerate(mode_st)
				    	v[m,ind2] = i
				    	@test v[m,ind2] == i
				    end
		    	end
		    end
		end
		@testset "second SH" begin
		    v = SHArray((1:2,mode_st))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(v)
				    	@test v[i] == 0
				    end
		    	end
		    	@testset "mode_st" begin
				    for (i,m) in enumerate(mode_st),ind1 in axes(v,1)
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
		    	@testset "mode_st" begin
				    for (i,m) in enumerate(mode_st),ind1 in axes(v,1)
				    	v[ind1,m] = i
				    	@test v[ind1,m] == i
				    end
		    	end
		    end
		end
		@testset "both SH" begin
		    v = SHArray((mode_st,mode_st))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(v)
				    	@test v[i] == 0
				    end
		    	end
		    	@testset "mode_st" begin
				    for m2 in mode_st,m1 in mode_st
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
		    	@testset "mode_st" begin
				    for (i,m2) in enumerate(mode_st),m1 in mode_st
				    	v[m1,m2] = i
				    	@test v[m1,m2] == i
				    end
		    	end
		    end
		end
	end
end

@testset "similar" begin
    mode_st = st(0:1,0:0)
	mode_ts = ts(0:1,0:0)
	mode_s′s = s′s(0:2,2,0:2)

	function testsimilar(arr)
		@test similar(arr) isa typeof(arr)
		@test size(parent(similar(arr))) == size(parent(arr))
		@test modes(similar(arr)) == modes(arr)
		@test SHdims(similar(arr)) == SHdims(arr)
	end
	
	SA = SHArray((mode_st,1:2,mode_ts))
	testsimilar(SA)
	B = BipolarVSH(mode_st,0:0,0:0)
	testsimilar(B)
	v = SHVector(mode_st)
	testsimilar(v)
	M = SHMatrix(mode_st,mode_st)
	testsimilar(M)
end