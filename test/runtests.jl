using SphericalHarmonicArrays,OffsetArrays
import SphericalHarmonicArrays: SizeMismatchArrayModeError, 
SizeMismatchError, MismatchedDimsError, UnexpectedAxisTypeError,
NotAnSHAxisError, ModeMismatchError

using Test

@testset "Constructors" begin
	mode_lm = LM(0:1,0:0)
	mode_ml = ML(0:1,0:0)
	mode_l₂l₁ = L₂L₁Δ(0:2,2,0:2)
	@testset "0dim" begin
	    @test SHArray(zeros()) == zeros()
	end
	@testset "normal" begin
	    arr = zeros(ComplexF64,length(mode_lm))
	    @test SHArray(arr) == arr
	end
	@testset "one SH axis" begin
		arr = zeros(ComplexF64,length(mode_lm))
		@testset "SHArray" begin
		    @test SHArray(mode_lm) == arr
		    @test SHArray((mode_lm,)) == arr
		    @test SHArray(mode_lm,(1,)) == arr
		    @test SHArray((mode_lm,),(1,)) == arr
		    @test SHArray(arr,mode_lm) == arr
		    @test SHArray(arr,(mode_lm,)) == arr
		    @test SHArray(arr,mode_lm,(1,)) == arr
		    @test SHArray(arr,(mode_lm,),(1,)) == arr
		end
		@testset "SHVector" begin
		    @test SHVector(mode_lm) == arr
		    @test SHVector{ComplexF64}(mode_lm) == arr
		    @test SHVector{Float64}(mode_lm) == real(arr)
		    @test SHVector(arr,mode_lm) == arr
		end

	    @testset "Errors" begin
	    	@test_throws UnexpectedAxisTypeError SHArray(mode_lm,(2,))
	    end
	end
    @testset "2D mixed axes" begin
    	@testset "first SH" begin
		    arr = zeros(ComplexF64,length(mode_lm),2)
		    ax = (mode_lm,1:2)
		    shdims = (1,)
		    @test SHArray(ax) == arr
		    @test SHArray{ComplexF64}(ax) == arr
		    @test SHArray{Float64}(ax) == real(arr)
		    @test SHArray(ax,shdims) == arr
		    @test SHArray{ComplexF64}(ax,shdims) == arr
		    @test SHArray{Float64}(ax,shdims) == real(arr)
		    @test SHArray(arr,ax) == arr
		    @test SHArray(arr,ax,shdims) == arr

		    @testset "Errors" begin
		    	@test_throws MismatchedDimsError SHArray(ax,(1,2,3))
		    	@test_throws UnexpectedAxisTypeError SHArray(ax,(2,))
		    end
		end
		@testset "second SH" begin
			arr = zeros(ComplexF64,2,length(mode_lm),)
			ax = (1:2,mode_lm)
			shdims = (2,)
		    @test SHArray(ax) == arr
		    @test SHArray{ComplexF64}(ax) == arr
		    @test SHArray{Float64}(ax) == real(arr)
		    @test SHArray(ax,shdims) == arr
		    @test SHArray{ComplexF64}(ax,shdims) == arr
		    @test SHArray{Float64}(ax,shdims) == real(arr)
		    @test SHArray(arr,ax) == arr
		    @test SHArray(arr,ax,shdims) == arr
		    
		    @testset "Errors" begin
		    	@test_throws MismatchedDimsError SHArray(ax,(1,2,3))
		    	@test_throws UnexpectedAxisTypeError SHArray(ax,(1,))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,(1:2,LM(l_range(mode_lm))),(2,))
		    	@test_throws SizeMismatchArrayModeError SHArray(arr,(1:2,LM(l_range(mode_lm))))
		    	@test_throws SizeMismatchError SHArray(arr,(1:3,LM(l_range(mode_lm))),(2,))
		    	@test_throws SizeMismatchError SHArray(arr,(1:3,LM(l_range(mode_lm))))
		    end
		end
		@testset "both SH" begin
			arr = zeros(ComplexF64,length(mode_lm),length(mode_lm))
			ax = (mode_lm,mode_lm)
			shdims = (1,2)
		    @test SHArray(ax) == arr
		    @test SHArray{ComplexF64}(ax) == arr
		    @test SHArray{Float64}(ax) == real(arr)
		    @test SHArray(ax,shdims) == arr
		    @test SHArray{ComplexF64}(ax,shdims) == arr
		    @test SHArray{Float64}(ax,shdims) == real(arr)
		    @test SHArray(arr,ax) == arr
		    @test SHArray(arr,ax,shdims) == arr

		    @testset "Errors" begin
		    	@test_throws MismatchedDimsError SHArray(ax,(1,2,3))
		    	@test_throws UnexpectedAxisTypeError SHArray{Float64}(ax,(1,))
		    	@test_throws UnexpectedAxisTypeError SHArray{Float64}(ax,(2,))
		    	ax = (LM(l_range(mode_lm)),LM(l_range(mode_lm)))
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
			    @test shdims(SHMatrix(arr,mode1,mode2)) == (1,2)
			    @test shdims(SHMatrix(arr,(mode1,mode2))) == (1,2)
			    @test shdims(SHMatrix(mode1,mode2)) == (1,2)
			    @test shdims(SHMatrix((mode1,mode2))) == (1,2)
			    @test modes(SHMatrix(arr,mode1,mode2)) == (mode1,mode2)
			    @test modes(SHMatrix(arr,(mode1,mode2))) == (mode1,mode2)
			    @test modes(SHMatrix(mode1,mode2)) == (mode1,mode2)
			    @test modes(SHMatrix((mode1,mode2))) == (mode1,mode2)	
			end
			@testset "LM LM" begin
			    testSHMatrix(mode_lm,mode_lm)
			end
			@testset "LM ML" begin
			    testSHMatrix(mode_lm,mode_ml)
			end
			@testset "ML LM" begin
			    testSHMatrix(mode_ml,mode_lm)
			end
			@testset "ML ML" begin
			    testSHMatrix(mode_ml,mode_ml)
			end
		    @testset "LM L₂L₁Δ" begin
			    testSHMatrix(mode_lm,mode_l₂l₁)
			end
			@testset "ML L₂L₁Δ" begin
			    testSHMatrix(mode_ml,mode_l₂l₁)
			end
		    @testset "L₂L₁Δ LM" begin
			    testSHMatrix(mode_l₂l₁,mode_lm)
			end
			@testset "L₂L₁Δ ML" begin
			    testSHMatrix(mode_l₂l₁,mode_ml)
			end
			@testset "L₂L₁Δ L₂L₁Δ" begin
			    testSHMatrix(mode_l₂l₁,mode_l₂l₁)
			end
		end
	end
	@testset "undef" begin
	    @testset "SHVector" begin

	        sha = SHVector{Vector}(undef,(mode_lm,))
	        @test !any([isassigned(sha,i) for i in eachindex(sha)])
	        @test size(sha) == (length(mode_lm),)
	        sha = SHVector{Vector}(undef,mode_lm)
	        @test !any([isassigned(sha,i) for i in eachindex(sha)])
	    end
	    @testset "SHMatrix" begin

	       sha = SHMatrix{Vector}(undef,(mode_lm,mode_ml))
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	       @test size(sha) == (length(mode_lm),length(mode_ml))
	       sha = SHMatrix{Vector}(undef,mode_lm,mode_ml) 
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	    end
	    @testset "SHArray" begin

	       sha = SHArray{Vector,2}(undef,(1:2,mode_ml))
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	       @test size(sha) == (2,length(mode_ml))
	       sha = SHArray{Vector,2}(undef,1:2,mode_ml)
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	       
	       sha = SHArray{Vector}(undef,(1:2,mode_ml))
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	       sha = SHArray{Vector}(undef,1:2,mode_ml)
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	       @test size(sha) == (2,length(mode_ml))
	       
	       sha = SHArray{Vector,2}(undef,(1:2,mode_ml),(2,))
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	       sha = SHArray{Vector}(undef,(1:2,mode_ml),(2,))
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	    end
	end
end

@testset "Base functions" begin
	mode_lm = LM(0:1,0:0)
	mode_ml = ML(0:1,0:0)
	
	arr = zeros(length(mode_lm),length(mode_ml))
	sm = SHMatrix(arr,(mode_lm,mode_ml))

	@testset "parent" begin
		@test parent(sm) == arr
		@test parent(sm) === arr
	end
    @testset "axes" begin
        @test axes(sm,1) == axes(arr,1)
        @test axes(sm,2) == axes(arr,2)
    end
    @testset "size" begin
        @test size(sm,1) == size(arr,1)
        @test size(sm,2) == size(arr,2)
    end
    @testset "dataids" begin
        @test Base.dataids(sm) == Base.dataids(arr)
    end
end

@testset "Indexing" begin
	mode_lm = LM(0:1,0:0)
	@testset "1D SH" begin
	    v = SHVector(mode_lm)
	    @testset "getindex" begin
	    	@testset "linear" begin
		    	for i in eachindex(v)
			    	@test v[i] == 0
			    end
	    	end
	    	@testset "mode_lm" begin
			    for m in mode_lm
			    	@test v[m] == 0
			    end
	    	end
	    	@testset "colon" begin
	    	    @test all(@. v[:] == 0)
	    	    @test v[:] == zeros(size(v))
	    	end
	    end
	    @testset "setindex" begin
	    	@testset "linear" begin
			    for i in eachindex(v)
			    	v[i] = i
			    	@test v[i] == i
			    end
	    	end
	    	@testset "mode_lm" begin
			    for (i,m) in enumerate(mode_lm)
			    	v[m] = i
			    	@test v[m] == i
			    end
	    	end
	    	@testset "colon" begin
	    	    v[:] .= 4
	    	    @test all(@. v[:] == 4)
	    	end
	    end
	end
	@testset "1D normal" begin
	    v = SHArray(zeros(length(mode_lm)))
	    @testset "getindex" begin
	    	@testset "linear" begin
		    	for i in eachindex(v)
			    	@test v[i] == 0
			    end
			end
			@testset "mode_lm" begin
			    for m in mode_lm
			    	@test_throws NotAnSHAxisError v[m]
			    end
	    	end
	    	@testset "colon" begin
	    	    @test all(@. v[:] == 0)
	    	    @test v[:] == zeros(size(v))
	    	end
	    end
	    @testset "setindex" begin
	    	@testset "linear" begin
			    for i in eachindex(v)
			    	v[i] = i
			    	@test v[i] == i
			    end
	    	end
	    	@testset "mode_lm" begin
			    for (i,m) in enumerate(mode_lm)
			    	@test_throws NotAnSHAxisError v[m] = i
			    end
	    	end
	    	@testset "colon" begin
	    	    v[:] .= 4
	    	    @test all(@. v[:] == 4)
	    	end
	    end
	end
	@testset "2D mixed axes" begin
		@testset "first SH" begin
		    a = SHArray((mode_lm,1:2))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(a)
				    	@test a[i] == 0
				    end
		    	end
		    	@testset "mode_lm" begin
				    for ind2 in axes(a,2), m in mode_lm
				    	@test a[m,ind2] == 0
				    end
		    	end
		    	@testset "colon" begin
		    		for j in axes(a,2)
	    	    		@test a[:,j] == zeros(size(a,1))
	    	    	end
	    	    	for i in axes(a,1)
	    	    		@test a[i,:] == zeros(size(a,2))
	    	    	end
	    	    	@test a[:,:] == zeros(size(a))
	    	    	@test a[:] == zeros(length(a))
	    		end
		    end
		    @testset "setindex" begin
		    	@testset "linear" begin
				    for i in eachindex(a)
				    	a[i] = i
				    	@test a[i] == i
				    end
		    	end
		    	@testset "mode_lm" begin
				    for ind2 in axes(a,2),(i,m) in enumerate(mode_lm)
				    	a[m,ind2] = i
				    	@test a[m,ind2] == i
				    end
		    	end
		    	@testset "colon" begin
		    	    for j in axes(a,2)
		    	    	@. a[:,j] = 1
	    	    		@test a[:,j] == ones(size(a,1))
	    	    		a[:,j] += a[:,j]
	    	    		@test a[:,j] == 2 .* ones(size(a,1))
	    	    	end
	    	    	for i in axes(a,1)
	    	    		@. a[i,:] = 1
	    	    		@test a[i,:] == ones(size(a,2))
	    	    		@. a[i,:] += a[i,:]
	    	    		@test a[i,:] == 2 .* ones(size(a,2))
	    	    	end
	    	    	@. a = 1
	    	    	@test a == ones(size(a))
	    	    	a[:,:] .= 1
	    	    	@test a == ones(size(a))
	    	    	a += a[:,:]
	    	    	@test a == 2 .* ones(size(a))
	    	    	a[:,:] += a[:,:]
	    	    	@test a == 4 .* ones(size(a))
	    	    	a[:] += a[:]
	    	    	@test a == 8 .* ones(size(a))
	    		end
		    end
		end
		@testset "second SH" begin
		    a = SHArray((1:2,mode_lm))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(a)
				    	@test a[i] == 0
				    end
		    	end
		    	@testset "mode_lm" begin
				    for (i,m) in enumerate(mode_lm),ind1 in axes(a,1)
				    	@test a[ind1,m] == 0
				    end
		    	end
		    	@testset "colon" begin
		    		for j in axes(a,2)
	    	    		@test a[:,j] == zeros(size(a,1))
	    	    	end
	    	    	for i in axes(a,1)
	    	    		@test a[i,:] == zeros(size(a,2))
	    	    	end
	    	    	@test a[:,:] == zeros(size(a))
	    	    	@test a[:] == zeros(length(a))
	    		end
		    end
		    @testset "setindex" begin
		    	@testset "linear" begin
				    for i in eachindex(a)
				    	a[i] = i
				    	@test a[i] == i
				    end
		    	end
		    	@testset "mode_lm" begin
				    for (i,m) in enumerate(mode_lm),ind1 in axes(a,1)
				    	a[ind1,m] = i
				    	@test a[ind1,m] == i
				    end
		    	end
		    	@testset "colon" begin
		    	    for j in axes(a,2)
		    	    	@. a[:,j] = 1
	    	    		@test a[:,j] == ones(size(a,1))
	    	    		a[:,j] += a[:,j]
	    	    		@test a[:,j] == 2 .* ones(size(a,1))
	    	    	end
	    	    	for i in axes(a,1)
	    	    		@. a[i,:] = 1
	    	    		@test a[i,:] == ones(size(a,2))
	    	    		@. a[i,:] += a[i,:]
	    	    		@test a[i,:] == 2 .* ones(size(a,2))
	    	    	end
	    	    	@. a = 1
	    	    	@test a == ones(size(a))
	    	    	a[:,:] .= 1
	    	    	@test a == ones(size(a))
	    	    	a += a[:,:]
	    	    	@test a == 2 .* ones(size(a))
	    	    	a[:,:] += a[:,:]
	    	    	@test a == 4 .* ones(size(a))
	    	    	a[:] += a[:]
	    	    	@test a == 8 .* ones(size(a))
	    		end
		    end
		end
		@testset "both SH" begin
		    a = SHArray((mode_lm,mode_lm))
		    @testset "getindex" begin
		    	@testset "linear" begin
			    	for i in eachindex(a)
				    	@test a[i] == 0
				    end
		    	end
		    	@testset "mode_lm" begin
				    for m2 in mode_lm,m1 in mode_lm
				    	@test a[m1,m2] == 0
				    end
		    	end
		    	@testset "colon" begin
		    		for j in axes(a,2)
	    	    		@test a[:,j] == zeros(size(a,1))
	    	    	end
	    	    	for i in axes(a,1)
	    	    		@test a[i,:] == zeros(size(a,2))
	    	    	end
	    	    	@test a[:,:] == zeros(size(a))
	    	    	@test a[:] == zeros(length(a))
	    		end
		    end
		    @testset "setindex" begin
		    	@testset "linear" begin
				    for i in eachindex(a)
				    	a[i] = i
				    	@test a[i] == i
				    end
		    	end
		    	@testset "mode_lm" begin
				    for (i,m2) in enumerate(mode_lm),m1 in mode_lm
				    	a[m1,m2] = i
				    	@test a[m1,m2] == i
				    end
		    	end
		    	@testset "colon" begin
		    	    for j in axes(a,2)
		    	    	@. a[:,j] = 1
	    	    		@test a[:,j] == ones(size(a,1))
	    	    		a[:,j] += a[:,j]
	    	    		@test a[:,j] == 2 .* ones(size(a,1))
	    	    	end
	    	    	for i in axes(a,1)
	    	    		@. a[i,:] = 1
	    	    		@test a[i,:] == ones(size(a,2))
	    	    		@. a[i,:] += a[i,:]
	    	    		@test a[i,:] == 2 .* ones(size(a,2))
	    	    	end
	    	    	@. a = 1
	    	    	@test a == ones(size(a))
	    	    	a[:,:] .= 1
	    	    	@test a == ones(size(a))
	    	    	a += a[:,:]
	    	    	@test a == 2 .* ones(size(a))
	    	    	a[:,:] += a[:,:]
	    	    	@test a == 4 .* ones(size(a))
	    	    	a[:] += a[:]
	    	    	@test a == 8 .* ones(size(a))
	    		end
		    end
		end
	end
	@testset "nested" begin
	    s = SHVector{SHVector}(undef,LM(1:1));
	    v = SHVector(LM(1:2))
	    @testset "linear" begin
		    for (ind,m) in enumerate(shmodes(s))
				s[ind] = v
				@test s[m] == v
				@test s[ind] == v
			end
	    end
	    @testset "colon" begin
	        for ind in eachindex(s)
				s[ind] = v
			end
	        @test s[:] == [v for i in axes(s,1)]
	    end
	end
end

@testset "similar" begin
    mode_lm = LM(0:1,0:0)
	mode_ml = ML(0:1,0:0)
	mode_l₂l₁ = L₂L₁Δ(0:2,2,0:2)

	function testsimilar(arr)
		@test similar(arr) isa typeof(arr)
		@test size(parent(similar(arr))) == size(parent(arr))
		@test modes(similar(arr)) == modes(arr)
		@test shdims(similar(arr)) == shdims(arr)
	end
	
	SA = SHArray((mode_lm,1:2,mode_ml))
	testsimilar(SA)
	v = SHVector(mode_lm)
	testsimilar(v)
	M = SHMatrix(mode_lm,mode_lm)
	testsimilar(M)
end

@testset "nested" begin
	sharr = SHVector(LM(0:2))
	v = [sharr,sharr]
	n = length(v)
    @test typeof(v) == Array{typeof(sharr),1}
    shv = SHVector(v,LM(n:n,1:n)) # get the number of modes to match
    @test typeof(shv) == SHArray{typeof(sharr),1,typeof(v),Tuple{LM},1}
    @test_throws SizeMismatchArrayModeError SHVector(v,LM(0:0,0:0))
end

@testset "broadcasting" begin
	@testset "same ndims" begin
		@testset "SHArray SHArray" begin
			@testset "first axis" begin
			    s = SHArray(LM(0:1),-1:1); @. s = 2
			    t = similar(s); @. t = 4
			    s2 = SHArray(LM(2:2,-2:1),-1:1) # dimensions and mode types match but different modes
			    s3 = SHArray(ML(2:2,-2:1),-1:1) # dimensions and modes match but different mode types
			    s4 = SHArray(LM(0:2),-1:1) # dimensions don't match
			    s5 = SHArray((-1:2,LM(1:1))); # sizes match but axes don't
			    s6 = SHArray(LM(0:1),0:2); # sizes match but axes don't

			    u = s + t
			    @test all(u .== s[1]+t[1])
			    @test u == t + s

			    u = @. s + t
			    @test all(u .== s[1]+t[1])
			    @test u == (@. t + s)

			    u = @. s + 1
			    @test all(u .== s[1]+1)
			    @test u == (@. 1 + s)

			    u = @. 1 + s + t
			    @test all(u .== s[1]+t[1]+1)

			    u = @. s*t
			    @test all(u .== s[1]*t[1])
			    @test u == (@. t*s)

			    u = @. s*t + 1
			    @test all(u .== s[1]*t[1]+1)

			    @test_throws ModeMismatchError s + s2
			    @test_throws ModeMismatchError s + s3
			    @test_throws DimensionMismatch s + s4
			    @test_throws DimensionMismatch s + s5
			    @test_throws DimensionMismatch s + s6
			end
			@testset "second axis" begin
			    s = SHArray((-1:1,LM(0:1))); @. s = 2
			    t = similar(s); @. t = 4

			    u = s + t
			    @test all(u .== s[1]+t[1])
			    @test (t + s) == u
			    
			    u = @. s + t
			    @test all(u .== s[1]+t[1])
			    @test u == (@. t + s)
			    
			    u = @. s + 1
			    @test all(u .== s[1]+1)
			    @test u == (@. 1 + s)
			    
			    u = @. 1 + s + t
			    @test all(u .== s[1]+t[1]+1)
			    
			    u = @. s*t
			    @test all(u .== s[1]*t[1])
			    @test u == (@. t*s)
			    
			    u = @. s*t + 1
			    @test all(u .== s[1]*t[1]+1)
			end
		end
		@testset "SHArray Array" begin
		    s = SHArray((-1:1,LM(0:1))); @. s = 2
		    t = zeros(axes(s)); @. t = 5

		    u = s + t
		    @test all(u .== s[1]+t[1])
		    @test u == (t + s)
		    
		    u = @. s + t
		    @test all(u .== s[1]+t[1])
		    @test u == (@. t + s)
		    
		    u = @. s + 1
		    @test all(u .== s[1]+1)
		    @test u == (@. 1 + s)
		    
		    u = @. 1 + s + t
		    @test all(u .== s[1]+t[1]+1)
		    
		    u = @. s*t
		    @test all(u .== s[1]*t[1])
		    @test u == (@. t*s)
		    
		    u = @. s*t + 1
		    @test all(u .== s[1]*t[1]+1)
		end
	end
	@testset "different ndims" begin
		@testset "SHArray SHVector" begin
		    s = SHArray(LM(0:1),-1:1); @. s = 2
		    t = SHVector(first(modes(s))); @. t = 4

		    u = @. s + t
		    @test all(u .== s[1]+t[1])
		    @test modes(u) == modes(s)
		    @test u == (@. t + s)

		    u = @. s*t
		    @test all(u .== s[1]*t[1])
		    @test modes(u) == modes(s)
		    @test u == (@. t*s)

		    u = @. 1 + s + t
		    @test all(u .== s[1]+t[1]+1)
		    @test modes(u)==modes(s)

		    u = @. s*t + 1
		    @test all(u .== s[1]*t[1]+1)
		    @test modes(u)==modes(s)
		end
		@testset "SHArray Vector" begin
		    s = SHArray(LM(0:1),-1:1); @. s = 2
		    t = zeros(axes(s,1)); @. t = 4

		    u = @. s + t
		    @test all(u .== s[1]+t[1])
		    @test u == (@. t + s)

		    u = @. s*t
		    @test all(u .== s[1]*t[1])
		    @test u == (@. t*s)

		    u = @. 1 + s + t
		    @test all(u .== s[1]+t[1]+1)

		    u = @. s*t + 1
		    @test all(u .== s[1]*t[1]+1)
		end
	end
end

@testset "accessor functions" begin
	mode_lm = LM(0:1,0:0)

	@testset "SHArray" begin
		arr = zeros(2,length(mode_lm),1)
		ax = (axes(arr,1),mode_lm,axes(arr,3))
	    sa = SHArray(arr,ax,(2,))
	    @test modes(sa) == ax
	    @test shmodes(sa) == (mode_lm,)
	    @test shdims(sa) == (2,)
	end

	@testset "SHArrayOnlyFirstAxis" begin
		@testset "SHVector" begin
		    sv = SHVector(mode_lm)
		    @test modes(sv) == (mode_lm,)
		    @test shmodes(sv) == mode_lm
		    @test shdims(sv) == (1,)
		end
		@testset "SHArray" begin
			arr = zeros(length(mode_lm),2,1)
			ax = (mode_lm,axes(arr)[2:3]...)
		    sa = SHArray(arr,ax,(1,))
		    @test modes(sa) == ax
		    @test shmodes(sa) == mode_lm
		    @test shdims(sa) == (1,)
		end
	end
end

@testset "SphericalHarmonicModes" begin
    mode_lm = LM(0:1,0:0)
    mode_ml = ML(0:1,0:0)
    mode_l₂l₁ = L₂L₁Δ(0:2,2,0:2)
    @testset "SHVector" begin
    	@testset "LM" begin
		    sv = SHVector(mode_lm)
		    @test l_range(sv) == l_range(mode_lm)
		    @test l_range(sv,0) == l_range(mode_lm,0)
		    @test modeindex(sv,(1,0)) == modeindex(mode_lm,(1,0))
		    @test modeindex(sv,1,0) == modeindex(mode_lm,1,0)
		    @test modeindex(sv,:,:) == Colon()
		    @test modeindex(sv,(:,:)) == Colon()
    	end
    	@testset "ML" begin
		    sv = SHVector(mode_ml)
		    @test m_range(sv) == m_range(mode_ml)
		    @test m_range(sv,1) == m_range(mode_ml,1)
		    @test modeindex(sv,(1,0)) == modeindex(mode_ml,(1,0))
		    @test modeindex(sv,1,0) == modeindex(mode_ml,1,0)
		    @test modeindex(sv,:,:) == Colon()
		    @test modeindex(sv,(:,:)) == Colon()
    	end
    	@testset "L₂L₁Δ" begin
	    	sv = SHVector(mode_l₂l₁)
    	   	@test l₁_range(sv) == l₁_range(mode_l₂l₁)
		    @test l₂_range(sv) == l₂_range(mode_l₂l₁) 
		    @test l₂_range(sv,1) == l₂_range(mode_l₂l₁,1)
		    @test modeindex(sv,(1,0)) == modeindex(mode_l₂l₁,(1,0))
		    @test modeindex(sv,1,0) == modeindex(mode_l₂l₁,1,0)
		    @test modeindex(sv,:,:) == Colon()
		    @test modeindex(sv,(:,:)) == Colon()
    	end
	end
end
