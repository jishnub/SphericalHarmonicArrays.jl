using SphericalHarmonicArrays
import SphericalHarmonicArrays: ModeMismatchError

import SphericalHarmonicArrays: modes, shdims, shmodes

using OffsetArrays
using SphericalHarmonicModes
using Test
using LinearAlgebra

@test isempty(Test.detect_ambiguities(Base, Core, SphericalHarmonicArrays))

@testset "Constructors" begin
	mode_lm = LM(0:1,0:0)
	mode_ml = ML(0:1,0:0)
	mode_l₂l₁ = L2L1Triangle(0:2,2,0:2)
	@testset "0dim" begin
	    @test SHArray(zeros(), ()) == zeros()
	    @test SHArray(zeros()) == zeros()
	end
	function checkarrs(sa, arr::AbstractArray)
		@test sa == arr
	    @test parent(sa) === arr
	    @test SphericalHarmonicArrays.parenttype(sa) == typeof(arr)
	    @test axes(sa) == axes(arr)
	end
	function checkarraxes(sa, ax::Tuple)
		@test axes(sa) == map(SphericalHarmonicArrays.moderangeaxes, ax)
		@test sa.modes == ax
	end
	@testset "non SH" begin
	    arr = zeros(ComplexF64, length(mode_lm))
	    sa = SHArray(arr, axes(arr))
	    checkarrs(sa, arr)
	    
	    sa = SHArray(arr)
	    checkarrs(sa, arr)
	end
	@testset "one SH axis" begin
		arr = zeros(ComplexF64, length(mode_lm))
		@testset "SHArray" begin

		    sa = SHArray{ComplexF64}((mode_lm,))
		    checkarraxes(sa, (mode_lm,))

		    sa = SHArray(arr,(mode_lm,))
		    checkarraxes(sa, (mode_lm,))
		end
		@testset "SHVector" begin
		    @test SHVector{ComplexF64}(mode_lm) == arr
		    @test SHVector{Float64}(mode_lm) == real(arr)
		    @test SHVector(arr, mode_lm) == arr

		    sv = SHVector(arr, mode_lm)
		    svundef = SHVector{eltype(arr)}(undef, mode_lm)
		    @test svundef.modes == sv.modes
		    @test axes(parent(svundef)) == axes(parent(sv))
		end
	end
    @testset "2D mixed axes" begin
    	@testset "first SH" begin
		    arr = zeros(ComplexF64,length(mode_lm),2)
		    ax = (mode_lm,1:2)

		    function testSA(sa, arr, ax)
		    	@test parent(sa) == arr
			    @test sa.modes == ax
		    end

		    sa = SHArray{ComplexF64}(ax)
		    testSA(sa, arr, ax)

		    sa = SHArray(arr, ax)
		    testSA(sa, arr, ax)

		    sa = SHArray(arr, ax)
		    testSA(sa, arr, ax)
		    
		    sa = SHArray(arr, 1=>mode_lm)
		    testSA(sa, arr, ax)

		    @test SHArray{ComplexF64}(ax) == sa
		    @test SHArray{Float64}(ax) == SHArray(real(arr), ax)
		    @test SHArray(arr,ax) == sa

		    saundef = SHArray{ComplexF64}(undef, ax)
		    @test saundef.modes == sa.modes
		    @test axes(parent(saundef)) == axes(parent(sa))

		    sa = SHArray{ComplexF64}((mode_lm,2))
		    @test axes(sa) == (1:length(mode_lm), 1:2)
		end
		@testset "second SH" begin
			arr = zeros(ComplexF64,2,length(mode_lm),)
			ax = (1:2,mode_lm)
		    @test SHArray{ComplexF64}(ax) == arr
		    @test SHArray{Float64}(ax) == real(arr)
		    @test SHArray(arr,ax) == arr
		    
		    @testset "Errors" begin
		    	@test_throws ArgumentError SHArray(arr,(1:2,LM(l_range(mode_lm))))
		    	@test_throws ArgumentError SHArray(arr,(1:3,LM(l_range(mode_lm))))
		    end
		end
		@testset "both SH" begin
			arr = zeros(ComplexF64,length(mode_lm),length(mode_lm))
			ax = (mode_lm,mode_lm)
		    @test SHArray{ComplexF64}(ax) == arr
		    @test SHArray{Float64}(ax) == real(arr)
		    @test SHArray(arr,ax) == arr

		    @testset "Errors" begin
		    	ax = (LM(l_range(mode_lm)),LM(l_range(mode_lm)))
		    	@test_throws ArgumentError SHArray(arr,ax)
		    end
		end
		@testset "SHMatrix" begin
			function testSHMatrix(T, s, arr, mode1, mode2)
				@test s isa SHMatrix{T}
				@test s == arr
				@test parent(s) === arr
				@test shdims(s) == (1,2)
				@test modes(s) == (mode1, mode2)
			end
			function testSHMatrix(T, s, mode1, mode2)
				@test s isa SHMatrix{T}
				@test shdims(s) == (1,2)
				@test modes(s) == (mode1, mode2)
			end
			function testSHMatrix(mode1,mode2)
				arr = zeros(ComplexF64,length(mode1),length(mode2))
				
				sm1 = SHMatrix(arr, mode1, mode2)
			    testSHMatrix(ComplexF64, sm1, arr, mode1, mode2)
				
				sm2 = SHMatrix(arr, (mode1,mode2))
			    testSHMatrix(ComplexF64, sm2, arr, mode1, mode2)

			    @test sm1 == sm2

			    sm3 = SHMatrix{ComplexF64}(mode1, mode2)
			    testSHMatrix(ComplexF64, sm3, mode1, mode2)
			    @test sm3 == arr
			    @test all(iszero, sm3)

			    sm4 = SHMatrix{ComplexF64}((mode1,mode2))
			    testSHMatrix(ComplexF64, sm4, mode1, mode2)
			    @test sm4 == sm3
			    @test all(iszero, sm4)

			    sm5 = SHMatrix{Float64}(mode1, mode2)
			    testSHMatrix(Float64, sm5, mode1, mode2)
			    @test sm5 == real(arr)
			    @test all(iszero, sm5)

			    sm6 = SHMatrix{Float64}((mode1,mode2))
			    testSHMatrix(Float64, sm6, mode1, mode2)
			    @test sm6 == sm5
			    @test all(iszero, sm6)
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
		    @testset "LM L2L1Triangle" begin
			    testSHMatrix(mode_lm,mode_l₂l₁)
			end
			@testset "ML L2L1Triangle" begin
			    testSHMatrix(mode_ml,mode_l₂l₁)
			end
		    @testset "L2L1Triangle LM" begin
			    testSHMatrix(mode_l₂l₁,mode_lm)
			end
			@testset "L2L1Triangle ML" begin
			    testSHMatrix(mode_l₂l₁,mode_ml)
			end
			@testset "L2L1Triangle L2L1Triangle" begin
			    testSHMatrix(mode_l₂l₁,mode_l₂l₁)
			end

			sm = SHMatrix{ComplexF64}((mode_lm, mode_lm))
		    smundef = SHMatrix{ComplexF64}(undef, (mode_lm, mode_lm))
		    @test smundef.modes == sm.modes
		    @test axes(parent(smundef)) == axes(parent(sm))
		end
		@testset "Pairs" begin
		    mode1 = LM(1:1, 0:1)
		    mode2 = LM(2:2, 0:2)
		    mode3 = LM(2:2)

		    s = SHArray(zeros(map(length, (mode1, mode2))), 1=>mode1, 2=>mode2)
		    @test s.modes == (mode1, mode2)
		    
		    s = SHArray(zeros(map(length, (mode1, mode2))), 2=>mode2, 1=>mode1)
		    @test s.modes == (mode1, mode2)

		    @test_throws ArgumentError SHArray(zeros(map(length, (mode1, mode2))), 2=>mode2, 3=>mode1)

		    # type-unstable version
		    s = SHArray(zeros(map(length, (mode1, mode3))), 1=>mode1, 2=>mode3)
		    @test s.modes == (mode1, mode3)
		end
	end
	@testset "2D none SH" begin
		sa = SHArray{ComplexF64}((2,2))
	    @test axes(sa) == (1:2, 1:2)
	    @test sa.modes == (1:2, 1:2)
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
	       
	       sha = SHArray{Vector}(undef,(1:2,mode_ml))
	       @test !any([isassigned(sha,i) for i in eachindex(sha)])
	       @test size(sha) == (2,length(mode_ml))
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

    @testset "zeros and ones" begin
    	for (DT, f) in ((:zeros, :zero), (:ones, :one))
        	@eval begin
        		mode_lm = LM(0:1,0:0)

	            sa = $DT(mode_lm, mode_lm)
	            @test sa.modes == (mode_lm, mode_lm)
	            @test eltype(sa) == Float64
	            @test all(==($f(Float64)), sa)

	            sa = $DT((mode_lm, mode_lm))
	            @test sa.modes == (mode_lm, mode_lm)
	            @test eltype(sa) == Float64
	            @test all(==($f(Float64)), sa)

	            sa = $DT(mode_lm, 2)
	            @test sa.modes == (mode_lm, 1:2)
	            @test eltype(sa) == Float64
	            @test all(==($f(Float64)), sa)

	            sa = $DT(ComplexF64, mode_lm, mode_lm)
	            @test sa.modes == (mode_lm, mode_lm)
	            @test eltype(sa) == ComplexF64
	            @test all(==($f(ComplexF64)), sa)
	            
	            sa = $DT(ComplexF64, mode_lm, 1:2)
	            @test sa.modes == (mode_lm, 1:2)
	            @test eltype(sa) == ComplexF64
	            @test all(==($f(ComplexF64)), sa)
	        end
	    end
    end
end

@testset "Indexing" begin
	mode_lm = LM(0:1,0:0)
	@testset "0D" begin
	    s = SHArray(zeros())
	    @test s[] == 0
	    s[] = 3
	    @test s[] == 3
	end
	@testset "1D SH" begin
	    v = SHVector{ComplexF64}(mode_lm)
	    @testset "getindex" begin
	    	@testset "linear" begin
		    	for i in eachindex(v)
			    	@test v[i] == 0
			    end
			    for i in eachindex(IndexCartesian(),v)
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
			    for (ind,I) in enumerate(eachindex(IndexCartesian(),v))
			    	v[I] = ind
			    	@test v[I] == ind
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
			    	@test_throws ArgumentError v[m]
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
			    	@test_throws ArgumentError v[m] = i
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
		    a = SHArray{ComplexF64}((mode_lm,1:2))
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
		    a = SHArray{ComplexF64}((1:2,mode_lm))
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
		    a = SHArray{ComplexF64}((mode_lm,mode_lm))
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
	    v = SHVector{ComplexF64}(LM(1:2))
	    @testset "linear" begin
		    for (ind,m) in enumerate(first(shmodes(s)))
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
	@testset "colon and ranges" begin
	    sa = SHArray{ComplexF64}((mode_lm,2))
	    for i in eachindex(sa)
	    	sa[i] = i
	    end
	    @test sa[:,1] == [1,2]
	    @test sa[:,2] == [3,4]
	    @test sa[1,:] == [1,3]
	    @test sa[(0,0),:] == [1,3]
	    @test sa[2,:] == [2,4]
	    @test sa[(1,0),:] == [2,4]
	end
end

@testset "similar" begin
    mode_lm = LM(0:1,0:0)
	mode_ml = ML(0:1,0:0)
	mode_l₂l₁ = L2L1Triangle(0:2,2,0:2)

	function testsimilar(arr)
		@test similar(arr) isa typeof(arr)
		@test size(parent(similar(arr))) == size(parent(arr))
		@test modes(similar(arr)) == modes(arr)
		@test shdims(similar(arr)) == shdims(arr)
	end
	
	SA = SHArray{ComplexF64}((mode_lm,1:2,mode_ml))
	testsimilar(SA)
	v = SHVector{ComplexF64}(mode_lm)
	testsimilar(v)
	M = SHMatrix{ComplexF64}(mode_lm,mode_lm)
	testsimilar(M)
end

@testset "nested" begin
	m = LM(0:2)
	sharr = SHVector{ComplexF64}(m)
	v = [sharr,sharr]
	n = length(v)
    @test typeof(v) == Array{typeof(sharr),1}
    modesv = LM(n:n,1:n)
    shv = SHVector(v, modesv) # get the number of modes to match
    @test typeof(shv) == SHArray{typeof(sharr),1,typeof(v),Tuple{typeof(modesv)}}
    @test_throws ArgumentError SHVector(v,LM(0:0,0:0))
end

@testset "broadcasting" begin
	@testset "unalias" begin
	    s = SHArray(zeros(3,3), (LM(1:1), LM(1:1)))
	    @test Broadcast.broadcast_unalias(s,s) === s
	end
	@testset "same ndims" begin
		@testset "SHArray SHArray" begin
			@testset "first axis" begin
			    s = SHArray{ComplexF64}((LM(0:1),-1:1)); @. s = 2
			    t = similar(s); @. t = 4
			    s2 = SHArray{ComplexF64}((LM(2:2,-2:1),-1:1)) # dimensions and mode types match but different modes
			    s3 = SHArray{ComplexF64}((ML(2:2,-2:1),-1:1)) # dimensions and modes match but different mode types
			    s4 = SHArray{ComplexF64}((LM(0:2),-1:1)) # dimensions don't match
			    s5 = SHArray{ComplexF64}((-1:2,LM(1:1))); # sizes match but axes don't
			    s6 = SHArray{ComplexF64}((LM(0:1),0:2)); # sizes match but axes don't

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
			    s = SHArray{ComplexF64}((-1:1,LM(0:1))); @. s = 2
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
		    s = SHArray{ComplexF64}((-1:1,LM(0:1))); @. s = 2
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
		    s = SHArray{ComplexF64}((LM(0:1),-1:1)); @. s = 2
		    t = SHVector{ComplexF64}(first(modes(s))); @. t = 4

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
		    s = SHArray{ComplexF64}((LM(0:1),-1:1)); @. s = 2
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
	    @testset "SHArrayOneAxis" begin
	    	arr = zeros(2,length(mode_lm),1)
			ax = (axes(arr,1), mode_lm, axes(arr,3))
		    sa = SHArray(arr,ax)
		    @test modes(sa) == ax
		    @test shmodes(sa) == (mode_lm,)
		    @test shdims(sa) == (2,)
		end
		@testset "general" begin
		    arr = zeros(2, length(mode_lm), length(mode_lm))
			ax = (axes(arr,1), mode_lm, mode_lm)
		    sa = SHArray(arr, ax)
		    @test modes(sa) == ax
		    @test shmodes(sa) == (mode_lm, mode_lm)
		    @test shdims(sa) == (2,3) 
		end
	end

	@testset "SHArrayOnlyFirstAxis" begin
		@testset "SHVector" begin
		    sv = SHVector{ComplexF64}(mode_lm)
		    @test modes(sv) == (mode_lm,)
		    @test shmodes(sv) == (mode_lm,)
		    @test shdims(sv) == (1,)
		end
		@testset "SHArray" begin
			arr = zeros(length(mode_lm),2,1)
			ax = (mode_lm, axes(arr)[2:3]...)
		    sa = SHArray(arr,ax)
		    @test modes(sa) == ax
		    @test shmodes(sa) == (mode_lm,)
		    @test shdims(sa) == (1,)
		end
	end
end

@testset "show" begin
    io = IOBuffer()
    showerror(io, SphericalHarmonicArrays.ModeMismatchError(LM(1:2), LM(1:4)))
    showerror(io, SphericalHarmonicArrays.ModeMismatchError(LM(1:2), ML(1:4)))

    sa = SHArray(zeros(2))
    Base.showarg(io, sa, false)

    take!(io)
    d = Diagonal([1,2])
    s = SHArray(d, (1:2, LM(1:1, 0:1)))
    Base.print_matrix(io, d)
    showd = String(take!(io))
    Base.print_matrix(io, s)
    shows = String(take!(io))
    @test shows == showd

    @testset for j in 1:2, i in 1:2
    	srep = Base.replace_in_print_matrix(s, i, j, string(s[i,j]))
    	drep = Base.replace_in_print_matrix(s, i, j, string(d[i,j]))
    	srep == drep
    end
end