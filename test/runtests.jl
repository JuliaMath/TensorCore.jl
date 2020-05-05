using TensorCore
using LinearAlgebra
using Test

@testset "Ambiguities" begin
    @test isempty(detect_ambiguities(TensorCore, Base, Core, LinearAlgebra))
end

@testset "TensorCore.jl" begin
    for T in (Int, Float32, Float64, BigFloat)
        a = [T[1, 2], T[-3, 7]]
        b = [T[5, 11], T[-13, 17]]
        @test map(⋅, a, b) == map(dot, a, b) == [27, 158]
        @test map(⊙, a, b) == map(hadamard, a, b) == [a[1].*b[1], a[2].*b[2]]
        @test map(⊗, a, b) == map(tensor, a, b) == [a[1]*transpose(b[1]), a[2]*transpose(b[2])]
        @test hadamard!(fill(typemax(Int), 2), T[1, 2], T[-3, 7]) == [-3, 14]
        @test tensor!(fill(typemax(Int), 2, 2), T[1, 2], T[-3, 7]) == [-3 7; -6 14]
    end

    @test_throws DimensionMismatch [1,2] ⊙ [3]
    @test_throws DimensionMismatch hadamard!([0, 0, 0], [1,2], [-3,7])
    @test_throws DimensionMismatch hadamard!([0, 0], [1,2], [-3])
    @test_throws DimensionMismatch hadamard!([0, 0], [1], [-3,7])
    @test_throws DimensionMismatch tensor!(Matrix{Int}(undef, 2, 2), [1], [-3,7])
    @test_throws DimensionMismatch tensor!(Matrix{Int}(undef, 2, 2), [1,2], [-3])

    u, v = [2+2im, 3+5im], [1-3im, 7+3im]
    @test u ⋅ v == conj(u[1])*v[1] + conj(u[2])*v[2]
    @test u ⊙ v == [u[1]*v[1], u[2]*v[2]]
    @test u ⊗ v == [u[1]*v[1] u[1]*v[2]; u[2]*v[1] u[2]*v[2]]
    @test hadamard(u, v) == u ⊙ v
    @test tensor(u, v)   == u ⊗ v
    dest = similar(u)
    @test hadamard!(dest, u, v) == u ⊙ v
    dest = Matrix{Complex{Int}}(undef, 2, 2)
    @test tensor!(dest, u, v) == u ⊗ v

    for (A, B, b) in (([1 2; 3 4], [5 6; 7 8], [5,6]),
                      ([1+0.8im 2+0.7im; 3+0.6im 4+0.5im],
                       [5+0.4im 6+0.3im; 7+0.2im 8+0.1im],
                       [5+0.6im,6+0.3im]))
        @test A ⊗ b == cat(A*b[1], A*b[2]; dims=3)
        @test A ⊗ B == cat(cat(A*B[1,1], A*B[2,1]; dims=3),
                           cat(A*B[1,2], A*B[2,2]; dims=3); dims=4)
    end

    A, B = reshape(1:27, 3, 3, 3), reshape(1:4, 2, 2)
    @test A ⊗ B == [a*b for a in A, b in B]

    # Adjoint/transpose is a dual vector, not an AbstractMatrix
    v = [1,2]
    @test_throws ErrorException v ⊗ v'
    @test_throws ErrorException v ⊗ transpose(v)
    @test_throws ErrorException v' ⊗ v
    @test_throws ErrorException transpose(v) ⊗ v
    @test_throws ErrorException v' ⊗ v'
    @test_throws ErrorException transpose(v) ⊗ transpose(v)
    @test_throws ErrorException v' ⊗ transpose(v)
    @test_throws ErrorException transpose(v) ⊗ v'
    @test_throws ErrorException A ⊗ v'
    @test_throws ErrorException A ⊗ transpose(v)

    # Docs comparison to `kron`
    v, w = [1,2,3], [5,7]
    @test kron(v,w) == vec(w ⊗ v)
    @test w ⊗ v == reshape(kron(v,w), (length(w), length(v)))
end
