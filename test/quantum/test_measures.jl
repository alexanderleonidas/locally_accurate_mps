using Test
using ITensors
using ITensorMPS
using .AKLT
using .MatrixProductStates
using .DensityMatrix
using .Measures

@testset "Measures Tests" begin
    N = 5
    sites = siteinds("S=1", N)
    gs = aklt_ground_state(sites)

    @testset "Purified Distance & Local Subregion Verification" begin
        ρ_mps = reduced_density_matrix(gs, 1, 3)
        # Purified distance to itself must be zero
        d_self = purified_distance(ρ_mps, ρ_mps)
        @test isapprox(d_self, 0.0; atol=1e-12)

    end

    @testset "Trace Distance & Local Subregion Verification" begin
        ρ_mps = reduced_density_matrix(gs, 2, 4)
        # Trace distance to itself must be zero
        d_self = trace_distance(ρ_mps, ρ_mps)
        @test isapprox(d_self, 0.0; atol=1e-12)
    end

    @testset "k-Local Trace Distance & Truncation Error Verification" begin
        k = 2
        # Truncate MPS bond dimension
        truncated_mps = truncate_bond_dimension(gs, 1)
        # Measure truncation error
        err = frobenius_norm(gs, truncated_mps)
        @test err >= 0.0
        @test isapprox(k_local_trace_distance(gs, gs, k), 0.0; atol=1e-12)
    end
end
