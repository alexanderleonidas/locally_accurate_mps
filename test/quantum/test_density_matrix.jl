using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using .AKLT
using .DensityMatrix
using .MatrixProductOperators

@testset "Density Matrix Tests" begin
    N = 4
    sites = siteinds("S=1", N)
    gs = aklt_ground_state(sites) # Generate the AKLT ground state MPS
    rs = random_mps(sites, 2) # Generate a random MPS with bond dimension 2
    start_site = 2
    stop_site = 3

    # full_rho_mpo = outer(gs', gs)
    rho_mpo = reduced_density_matrix(gs, start_site, stop_site)
    rand_rho_mpo = reduced_density_matrix(rs, start_site, stop_site)

    @testset "Hermiticity" begin
        # 1. Test Hermiticity (ρ = ρ†)
        rho_dag = dag(swapprime(rho_mpo, 0, 1))
        @test norm(rho_mpo - rho_dag) < 1e-10
        rand_dag = dag(swapprime(rand_rho_mpo, 0, 1))
        @test norm(rand_rho_mpo - rand_dag) < 1e-10
    end

    @testset "Trace" begin
        # 2. Test Trace (Tr(ρ) = 1)
        @test isapprox(mpo_trace(rho_mpo), 1.0; atol=1e-10)
        @test isapprox(mpo_trace(rand_rho_mpo), 1.0; atol=1e-10)
    end
end

@testset "Product-state site ordering" begin
    N = 6
    d = 3
    sites = siteinds("Qudit", N; dim=d)
    basis_states = ["0", "1", "2", "0", "1", "2"]
    ψ = MPS(sites, basis_states)

    # ITensor's combined index is column-major: the first requested site
    # is the fastest-varying coordinate.
    linear_index(states) = 1 + sum(states[i] * d^(i - 1) for i in eachindex(states))
    assert_product_rdm(sites_ordered) = begin
        ρ = get_rdm(ψ, sites_ordered)
        state_tuple = [parse(Int, basis_states[s]) for s in sites_ordered]
        expected = linear_index(state_tuple)

        @test norm(ρ - Diagonal(diag(ρ))) < 1e-12
        @test count(x -> !isapprox(x, 0.0; atol=1e-12), ρ) == 1
        @test isapprox(ρ[expected, expected], 1.0; atol=1e-12)
        @test isapprox(tr(ρ), 1.0; atol=1e-12)
    end

    @testset "Wrap-around block preserves logical order" begin
        wrap_sites = wraparound_sites(5, 2, N)
        @test wrap_sites == [5, 6, 1, 2]
        assert_product_rdm(wrap_sites)
    end

    @testset "Interior hole preserves site order" begin
        assert_product_rdm([1, 2, 4, 5])
    end
end

@testset "AKLT dense reference for wrap-around RDM" begin
    N = 8
    d = 3
    kept_sites = wraparound_sites(7, 2, N)

    # Build the open-boundary AKLT state directly from its bond-dimension-2
    # matrices. This reference does not use the MPS tensors or any MPO code.
    A = Array{Float64}(undef, d, 2, 2)
    A[1, :, :] = [0.0 -1.0; 0.0 0.0]
    A[2, :, :] = [1 / sqrt(2) 0.0; 0.0 -1 / sqrt(2)]
    A[3, :, :] = [0.0 0.0; 1.0 0.0]
    left_boundary = [0.0 -1.0; 1 / sqrt(2) 0.0; 0.0 0.0]
    right_boundary = [-1.0 0.0; 0.0 -1 / sqrt(2); 0.0 0.0]

    ψ_dense = zeros(Float64, ntuple(_ -> d, N))
    for state in CartesianIndices(ψ_dense)
        amplitude = reshape(left_boundary[state[1], :], 1, 2)
        for site in 2:(N - 1)
            amplitude = amplitude * A[state[site], :, :]
        end
        ψ_dense[state] = (amplitude * reshape(right_boundary[state[N], :], 2, 1))[1]
    end
    ψ_dense ./= sqrt(sum(abs2, ψ_dense))

    function dense_rdm(statevector, kept)
        traced = setdiff(1:ndims(statevector), kept)
        result = zeros(Float64, d^length(kept), d^length(kept))
        linear_index(values) = 1 + sum((values[i] - 1) * d^(i - 1) for i in eachindex(values))

        for row_values in Iterators.product(ntuple(_ -> 1:d, length(kept))...)
            row_full = zeros(Int, N)
            row_full[kept] = collect(row_values)
            row_index = linear_index(row_values)

            for col_values in Iterators.product(ntuple(_ -> 1:d, length(kept))...)
                col_full = zeros(Int, N)
                col_full[kept] = collect(col_values)
                col_index = linear_index(col_values)

                total = 0.0
                for traced_values in Iterators.product(ntuple(_ -> 1:d, length(traced))...)
                    row_full[traced] = collect(traced_values)
                    col_full[traced] = collect(traced_values)
                    total += statevector[CartesianIndex(Tuple(row_full))] *
                             statevector[CartesianIndex(Tuple(col_full))]
                end
                result[row_index, col_index] = total
            end
        end
        return result
    end

    sites = siteinds("S=1", N)
    ψ = aklt_ground_state(sites)
    expected = dense_rdm(ψ_dense, kept_sites)
    actual = get_rdm(ψ, kept_sites)

    @test kept_sites == [7, 8, 1, 2]
    @test isapprox(actual, expected; atol=1e-12, rtol=1e-12)

    kept_sites = [2, 3, 6, 7]   # contiguous span 2..7, holes at 4,5 — no wraparound
    # reuse ψ_dense and dense_rdm exactly as built above
    expected = dense_rdm(ψ_dense, kept_sites)
    actual = get_rdm(ψ, kept_sites)
    println("rank of expected: ", rank(expected))
    @test isapprox(actual, expected; atol=1e-12, rtol=1e-12)

    test_rdm = get_rdm(ψ, collect(1:4))
    println("test rank: ", rank(test_rdm))

end
