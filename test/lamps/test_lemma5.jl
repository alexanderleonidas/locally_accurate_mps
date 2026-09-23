using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using .StateConstruction
using .DenseStateConstruction

@testset "Lemma 5 Verification on Dense States" begin
    # Setup test spaces
    dA, dB = 3, 3
    siteA = Index(dA, "Site,A")
    siteB = Index(dB, "Site,B")
    siteA_p = prime(siteA)
    siteB_p = prime(siteB)

    # Define test ranks
    rank_A, rank_B = 2, 3

    # Construct random density matrix tau_A
    rand_A = random_itensor(siteA, Index(rank_A, "Link"))
    tau_A = rand_A * prime(rand_A, siteA)
    tau_A /= scalar(tau_A * delta(siteA, siteA_p))

    # Construct random density matrix tau_B
    rand_B = random_itensor(siteB, Index(rank_B, "Link"))
    tau_B = rand_B * prime(rand_B, siteB)
    tau_B /= scalar(tau_B * delta(siteB, siteB_p))

    # Run construction
    sigma_AB = combine_dense_states(tau_A, tau_B)

    @testset "Marginal Recovery" begin
        # Partial trace over site B to recover tau_A
        reduced_A = sigma_AB * delta(siteB, siteB_p)
        # Partial trace over site A to recover tau_B
        reduced_B = sigma_AB * delta(siteA, siteA_p)

        @test norm(reduced_A - tau_A) ≈ 0.0 atol=1e-8
        @test norm(reduced_B - tau_B) ≈ 0.0 atol=1e-8
    end

    @testset "Rank Condition" begin
        F_AB = eigen(sigma_AB, (siteA, siteB), (siteA_p, siteB_p); ishermitian=true)
        D_AB = F_AB.D

        tol = 1e-10
        eigvals_AB = [D_AB[i, i] for i in 1:dim(inds(D_AB)[1]) if D_AB[i, i] > tol]

        rank_sigma_AB = length(eigvals_AB)
        max_expected_rank = max(rank_A, rank_B)

        @test rank_sigma_AB <= max_expected_rank
    end
end

@testset "Lemma 5 Verification on MPOs" begin
    sites_A = siteinds("S=1/2", 2)
    sites_B = siteinds("S=1/2", 2)
    psi_A = MPS(sites_A, ["Up", "Up"])
    psi_B = MPS(sites_B, ["Dn", "Dn"])

    tau_A = outer(prime(psi_A, "Site"), psi_A)
    tau_B = outer(prime(psi_B, "Site"), psi_B)
    sigma_AB = combine_states(tau_A, tau_B; tol=1e-8)

    @test sigma_AB isa MPO
    @test length(sigma_AB) == 4
    @test isapprox(mpo_trace(sigma_AB), 1.0; atol=1e-6)

    sigma_dense = contract(sigma_AB)
    sites_AB = vcat(sites_A, sites_B)
    row_combiner = combiner(sites_AB...)
    col_combiner = combiner(prime.(sites_AB)...)
    sigma_matrix = Matrix(
        sigma_dense * row_combiner * col_combiner,
        combinedind(row_combiner),
        combinedind(col_combiner),
    )

    @test isapprox(sigma_matrix, sigma_matrix'; atol=1e-6)
    @test isapprox(sigma_matrix[13, 13], 1.0; atol=1e-5)
    @test norm(sigma_matrix) ≈ 1.0 atol=1e-5
end
