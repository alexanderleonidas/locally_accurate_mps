using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using .AKLT

@testset "AKLT Model Tests" begin
    N = 4
    sites = siteinds("S=1", N)
    H = aklt_hamiltonian_mpo(sites)

    @testset "Ground State Energy Expectation Test" begin
        gs = aklt_ground_state(sites)

        # Compute ground state energy expectation value <ψ|H|ψ>
        E_val = inner(gs', H, gs) # <ψ|ψ> = 1 since the MPS is normalized

        # Compute ground state energy by diagonalizing the Hamiltonian MPO
        H_dense = contract(H)
        comb_row = combiner(sites...)
        comb_col = combiner(prime.(sites)...)
        H_mat_tensor = H_dense * comb_row * comb_col
        H_mat = Matrix(H_mat_tensor, combinedind(comb_row), combinedind(comb_col))
        # Compute eigenvalues (Eigvals are real since H is Hermitian)
        energies = eigen(Hermitian(H_mat)).values

        # Verify ground state energy per bond is consistent
        @test E_val ≈ energies[1] atol=1e-6
    end

    @testset "Hamiltonian Properties" begin
        H_dense = contract(H)
        comb_row = combiner(sites...)
        comb_col = combiner(prime.(sites)...)
        H_mat = Matrix(H_dense * comb_row * comb_col, combinedind(comb_row), combinedind(comb_col))
        @test ishermitian(H_mat)
    end
end
