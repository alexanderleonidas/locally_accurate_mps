using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using .TFIM

@testset "TFIM Model Tests" begin
    N = 4
    J = 1.0
    h = 0.5
    sites = siteinds("SpinHalf", N)
    # Construct Hamiltonian MPO
    H = tfim_hamiltonian(sites; J=J, h=h)
    H_dense = contract(H)

    @testset "Ground State Energy Expectation Test" begin
        # Exact Diagonalisation via dense Matrix
        comb_row = combiner(sites...)
        comb_col = combiner(prime.(sites)...)
        H_mat = Matrix(H_dense * comb_row * comb_col, combinedind(comb_row), combinedind(comb_col))

        energies_ed = eigen(Hermitian(H_mat)).values
        E0_ed = energies_ed[1]

        # DMRG Ground State search to verify MPO
        psi0 = random_mps(sites; linkdims=2)
        sweeps = Sweeps(5)
        maxdim!(sweeps, 10, 20, 100)
        cutoff!(sweeps, 1e-10)

        energy_dmrg, psi_dmrg = dmrg(H, psi0, sweeps; outputlevel=0)

        # Assertions
        @test isapprox(energy_dmrg, E0_ed, atol=1e-8)
    end

    @testset "Hamiltonian Properties" begin
        # Test Hermiticity of the MPO
        H_dag = swapprime(dag(H_dense), 1 => 2)
        @test isapprox(H_dense, H_dag, atol=1e-12)
    end
end
