include("src/LocallyAccurateMPS.jl")
using .LocallyAccurateMPS
greet()
main()

using ITensors, ITensorMPS
using .LocallyAccurateMPS.DensityMatrix
using .LocallyAccurateMPS.MatrixProductStates
using .LocallyAccurateMPS.AKLT
using .LocallyAccurateMPS.Measures

# let
#   N = 6
#   sites = siteinds("S=1",N)

#   H = aklt_hamiltonian_mpo(sites)
#   ground_state = aklt_ground_state(sites)

#   psi0 = random_mps(sites;linkdims=10)

#   nsweeps = 5
#   maxdim = [10,20,100,100,200]
#   cutoff = [1E-10]

#   energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

#   println("Trace Distance ", trace_distance(psi,ground_state))
#   println("Max 2-local Trace Distance ", k_local_trace_distance(psi,ground_state,2))
#   println("Frobenius Norm ", frobenius_norm_mps(psi,ground_state))
#   @show analyze_bond_dimension_spectrum(psi)
#   @show analyze_bond_dimension_spectrum(ground_state)

#   return
# end

# let
#     N = 8
#     sites = siteinds("S=1/2", N)
#
#     # 1. Global pure state trace distance
#     psi1 = random_mps(sites; linkdims=4)
#     psi2 = random_mps(sites; linkdims=4)
#     println("Global Purified Distance D(|ψ1⟩, |ψ2⟩) = ", round(purified_distance(psi1, psi2), digits=6))
#     # println("Global Trace Distance D(|ψ1⟩, |ψ2⟩) = ", round(trace_distance(psi1, psi2), digits=6))
#
#     # 2. Purity check
#     println("Pure MPO Purity   = ", round(purity(psi1), digits=6))
#
#     rho_mixed = MPO(sites, "Id")
#     rho_mixed /= prod(dim.(sites))  # Normalize to unit trace
#     # for i in 1:N; rho_mixed[i] .*= 0.5; end
#     println("Mixed MPO Purity  = ", round(purity(rho_mixed), digits=6))
#
#     # 3. Construct RDM as MPO
#     sub = 2:4
#     rho_A_mpo = reduced_density_matrix(psi1, sub)
#     println("Constructed RDM MPO for subregion $sub:")
#     println("  - Number of MPO sites: ", length(rho_A_mpo))
#     println("  - RDM MPO Purity     : ", round(purity(rho_A_mpo), digits=6))
#
#     # 4. RDM trace distance using MPOs
#     # println("k-local Trace Distance for subregion $sub = ", round(k_local_trace_distance(psi1, psi2, 3), digits=6))
#     println("k-local Purified Distance for pure MPO subregion $sub = ", round(k_local_purified_distance(psi1, psi2, 3), digits=6))
#     println("k-local Purified Distance for ⟨ψ1|ψ1⟩ subregion $sub = ", round(k_local_purified_distance(psi1, psi1, 3), digits=6))
#
#     println("="^70)
# end
