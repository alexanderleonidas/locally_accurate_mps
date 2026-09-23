module XXZ
using ITensors
using ITensorMPS

export xxz_hamiltonian_mpo, xxz_ground_state

"""
    xxz_hamiltonian_mpo(sites; Δ=1.0, J=1.0) -> MPO
"""
function xxz_hamiltonian_mpo(sites::Vector{<:Index}; Δ::Float64=1.0, J::Float64=1.0)::MPO
    N = length(sites)
    os = OpSum()
    for i in 1:(N - 1)
        os += J * 0.5, "S+", i, "S-", i + 1
        os += J * 0.5, "S-", i, "S+", i + 1
        os += J * Δ,   "Sz", i, "Sz", i + 1
    end
    return MPO(os, sites)
end

"""
    xxz_ground_state(sites; Δ=1.0, J=1.0, nsweeps=10, maxdim=100) -> (Float64, MPS)
"""
function xxz_ground_state(sites::Vector{<:Index}; Δ::Float64=1.0, J::Float64=1.0, nsweeps::Int=10, maxdim::Int=100)
    H = xxz_hamiltonian_mpo(sites; Δ=Δ, J=J)
    psi0 = random_mps(sites; linkdims=10)

    cutoff = [1e-11]
    maxdims = min.(maxdim, [10, 20, 50, maxdim])

    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdims, cutoff=cutoff, outputlevel=0)
    return energy, psi
end

end
