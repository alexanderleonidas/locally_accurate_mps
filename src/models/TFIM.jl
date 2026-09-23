module TFIM

using ITensors
using ITensorMPS

export tfim_ground_state, tfim_hamiltonian

"""
    tfim_hamiltonian_mpo(sites; J=1.0, h=1.0) -> MPO

Construct the Transverse-Field Ising Model Hamiltonian as an MPO.
H = -4J ∑_i S^z_i S^z_{i+1} - 2h ∑_i S^x_i
"""
function tfim_hamiltonian(sites::Vector{<:Index}; J::Float64=1.0, h::Float64=1.0)::MPO
    N = length(sites)
    os = OpSum()

    # Nearest-neighbor ZZ interaction (-4J Sz_i Sz_{i+1})
    for i in 1:(N - 1)
        os += -4.0 * J, "Sz", i, "Sz", i + 1
    end

    # On-site transverse magnetic field (-2h Sx_i)
    for i in 1:N
        os += -2.0 * h, "Sx", i
    end

    return MPO(os, sites)
end

"""
    tfim_ground_state(sites; J=1.0, h=1.0, nsweeps=10, maxdim=100) -> (Float64, MPS)

Computes the ground state of the TFIM model using DMRG.
"""
function tfim_ground_state(sites::Vector{<:Index}; J::Float64=1.0, h::Float64=1.0, nsweeps::Int=10, maxdim::Int=100)
    H = tfim_hamiltonian(sites; J=J, h=h)
    psi0 = random_mps(sites; linkdims=10)

    # DMRG convergence parameters
    cutoff = [1e-11]
    maxdims = min.(maxdim, [10, 20, 50, maxdim])

    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdims, cutoff=cutoff, outputlevel=0)
    return energy, psi
end

end # end module TFIM
