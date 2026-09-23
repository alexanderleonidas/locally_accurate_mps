module AKLT

using ITensors
using ITensorMPS

export aklt_ground_state, aklt_hamiltonian_mpo, aklt_hamiltonian_mpo2

"""
    aklt_ground_state(sites) -> MPS

Construct the exact ground state of the AKLT Hamiltonian for N spin-1 sites as an ITensor MPS.
"""
function aklt_ground_state(sites::Vector{<:Index})::MPS
    N = length(sites)
    mps = MPS(sites)

    if N == 1
        t = ITensor(sites[1])
        t[sites[1] => 2] = 1.0
        mps[1] = t
        return mps
    end

    links = [Index(2, "Link,l=$i") for i in 1:N-1]
    norm_val = 1.0 / √2

    for i in 1:N
        s = sites[i]

        if i == 1
            t = ITensor(s, links[1])
            # v_L^\dagger * A^m  (Row 1 of A^m)
            # m = +1 (s=1): A^+1[1,2] = -√2
            t[s => 1, links[1] => 2] = -√2 * norm_val
            # m = 0  (s=2): A^0[1,1]  = 1
            t[s => 2, links[1] => 1] = 1.0 * norm_val
            mps[i] = t

        elseif i == N
            t = ITensor(s, links[N-1])
            # A^m * v_R  (Column 2 of A^m)
            # m = +1 (s=1): A^+1[1,2] = -√2
            t[s => 1, links[N-1] => 1] = -√2 * norm_val
            # m = 0  (s=2): A^0[2,2]  = -1
            t[s => 2, links[N-1] => 2] = -1.0 * norm_val
            mps[i] = t

        else
            t = ITensor(links[i-1], s, links[i])
            # Bulk tensor A^m_{\alpha, \beta}
            # m = +1 (s=1)
            t[links[i-1] => 1, s => 1, links[i] => 2] = -√2 * norm_val
            # m = 0  (s=2)
            t[links[i-1] => 1, s => 2, links[i] => 1] = 1.0 * norm_val
            t[links[i-1] => 2, s => 2, links[i] => 2] = -1.0 * norm_val
            # m = -1 (s=3)
            t[links[i-1] => 2, s => 3, links[i] => 1] = √2 * norm_val
            mps[i] = t
        end
    end

    normalize!(mps)
    return mps
end

"""
    aklt_hamiltonian_mpo(sites) -> MPO

Construct the AKLT Hamiltonian as an MPO using ITensor's OpSum.
H = Σ_i [ 1/6 (S_i · S_{i+1})^2 + 1/2 S_i · S_{i+1} + 1/3 ]
"""
function aklt_hamiltonian_mpo(sites::Vector{<:Index})::MPO
    N = length(sites)
    os = OpSum()

    for i in 1:(N - 1)
        # Linear term: S_i · S_{i+1} = 1/2 (S+_i S-_{i+1} + S-_i S+_{i+1}) + Sz_i Sz_{i+1}
        os += 0.5, "Sz", i, "Sz", i+1
        os += 0.25, "S+", i, "S-", i+1
        os += 0.25, "S-", i, "S+", i+1

        # Quadratic term: 1/6 * (S_i . S_{i+1})^2
        # (Sz Sz)^2
        os += 1/6, "Sz", i, "Sz", i, "Sz", i+1, "Sz", i+1

        # (Sz S+ S- Sz terms)
        os += 1/12, "Sz", i, "S+", i, "Sz", i+1, "S-", i+1
        os += 1/12, "Sz", i, "S-", i, "Sz", i+1, "S+", i+1
        os += 1/12, "S+", i, "Sz", i, "S-", i+1, "Sz", i+1
        os += 1/12, "S-", i, "Sz", i, "S+", i+1, "Sz", i+1

        # (S+ S- S+ S- terms)
        os += 1/24, "S+", i, "S+", i, "S-", i+1, "S-", i+1
        os += 1/24, "S-", i, "S-", i, "S+", i+1, "S+", i+1
        os += 1/24, "S+", i, "S-", i, "S-", i+1, "S+", i+1
        os += 1/24, "S-", i, "S+", i, "S+", i+1, "S-", i+1

        # Constant term
        os += 1/3, "Id", i
    end

    return MPO(os, sites)
end

"""
    aklt_hamiltonian_mpo2(sites) -> MPO

Construct the AKLT Hamiltonian as an MPO using ITensor's OpSum.
H = Σ_i [ 1/3(S_i · S_{i+1})^2 + S_i · S_{i+1} ]
"""
function aklt_hamiltonian_mpo2(sites::Vector{<:Index})::MPO
    N = length(sites)
    os = OpSum()

    for i in 1:(N - 1)
        # Linear term: S_i · S_{i+1} = 1/2 (S+_i S-_{i+1} + S-_i S+_{i+1}) + Sz_i Sz_{i+1}
        os += 1.0, "Sz", i, "Sz", i+1
        os += 0.5, "S+", i, "S-", i+1
        os += 0.5, "S-", i, "S+", i+1

        # Quadratic term: 1/3 * (S_i . S_{i+1})^2
        # (Sz Sz)^2
        os += 1/3, "Sz", i, "Sz", i, "Sz", i+1, "Sz", i+1

        # (Sz S+ S- Sz terms)
        os += 1/6, "Sz", i, "S+", i, "Sz", i+1, "S-", i+1
        os += 1/6, "Sz", i, "S-", i, "Sz", i+1, "S+", i+1
        os += 1/6, "S+", i, "Sz", i, "S-", i+1, "Sz", i+1
        os += 1/6, "S-", i, "Sz", i, "S+", i+1, "Sz", i+1

        # (S+ S- S+ S- terms)
        os += 1/12, "S+", i, "S+", i, "S-", i+1, "S-", i+1
        os += 1/12, "S-", i, "S-", i, "S+", i+1, "S+", i+1
        os += 1/12, "S+", i, "S-", i, "S-", i+1, "S+", i+1
        os += 1/12, "S-", i, "S+", i, "S+", i+1, "S-", i+1
    end

    return MPO(os, sites)
end

end # module AKLT
