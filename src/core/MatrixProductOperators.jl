module MatrixProductOperators

using ITensors
using ITensorMPS

export identity_mpo, single_site_mpo, build_nn_hamiltonian_mpo, compress_mpo!, apply_mpo_to_mps, mpo_trace

"""
    identity_mpo(sites) -> MPO

Construct an identity MPO using the given ITensor site indices.
"""
function identity_mpo(sites::Vector{<:Index})::MPO
    N = length(sites)
    mpo = MPO(sites)
    for i in 1:N
        s = sites[i]
        # Create identity tensor acting on site s and its prime copy s'
        op_tensor = op("Id", s)
        mpo[i] = op_tensor
    end
    return mpo
end

"""
    single_site_mpo(op_matrix::AbstractMatrix, site::Int, sites) -> MPO

Build an MPO that applies a single-site matrix operator on `sites[site]` and identity elsewhere.
"""
function single_site_mpo(op_matrix::AbstractMatrix, site::Int, sites::Vector{<:Index})::MPO
    N = length(sites)
    mpo = MPO(sites)
    for i in 1:N
        s = sites[i]
        if i == site
            mpo[i] = ITensor(op_matrix, s', dag(s))
        else
            mpo[i] = op("Id", s)
        end
    end
    return mpo
end

"""
    build_nn_hamiltonian_mpo(h_local::AbstractMatrix, sites) -> MPO

Construct a nearest-neighbor Hamiltonian MPO H = Σ_{i=1}^{N-1} h_local using ITensor's AutoMPO/OpSum.
`h_local` must be a d^2 × d^2 matrix acting on two adjacent sites.
"""
function build_nn_hamiltonian_mpo(h_local::AbstractMatrix, sites::Vector{<:Index})::MPO
    N = length(sites)
    d = dim(sites[1])

    # Extract local operator basis matrices using Pauli / Gell-Mann basis expansion
    os = OpSum()

    # Build standard MPO via OpSum
    for i in 1:N-1
        # Construct two-site ITensor operator
        s1 = sites[i]
        s2 = sites[i+1]
        h_tensor = ITensor(h_local, s1', s2', dag(s1), dag(s2))

        # Add to operator sum
        os += h_tensor, i, i+1
    end

    return MPO(os, sites)
end

"""
    compress_mpo!(mpo::MPO; maxdim::Int=typemax(Int), cutoff::Float64=0.0)

In-place compression of an MPO via SVD sweep using ITensor.
"""
function compress_mpo!(mpo::MPO; maxdim::Int=typemax(Int), cutoff::Float64=0.0)
    truncate!(mpo; maxdim=maxdim, cutoff=cutoff)
    return mpo
end

"""
    apply_mpo_to_mps(mpo::MPO, mps::MPS) -> MPS

Apply an MPO operator to an MPS state and return the resulting MPS. This
is implemented by contracting site-wise and reconstructing an MPS from the
resulting site tensors. Works for MPO and MPS defined on the same site set.
"""
function apply_mpo_to_mps(mpo::MPO, mps::MPS)::MPS
    # Prefer any optimized apply method provided by ITensorMPS
    try
        return apply(mpo, mps)
    catch
        # Fallback: perform local contractions and attempt to rebuild an MPS
        N = length(mpo)
        @assert N == length(mps) "MPO and MPS must have same length"
        sites = [siteind(mps, i) for i in 1:N]
        tensors = Vector{ITensor}(undef, N)
        for i in 1:N
            tensors[i] = mpo[i] * mps[i]
        end
        # Attempt to construct MPS from site tensors; if not supported, raise
        return MPS(tensors, sites)
    end
end


"""
    mpo_trace(mpo::MPO) -> Number

Computes Tr(MPO) by tracing out every physical site index pair (plev 0 and plev 1)
on each MPO tensor, handling non-contiguous blocks cleanly.
"""
function mpo_trace(mpo::MPO)
    acc = ITensor(1.0)
    for s in eachindex(1:length(mpo))
        t_site = mpo[s]

        # Extract physical site indices (tagged "Site", plev 0)
        site_inds_0 = filter(i -> hastags(i, "Site") && plev(i) == 0, inds(t_site))

        tr_t = t_site
        for s0 in site_inds_0
            s1 = prime(s0)
            if hasind(tr_t, s1)
                tr_t = tr_t * delta(s0, s1)
            end
        end

        acc = acc * tr_t
    end

    return scalar(acc)
end

end # module MPOs
