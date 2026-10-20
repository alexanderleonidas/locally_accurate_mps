module DensityMatrix

using ITensors
using ITensorMPS

export reduced_density_matrix, get_rdm, get_block_rdm, wraparound_sites

# ===========================================================================
# get_block_rdm: the physics engine. Unchanged in its core contraction logic
# (that part was already correct — see the module-level note below on why),
# but with added input validation and a clarified docstring about the ONE
# thing it does NOT promise: leg order for wrap-around blocks.
# ===========================================================================

"""
    get_block_rdm(psi::MPS, sites_1based::Vector{Int}) -> MPO

Computes ρ_X = Tr_{X^c}(|ψ⟩⟨ψ|) as an MPO, for an arbitrary subset of
sites `sites_1based` (1-based). Handles contiguous blocks, blocks with
interior holes (markers), and wrap-around blocks (sites near N together
with sites near 1) all through the same single left-to-right pass:
non-subregion sites are traced by contracting their physical index
directly; subregion sites keep both ket (unprimed) and bra (primed)
physical legs open. Consecutive traced sites are accumulated into a
running "gap" tensor and merged into the next kept site, so entanglement
mediated by a hole (however long) is preserved exactly — this is a
correct partial trace regardless of how `sites_1based` is scattered
around the chain.

⚠ ORDER: the returned MPO's tensor list is always in *increasing chain
index* order (e.g. for sites={5,6,1,2} on N=6, you get [1,2,5,6], not
[5,6,1,2]) — this is a direct consequence of the underlying MPS having
genuine open-boundary structure (site N and site 1 are not physically
bonded), not an oversight. For a wrap-around block, that natural order
is NOT the marker construction's logical order. If you need the result
laid out in your own site order (as you always will for a wrap block),
use `get_rdm` below instead of consuming this MPO directly.
"""
function get_block_rdm(psi::MPS, sites_1based::Vector{Int})::MPO
    N = length(psi)
    isempty(sites_1based) && throw(ArgumentError("sites_1based must be non-empty"))
    for s in sites_1based
        1 <= s <= N || throw(ArgumentError("site $s out of range 1..$N"))
    end
    X = Set(sites_1based)

    mpo_tensors = ITensor[]
    env = ITensor(1.0)

    for k in 1:N
        if !(k in X)
            # Trace out site k: contract physical site index s_k
            traced_k = psi[k] * dag(prime(psi[k], "Link"))
            env = env * traced_k
        else
            # Keep site k: construct local MPO tensor with s_k (plev=0) and s'_k (plev=1)
            t_ket = psi[k]
            t_bra = dag(prime(psi[k])) # Primes both site and link indices
            site_tensor = t_ket * t_bra

            if order(env) > 0
                site_tensor = env * site_tensor
                env = ITensor(1.0)
            end

            push!(mpo_tensors, site_tensor)
        end
    end

    if order(env) > 0 && !isempty(mpo_tensors)
        mpo_tensors[end] = mpo_tensors[end] * env
    end

    return MPO(mpo_tensors)
end

# """
#     reduced_density_matrix(ψ::MPS, sites) -> MPO
#
# Alias for `get_block_rdm`, accepting a Vector, AbstractVector, or UnitRange of 1-based site indices.
# """
# reduced_density_matrix(ψ::MPS, sites::Vector{Int}) = get_block_rdm(ψ, sites)
# reduced_density_matrix(ψ::MPS, sites::AbstractVector{<:Integer}) = get_block_rdm(ψ, collect(Int, sites))
# reduced_density_matrix(ψ::MPS, r::UnitRange{Int}) = get_block_rdm(ψ, collect(r))
# reduced_density_matrix(ψ::MPS, start_idx::Int, end_idx::Int) = get_block_rdm(ψ, collect(start_idx:end_idx))

# ===========================================================================
# get_rdm: the dense, LOGICALLY-ORDERED extraction. This is what fixes the
# wrap-around leg-order problem, and what should be call for anything
# going into a trace-distance comparison against a reference k-site state.
# ===========================================================================

"""
    get_rdm(ψ::MPS, sites_ordered::Vector{Int}) -> Matrix

Dense reduced density matrix on `sites_ordered`, with rows/columns laid
out in EXACTLY the order given — not the chain's natural increasing
order. This is what makes a wrap-around block usable: pass it in its
logical order (e.g. `wraparound_sites(5, 2, 6)` → [5,6,1,2]) and the
returned matrix's site ordering will match that, ready to compare
directly against a "straight" k-site reference computed the same way.

Internally this still calls `get_block_rdm` (correct physics, natural
chain order) and then re-reads out the resulting tensor using each
site's own `siteind`, combined in your requested order — reordering
tensor-product legs is a well-defined relabeling operation and doesn't
require the MPO's internal bond structure to reflect it, which is why
this works even though reordering the MPO's tensor list itself does not
(see `get_block_rdm`'s docstring).

Only practical for small |sites_ordered| — this materializes a
d^n × d^n dense matrix.
"""
function get_rdm(ψ::MPS, sites_ordered::Vector{Int})
    rdm_mpo = get_block_rdm(ψ, sites_ordered)

    T = ITensor(1.0)
    for i in eachindex(1:length(rdm_mpo))
        T *= rdm_mpo[i]
    end

    row_inds = [siteind(ψ, s) for s in sites_ordered]
    col_inds = [i' for i in row_inds]

    c_row = combiner(row_inds...)
    c_col = combiner(col_inds...)
    M = T * c_row * c_col
    return matrix(M, combinedind(c_row), combinedind(c_col))
end
get_rdm(ψ::MPS, r::UnitRange{Int}) = get_rdm(ψ, collect(r))
get_rdm(ψ::MPS, start_idx::Int, end_idx::Int) = get_rdm(ψ, collect(start_idx:end_idx))

"""
    wraparound_sites(c::Int, d::Int, N::Int) -> Vector{Int}

1-based site list for a wrap-around block: sites c, c+1, ..., N, then
1, 2, ..., d — matching MarkerGeometry's M_{j,0} convention. That
module numbers sites 0-based, so add 1 to whatever it gives you before
calling this (or `get_rdm` directly).

Requires 1 ≤ c ≤ N and 0 ≤ d < c. If d ≥ c, this isn't actually a wrap
— you just want `collect(c:d)`.
"""
function wraparound_sites(c::Int, d::Int, N::Int)
    1 <= c <= N || throw(ArgumentError("c=$c out of range 1..$N"))
    0 <= d < c || throw(ArgumentError(
        "d=$d must satisfy 0 ≤ d < c=$c for a genuine wrap " *
        "(otherwise this is just collect(c:d), not a wrap)"))
    return d == 0 ? collect(c:N) : vcat(collect(c:N), collect(1:d))
end

function reduced_density_matrix(target_mps::MPS, block_sites::Vector{Int}; cutoff::Float64=1e-12, maxdim::Int=64)::MPO
    N = length(target_mps)
    sites_set = Set(block_sites)

    # 1. Form full density matrix tensors site-by-site
    T = Vector{ITensor}(undef, N)
    for i in 1:N
        ket = target_mps[i]
        bra = prime(dag(ket)) # Primes both Site and Link indices
        T[i] = ket * bra
    end

    # 2. Fuse internal double-links into standard MPO single links
    for i in 1:(N - 1)
        l_ket = commonind(target_mps[i], target_mps[i+1])
        l_bra = prime(l_ket)

        C = combiner(l_ket, l_bra; tags="Link,l=$i")
        T[i] *= C
        T[i+1] *= dag(C)
    end

    # 3. Trace out non-block sites dynamically
    mpo_tensors = ITensor[]
    E_gap = ITensor(1.0)

    for i in 1:N
        if i in sites_set
            # Absorb any accumulated gap environment into the active block tensor
            if norm(E_gap) != 1.0
                T[i] = E_gap * T[i]
                E_gap = ITensor(1.0) # Reset environment
            end
            push!(mpo_tensors, T[i])
        else
            # Trace out physical site
            s = siteind(target_mps, i)
            sp = prime(s)
            trace_op = delta(s, sp)
            traced_T = T[i] * trace_op

            # Accumulate into gap environment
            E_gap = E_gap * traced_T

            # Dynamic SVD Compression to cap the bond dimension
            inds_E = inds(E_gap)
            if length(inds_E) == 2
                U, S, V = svd(E_gap, inds_E[1]; cutoff=cutoff, maxdim=maxdim)
                E_gap = U * S * V
            end
        end
    end

    # 4. Handle trailing gaps for wrap-around blocks
    if norm(E_gap) != 1.0 && !isempty(mpo_tensors)
        mpo_tensors[end] = mpo_tensors[end] * E_gap
    end

    return MPO(mpo_tensors)
end

# """
#     reduced_density_matrix(ψ::MPS, sites::Vector{Int}) -> MPO
#
# Computes the Reduced Density Matrix (RDM) ρ_S as an MPO for an arbitrary subset of site
# indices `sites`.
#
# Includes a dynamic SVD compression step for interior gaps (e.g., wrap-around blocks)
# to prevent Out-Of-Memory (OOM) tensor blowups across large traced regions.
# """
# function reduced_density_matrix(ψ::MPS, sites::Vector{Int})::MPO
#     N = length(ψ)
#     @assert !isempty(sites) "Subregion sites cannot be empty."
#
#     # 1. Sort and extract unique 1-based site indices
#     s_sorted = sort(unique(sites))
#     @assert 1 <= s_sorted[1] && s_sorted[end] <= N "Site indices out of bounds (1..$N)."
#
#     S = Set(s_sorted)
#     s_min = s_sorted[1]
#     s_max = s_sorted[end]
#
#     # 2. Gauge transform MPS: set orthogonality center at s_min
#     ψ_c = orthogonalize(ψ, s_min)
#
#     mpo_tensors = ITensor[]
#     E = ITensor(1.0)  # Gap environment tensor
#
#     # 3. Iterate through chain from s_min to s_max
#     for k in s_min:s_max
#         if k in S
#             # Site k belongs to the subregion
#             A_ket = ψ_c[k]
#             A_bra = dag(prime(ψ_c[k]))  # Prime both site and link indices
#
#             T = A_ket * A_bra
#
#             # Attach compressed accumulated gap environment
#             if order(E) > 0
#                 T = T * E
#                 E = ITensor(1.0)
#             end
#
#             push!(mpo_tensors, T)
#         else
#             # Site k is a marker hole or wrap-around gap (trace it out)
#             A_ket = ψ_c[k]
#             # Prime ONLY the virtual links so the physical site automatically contracts
#             A_bra = dag(prime(ψ_c[k], "Link"))
#
#             gap_tensor = A_ket * A_bra
#             E = E * gap_tensor
#
#             # --- CRITICAL OOM FIX: Dynamic Gap Compression ---
#             # SVD the gap tensor sequentially to prevent the transfer matrix
#             # from blowing up to an unmanageable χ^4 dense tensor.
#             if !isempty(mpo_tensors)
#                 c_inds = commoninds(mpo_tensors[end], E)
#                 if length(c_inds) > 0
#                     # Factorize E to truncate insignificant long-range correlations
#                     U, D, V = svd(E, c_inds...; cutoff=1e-12, maxdim=256)
#
#                     # Absorb left-side SVD into the previous MPO tensor
#                     mpo_tensors[end] = mpo_tensors[end] * (U * D)
#
#                     # The new E continues forward with a highly truncated rank
#                     E = V
#                 end
#             end
#         end
#     end
#
#     # 4. Apply exterior boundary conditions
#     if s_min > 1
#         l_left = linkind(ψ_c, s_min - 1)
#         if !isnothing(l_left)
#             mpo_tensors[1] = mpo_tensors[1] * delta(l_left, prime(l_left))
#         end
#     end
#
#     if s_max < N
#         l_right = linkind(ψ_c, s_max)
#         if !isnothing(l_right)
#             mpo_tensors[end] = mpo_tensors[end] * delta(l_right, prime(l_right))
#         end
#     end
#
#     # 5. Fuse contiguous double-links into standard MPO single links
#     # This prevents downstream `ITensorMPS.add` or `dmrg` functions from crashing
#     for i in 1:(length(mpo_tensors) - 1)
#         shared_inds = commoninds(mpo_tensors[i], mpo_tensors[i+1])
#         if length(shared_inds) > 1
#             C = combiner(shared_inds; tags="Link,l=$i")
#             mpo_tensors[i] *= C
#             mpo_tensors[i+1] *= dag(C)
#         end
#     end
#
#     return MPO(mpo_tensors)
# end
#
# # Convenience Overloads
reduced_density_matrix(ψ::MPS, sites::AbstractVector{Int}) = reduced_density_matrix(ψ, collect(sites))
reduced_density_matrix(ψ::MPS, subregion::UnitRange{Int}) = reduced_density_matrix(ψ, collect(subregion))
reduced_density_matrix(ψ::MPS, start_idx::Int, end_idx::Int) = reduced_density_matrix(ψ, start_idx:end_idx)
end # module DensityMatrix
