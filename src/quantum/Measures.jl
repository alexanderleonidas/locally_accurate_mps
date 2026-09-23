module Measures

using ITensors
using ITensorMPS
using LinearAlgebra
using ..DensityMatrix


export purified_distance,
    trace_distance,
    purified_distance,
    k_local_trace_distance,
    k_local_purified_distance,
    frobenius_norm,
    purity


# ------------------------------------------------------------------------------
# Purified Distance for MPS and MPOs
# ------------------------------------------------------------------------------

"""
    purified_distance(ψ::MPS, φ::MPS) -> Float64

Compute the purified distance D(ρ, σ) = sqrt(1 - F(ρ, σ)^2) = sqrt(1 - |⟨ψ|φ⟩|^2) between two MPS.
F(ρ, σ) = tr(sqrt(sqrt(ρ) * σ * sqrt(ρ))) is the fidelity, and ρ = |ψ⟩⟨ψ| and σ = |φ⟩⟨φ|.
"""
function purified_distance(ψ::MPS, φ::MPS)::Float64
    @assert length(ψ) == length(φ) "MPS psi and phi must have the same length."
    overlap = abs(inner(ψ, φ)) / (norm(ψ) * norm(φ))
    return sqrt(max(0.0, 1.0 - overlap^2))
end

"""
    purified_distance(ρ::MPO, σ::MPO) -> Float64

Compute the purified distance D(ρ, σ) = sqrt(1 - F(ρ, σ)^2) between two MPOs.
Contracts local MPOs to dense matrices for exact eigen-decomposition.
Use for small MPOs only, as this is not scalable for large systems.
"""
function purified_distance(ρ::MPO, σ::MPO)::Float64
    @assert length(ρ) == length(σ) "MPO rho and sigma must have the same length."

    # Helper to convert MPO to a dense Julia Matrix
    mpo_to_matrix(mpo::MPO) = begin
        T = contract(mpo)
        c_row = combiner(inds(T; plev=1)...)
        c_col = combiner(inds(T; plev=0)...)
        return matrix(T * c_row * c_col, combinedind(c_row), combinedind(c_col))
    end

    ρ_mat = mpo_to_matrix(ρ)
    σ_mat = mpo_to_matrix(σ)

    # Exact matrix square root for positive semi-definite matrices
    sqrt_ρ = sqrt(Hermitian((ρ_mat + ρ_mat') / 2))
    core_mat = sqrt_ρ * σ_mat * sqrt_ρ
    sqrt_core = sqrt(Hermitian((core_mat + core_mat') / 2))

    f = real(tr(sqrt_core))
    return sqrt(max(0.0, 1.0 - min(1.0, f)^2))
end

"""
    k_local_purified_distance(mps1::MPS, mps2::MPS, k::Int) -> Float64

Compute maximum k-local purified distance D^(k)(ψ, φ) = max_{X: |X|=k} D(ψ_X, φ_X) across all contiguous subregions of size k.
"""
function k_local_purified_distance(ψ::MPS, φ::MPS, k::Int)::Float64
    N = length(ψ)
    @assert length(φ) == length(ψ) "MPS ψ and φ must have the same length."
    @assert siteinds(ψ) == siteinds(φ) "MPS ψ and φ must have the same site indices."

    if k < 1 || k > N
        error("Subsystem size k must be between 1 and N")
    end

    max_dist = 0.0

    for start in 1:(N - k + 1)
        stop = start + k - 1
        # Get the MPS for the subsystem
        ρ = reduced_density_matrix(ψ, start, stop)
        σ = reduced_density_matrix(φ, start, stop)

        dist = purified_distance(ρ, σ)
        max_dist = max(dist, max_dist)
    end

    return max_dist
end

# ------------------------------------------------------------------------------
# Frobenius Norm Evaluation for MPS and MPOs
# ------------------------------------------------------------------------------

"""
    frobenius_norm(mps1::MPS, mps2::MPS) -> Float64

Compute Frobenius norm ||ρ1 - ρ2||_F directly using MPS overlaps.
"""
function frobenius_norm(mps1::MPS, mps2::MPS)::Float64
    n1 = norm(mps1)^2
    n2 = norm(mps2)^2
    overlap = abs(inner(mps1, mps2))^2

    val = n1 + n2 - 2 * overlap
    return sqrt(max(0.0, val))
end

"""
    frobenius_norm(mpo1::MPO, mpo2::MPO) -> Float64

Compute Frobenius norm ||A - B||_F = sqrt(Tr[(A - B)†(A - B)]) for two MPOs
natively via inner products.
"""
function frobenius_norm(mpo1::MPO, mpo2::MPO)::Float64
    tr11 = real(inner(mpo1, mpo1))
    tr22 = real(inner(mpo2, mpo2))
    tr12 = real(inner(mpo1, mpo2))
    return sqrt(max(0.0, tr11 + tr22 - 2 * tr12))
end

# ------------------------------------------------------------------------------
# Pure State Verification & Purity Evaluation
# ------------------------------------------------------------------------------

"""
    purity(ρ::MPS) -> Float64

Purity γ = Tr(ρ²) for a density matrix MPO.
"""
function purity(ψ::MPS)::Float64
   return purity(outer(ψ', ψ))  # Convert MPS to MPO and compute purity
end

"""
    purity(ρ::MPO) -> Float64

Compute the purity γ = Tr(ρ²) for a density matrix MPO.
"""
function purity(ρ::MPO)::Float64
    return tr(inner(ρ, ρ))
end

# ------------------------------------------------------------------------------
# Trace Distance Evaluation for MPS and MPOs
# ------------------------------------------------------------------------------

"""
    trace_distance(rho::ITensor, sigma::ITensor) -> Float64

Compute trace distance D(ρ, σ) = 1/2 * ||ρ - σ||_1 for local ITensors
using native ITensor SVD decomposition without converting to dense matrices.
"""
function trace_distance(ρ::ITensor, σ::ITensor)::Float64
    Δ = ρ - σ

    # Partition primed indices (row space) from unprimed indices (col space)
    all_inds = inds(Δ)
    row_inds = filter(i -> plev(i) > 0, all_inds)

    if isempty(row_inds)
        error("ITensor must have primed indices representing the row space.")
    end

    # Perform native ITensor SVD: returns U, S, V where S is a diagonal ITensor
    _, S, _ = svd(Δ, row_inds)

    # Extract singular values from the diagonal ITensor S
    s_inds = inds(S)
    n = dim(s_inds[1])
    sv = [S[i, i] for i in 1:n]

    return 0.5 * sum(sv)
end

"""
    trace_distance(mpo1::MPO, mpo2::MPO) -> Float64

Compute trace distance between two local MPOs by contracting them into single ITensors.
"""
function trace_distance(ψ::MPO, φ::MPO)::Float64
    T1 = contract(ψ)
    T2 = contract(φ)
    return trace_distance(T1, T2)
end

"""
    trace_distance(mps1::MPS, mps2::MPS) -> Float64

Compute trace distance between two full MPS states.
"""
function trace_distance(ψ::MPS, φ::MPS)::Float64
    N = length(ψ)
    if length(φ) != N
        error("MPS lengths must match")
    end
    ρ = reduced_density_matrix(ψ, 1, N)
    σ = reduced_density_matrix(φ, 1, N)
    return trace_distance(ρ, σ)
end

"""
    k_local_trace_distance(mps1::MPS, mps2::MPS, k::Int) -> Float64

Compute maximum k-local trace distance across all contiguous subregions of size k.
"""
function k_local_trace_distance(ψ::MPS, φ::MPS, k::Int)::Float64
    N = length(ψ)
    @assert length(φ) == length(ψ) "MPS ψ and φ must have the same length."

    if k < 1 || k > N
        error("Subsystem size k must be between 1 and N")
    end

    max_dist = 0.0

    for start in 1:(N - k + 1)
        stop = start + k - 1
        ρ = reduced_density_matrix(ψ, start, stop)
        σ = reduced_density_matrix(φ, start, stop)

        dist = trace_distance(ρ, σ)
        max_dist = max(dist, max_dist)
    end

    return max_dist
end

end # module Measures
