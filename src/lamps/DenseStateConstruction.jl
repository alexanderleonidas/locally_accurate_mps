module DenseStateConstruction

using LinearAlgebra
using ITensors
using ITensorMPS
using ..MarkerGeometry
using ..DensityMatrix
using ..AKLT

export dense_construct_phi_j, combine_dense_states, combine_states_mpo, dense_purify_state, build_mps_global_approximator, run_demo

"""
    dense_construct_phi_j(target_mps::MPS, layout::AbstractMarkerLayout, j::Int) -> ITensor

Constructs the unnormalized state |φ_j⟩ for a given shift j, applying Lemma 5 iteratively across
non-marker blocks A_{j,i}. Supports both MarkerLayout and SharedMarkerLayout.
"""
function dense_construct_phi_j(target_mps::MPS, layout::AbstractMarkerLayout, j::Int)::ITensor
    sb = build_blocks(layout, j)

    tau_blocks = Union{ITensor, MPO}[]

    for block in sb.A
        isempty(block) && continue
        # Shift 0-based layout site indices to 1-based ITensor indices
        start_idx = minimum(block) + 1
        stop_idx = maximum(block) + 1
        # Store the reduced density matrix/operator
        mpo = reduced_density_matrix(target_mps, start_idx, stop_idx)
        push!(tau_blocks, mpo)
    end

    if length(tau_blocks) < 2
        if isempty(tau_blocks)
            return ITensor(1.0)
        end
        return tau_blocks[1] isa MPO ? contract(tau_blocks[1]) : tau_blocks[1]
    end

    sigma_current = tau_blocks[1]

    # Iterate up to length(tau_blocks)-2 to reserve the final blocks for purification
    for i in 2:(length(tau_blocks) - 2)
        sigma_current = combine_states(sigma_current, tau_blocks[i]; tol=1e-12)
    end

    return sigma_current isa MPO ? contract(sigma_current) : sigma_current
end

"""
    combine_dense_states(sigma::Union{ITensor, MPO}, tau::Union{ITensor, MPO}; tol::Float64=1e-12) -> ITensor

Applies Dalzell & Brandão's Lemma 5 algorithm to two reduced density operators (ITensors or MPOs)
on subsystems A and B.

Iteratively constructs joint pure state components |u_j> from the spectral
decompositions of remaining marginals alpha_j and beta_j, subtracting their
reduced density matrices until alpha_m = beta_m = 0.

Returns sigma_AB = sum_{j=1}^m |u_j><u_j|.
"""
function combine_dense_states(σ::Union{ITensor, MPO}, τ::Union{ITensor, MPO}; tol::Float64=1e-12)::ITensor
    sigma_tensor = σ isa MPO ? contract(σ) : σ
    tau_tensor   = τ isa MPO   ? contract(τ)   : τ

    # 1. Identify physical (unprimed) and bra/ket (primed) indices
    unprimed_A = filter(hasplev(0), inds(sigma_tensor))
    primed_A   = filter(hasplev(1), inds(sigma_tensor))

    unprimed_B = filter(hasplev(0), inds(tau_tensor))
    primed_B   = filter(hasplev(1), inds(tau_tensor))

    # Combiners to convert multi-index ITensors into 2D matrices
    c_row_A = combiner(primed_A...)
    c_col_A = combiner(unprimed_A...)

    c_row_B = combiner(primed_B...)
    c_col_B = combiner(unprimed_B...)

    mat_A = matrix(sigma_tensor * c_row_A * c_col_A, combinedind(c_row_A), combinedind(c_col_A))
    mat_B = matrix(tau_tensor * c_row_B * c_col_B, combinedind(c_row_B), combinedind(c_col_B))

    # Round 1 initialization: alpha_1 = tau_A, beta_1 = tau_B
    alpha = Matrix((mat_A + mat_A') / 2)
    beta  = Matrix((mat_B + mat_B') / 2)

    dim_A = size(alpha, 1)
    dim_B = size(beta, 1)

    # Accumulator for sigma_AB
    sigma_AB_mat = zeros(ComplexF64, dim_A * dim_B, dim_A * dim_B)

    # Round j loop until alpha_{m+1} = beta_{m+1} = 0
    while tr(alpha) > tol && tr(beta) > tol
        # Step 1: Spectral decomposition
        F_A = eigen(Hermitian(alpha))
        F_B = eigen(Hermitian(beta))

        # Sort non-decreasing
        p_A = sortperm(F_A.values)
        p_B = sortperm(F_B.values)

        lambda_all = F_A.values[p_A]
        s_vecs_all = F_A.vectors[:, p_A]

        mu_all     = F_B.values[p_B]
        r_vecs_all = F_B.vectors[:, p_B]

        # Filter positive non-zero eigenvalues above tolerance
        idx_A = findall(x -> x > tol, lambda_all)
        idx_B = findall(x -> x > tol, mu_all)

        (isempty(idx_A) || isempty(idx_B)) && break

        lambda = lambda_all[idx_A]
        s_vecs = s_vecs_all[:, idx_A]

        mu     = mu_all[idx_B]
        r_vecs = r_vecs_all[:, idx_B]

        a_j = length(lambda)
        b_j = length(mu)
        min_rank = min(a_j, b_j)

        # Step 2: Construct |u_j> = sum_{i=1}^{min(a_j, b_j)} sqrt(min(lambda_i, mu_i)) |s_i>_A \otimes |r_i>_B
        u_j = zeros(ComplexF64, dim_A * dim_B)
        tr_B_u = zeros(ComplexF64, dim_A, dim_A)
        tr_A_u = zeros(ComplexF64, dim_B, dim_B)

        for i in 1:min_rank
            gamma_i = sqrt(min(lambda[i], mu[i]))
            s_i = s_vecs[:, i]
            r_i = r_vecs[:, i]

            # Direct Kronecker product state vector
            u_j .+= gamma_i .* kron(r_i, s_i)

            # Direct partial traces using orthonormal basis projections
            tr_B_u .+= (gamma_i^2) .* (s_i * s_i')
            tr_A_u .+= (gamma_i^2) .* (r_i * r_i')
        end

        # Step 3: Update sigma_AB += |u_j><u_j|
        sigma_AB_mat .+= (u_j * u_j')

        # Step 4: Recursion relations
        alpha .-= tr_B_u
        beta  .-= tr_A_u
    end

    # Reconstruct combined ITensor
    all_primed   = unique(vcat(primed_A, primed_B))
    all_unprimed = unique(vcat(unprimed_A, unprimed_B))

    c_row_AB = combiner(all_primed...)
    c_col_AB = combiner(all_unprimed...)

    res_tensor = itensor(sigma_AB_mat, combinedind(c_row_AB), combinedind(c_col_AB))
    return res_tensor * c_row_AB * c_col_AB
end

"""
    combine_states_mpo(σ, τ; tol=1e-12) -> MPO

Dalzell & Brandão's Lemma 5, returning an MPO.

Given RDMs σ on subsystem A and τ on subsystem B (ITensor or MPO), returns
σ_AB = Σ_j |u_j><u_j| as an MPO on A ∪ B, such that

    Tr_B(σ_AB) = σ        Tr_A(σ_AB) = τ.

The MPO tensor order is the chain order inferred from the site tags.  The
loop is done natively: α and β are updated in place via rank-1 subtractions,
and only the (small) per-iteration eigenvectors are ever materialized.
"""
function combine_states_mpo(σ::Union{ITensor,MPO}, τ::Union{ITensor,MPO}; tol::Float64 = 1e-12)::MPO

    # Extract eigenvalues from the D returned by ITensor's `eigen`.  Depending
    # on ITensorMPS version D may be a 1-index vector or a 2-index diagonal.
    function _diag_evals(D::ITensor)
        if length(inds(D)) == 1
            return [real(scalar(D * onehot(inds(D)[1] => k))) for k in 1:dim(D, 1)]
        else
            return real.(diag(matrix(D)))
        end
    end

    # Chain order inferred from the "n=N" tag that siteind attaches.
    # function _chain_order(sites::Vector{<:Index})
    #     sort(sites, by = i -> begin
    #         m = match(r"n=(\d+)", string(tags(i)))
    #         m === nothing ? typemax(Int) : parse(Int, m.captures[1])
    #     end)
    # end

    σt = σ isa MPO ? contract(σ) : copy(σ)
    τt = τ isa MPO ? contract(τ) : copy(τ)

    # Ket (plev=0) and bra (plev=1) indices on each subsystem, paired by id.
    A_ket = sort(filter(i -> plev(i) == 0, inds(σt)),
                 by = i -> (string(tags(i)), id(i)))
    A_bra = [prime(i) for i in A_ket]
    B_ket = sort(filter(i -> plev(i) == 0, inds(τt)),
                 by = i -> (string(tags(i)), id(i)))
    B_bra = [prime(i) for i in B_ket]

    # Explicit trace bookkeeping (no extra contractions inside the loop).
    tr_α = real(scalar(σt * δ(A_bra..., A_ket...)))
    tr_β = real(scalar(τt * δ(B_bra..., B_ket...)))

    terms = Tuple{Float64,ITensor,ITensor}[]   # (γ², |s_i⟩_A, |r_i⟩_B)

    for _ in 1:10_000
        (tr_α < tol || tr_β < tol) && break

        D_A, U_A = eigen(σt, A_ket, A_bra; ishermitian = true)
        D_B, U_B = eigen(τt, B_ket, B_bra; ishermitian = true)

        eig_A = only(setdiff(inds(U_A), A_ket))
        eig_B = only(setdiff(inds(U_B), B_ket))

        λ = _diag_evals(D_A)
        μ = _diag_evals(D_B)

        idx_A = findall(>(tol), λ)
        idx_B = findall(>(tol), μ)
        (isempty(idx_A) || isempty(idx_B)) && break

        for i in 1:min(length(idx_A), length(idx_B))
            kA, kB = idx_A[i], idx_B[i]
            γ² = min(λ[kA], μ[kB])

            s_i = U_A * onehot(eig_A => kA)   # ITensor on A_ket
            r_i = U_B * onehot(eig_B => kB)   # ITensor on B_ket

            push!(terms, (γ², s_i, r_i))

            # Native update of remaining marginals:
            #   α ← α − γ²|s_i⟩⟨s_i|,   β ← β − γ²|r_i⟩⟨r_i|
            σt -= γ² * (s_i * prime(dag(s_i)))
            τt -= γ² * (r_i * prime(dag(r_i)))

            tr_α -= γ²
            tr_β -= γ²
        end
    end

    isempty(terms) && error("combine_states: both RDMs are ~zero (nothing to combine).")

    # Accumulate σ_AB = Σ_j γ²_j |u_j⟩⟨u_j| with |u_j⟩ = |s_j⟩_A ⊗ |r_j⟩_B.
    # Each term is rank-1, so we only ever pay for the local eigenvectors.
    σ_AB = let (γ², s_i, r_i) = terms[1]
        u = s_i * r_i
        γ² * u * prime(dag(u))
    end
    for k in 2:length(terms)
        γ², s_i, r_i = terms[k]
        u = s_i * r_i
        σ_AB += γ² * u * prime(dag(u))
    end

    # Factorize the ITensor into an MPO on the chain-ordered sites of A ∪ B.
    # sites = _chain_order(vcat(A_ket, B_ket))
    sites = sort(vcat(A_ket, B_ket), by = i -> begin
        m = match(r"n=(\d+)", string(tags(i)))
        m === nothing ? typemax(Int) : parse(Int, m.captures[1])
    end)
    return MPO(σ_AB, sites)
end


"""
    purify_state(sigma::ITensor, target_mps::MPS, layout::AbstractMarkerLayout, j::Int) -> MPS

Purifies the mixed state σ_{n-2} into a global pure state |Q_j⟩ using an MPS-native
construction without ever allocating a rank-N dense ITensor.
"""
function dense_purify_state(σ::ITensor, target_mps::MPS, layout::AbstractMarkerLayout, j::Int)::MPS
    sb = build_blocks(layout, j)
    Aj, Aj1 = purification_blocks(layout, sb)

    # 1. Identify purification and active sites
    purification_sites = sort(union(Aj, Aj1))
    purif_inds = [siteind(target_mps, s + 1) for s in purification_sites]

    unprimed_inds = filter(hasplev(0), inds(σ))
    primed_inds   = filter(hasplev(1), inds(σ))

    D, U = eigen(σ, unprimed_inds, primed_inds; ishermitian=true, cutoff=1e-12)

    link_ind = commonind(D, U)
    rank = dim(link_ind)

    total_purif_dim = prod(dim.(purif_inds))
    if rank > total_purif_dim
        error("Purification sink too small! Rank ($rank) > Sink Dimension ($total_purif_dim). Increase l.")
    end

    # 2. Prepare local purification sink tensor
    purif_tensor = ITensor(link_ind, purif_inds...)
    cart_indices = CartesianIndices(Tuple(dim.(purif_inds)))
    for k in 1:rank
        c_idx = cart_indices[k]
        assignment = [purif_inds[i] => c_idx[i] for i in eachindex(1:length(purif_inds))]
        purif_tensor[assignment..., link_ind => k] = 1.0
    end

    sqrt_D = map(x -> x > 0 ? sqrt(x) : 0.0, D)
    global_pure_tensor = U * sqrt_D * purif_tensor

    # 3. Determine marker values for all N sites
    N = length(target_mps)
    marker_vals = fill(0, N) # 0 = active/bulk site, 1 = |0>, 2 = |1>

    if layout isa MarkerLayout
        l = layout.l
        for jp in 0:(l-1)
            jp == j && continue
            site0 = layout.B[j+1, jp+1] + 1
            site1 = layout.B[jp+1, j+1] + 1
            marker_vals[site0] = 1
            marker_vals[site1] = min(2, dim(siteind(target_mps, site1)))
        end
    elseif layout isa SharedMarkerLayout
        for mk in layout.markers
            s_idx = mk.site + 1
            if j in mk.S0
                marker_vals[s_idx] = 1
            elseif j in mk.S1
                marker_vals[s_idx] = min(2, dim(siteind(target_mps, s_idx)))
            end
        end
    end

    # 4. Decompose local pure tensor over active sites into an MPS fragment
    active_sites = [siteind(target_mps, s) for s in 1:N if marker_vals[s] == 0]
    local_mps = MPS(global_pure_tensor, active_sites; cutoff=1e-12)

    # 5. Assemble the global N-site MPS without large dense contractions
    full_mps = MPS(N)
    active_ptr = 1

    for s in 1:N
        s_ind = siteind(target_mps, s)
        if marker_vals[s] != 0
            # Insert product state tensor (|0> or |1>) with trivial link indices
            val = marker_vals[s]
            t = ITensor(s_ind)
            t[s_ind => val] = 1.0
            full_mps[s] = t
        else
            # Insert tensor from local active MPS
            full_mps[s] = local_mps[active_ptr]
            active_ptr += 1
        end
    end

    return full_mps
end

"""
    build_global_approximator(phi_states::Vector{MPS}; maxdim::Int=200, cutoff::Float64=1e-10) -> MPS

Sums the shift states |φ_j⟩ to construct the complete local approximator |ψ̃⟩ as a single MPS.
"""
function build_mps_global_approximator(phi_states::Vector{MPS}; maxdim::Int=200, cutoff::Float64=1e-10)::MPS
    l = length(phi_states)

    # Initialize sum with the first shift state
    psi_tilde = copy(phi_states[1])

    for j in 2:l
        # Add MPSs natively using ITensorMPS.add
        psi_tilde = add(psi_tilde, phi_states[j]; cutoff=cutoff, maxdim=maxdim)
    end

    # Apply global normalization factor (1 / √l)
    psi_tilde *= (1.0 / sqrt(l))
    normalize!(psi_tilde)

    return psi_tilde
end

"""
    run_demo(; k::Int=2, ε::Float64=0.9) -> MPS

Executes an end-to-end demonstration constructing all shift states |φ_j⟩ and assembling
the final local approximator |ψ̃⟩ MPS for the AKLT ground state.
"""
function run_demo(; k::Int=2, ε::Float64=0.9)::MPS
    println("=== StateConstruction Demo ===")
    println("Parameters: k = $k, ε =$ε")

    layout = MarkerLayout(k, ε)
    println("MarkerLayout: l=$(layout.l) block size, N = $(layout.N) sites, n =$(layout.n) blocks")

    sites = siteinds("S=1", layout.N)
    println("Generating target state (AKLT ground state)...")
    psi_target = aklt_ground_state(sites)

    l = layout.l
    println("Constructing $l shift states |φ_j⟩...")
    phi_states = Vector{MPS}(undef, l)
    for j in 0:(l - 1)
        sigma_j = construct_phi_j(psi_target, layout, j)
        phi_states[j + 1] = purify_state(sigma_j, psi_target, layout, j)
        println("  Shift j = $j constructed.")
    end

    println("Building global approximator state |ψ̃⟩...")
    psi_tilde = build_global_approximator(phi_states; maxdim=200, cutoff=1e-10)

    println("\n=== Construction Summary ===")
    println("Shift states generated: ", length(phi_states))
    println("Approximator norm: ", norm(psi_tilde))
    println("Approximator max bond dimension: ", maximum(linkdims(psi_tilde)))

    return psi_tilde
end

end # module DenseStateConstruction
