# module StateConstruction
#
# using LinearAlgebra
# using ITensors
# using ITensorMPS
# using ..MarkerGeometry
# using ..DensityMatrix
# using ..AKLT
#
# export construct_phi_j, combine_states, purify_state, build_global_approximator, run_demo
#
# """
#     construct_phi_j(target_mps::MPS, layout::AbstractMarkerLayout, j::Int) -> MPO
#
# Constructs the unnormalized state |φ_j⟩ for a given shift j as an MPO, applying
# Lemma 5 iteratively across non-marker blocks using 1-based circular indexing (mod1).
# """
# function construct_phi_j(target_mps::MPS, layout::AbstractMarkerLayout, j::Int)::MPO
#     sb = build_blocks(layout, j)
#     tau_blocks = MPO[]
#
#     for block in sb.A
#         isempty(block) && continue
#
#         # Pass the exact list of 1-based indices instead of a min/max range
#         block_sites = sort(block .+ 1)
#
#         # Store the reduced density matrix natively as an MPO
#         mpo = reduced_density_matrix(target_mps, block_sites)
#         push!(tau_blocks, mpo)
#     end
#
#     num_blocks = length(tau_blocks)
#     if num_blocks < 2
#         return isempty(tau_blocks) ? MPO() : tau_blocks[1]
#     end
#
#     # Use mod1 for strict 1-based Julia index wrapping (1..num_blocks)
#     start_idx = mod1(j + 2, num_blocks)
#     sigma_current = tau_blocks[start_idx]
#
#     # Iterate through remaining blocks
#     for p in 1:(num_blocks - 3)
#         i = mod1(j + p + 2, num_blocks)
#         sigma_current = combine_states(sigma_current, tau_blocks[i]; tol=1e-8)
#
#         # Crucial: Truncate after combination to prevent exponential bond dimension blowup
#         truncate!(sigma_current; cutoff=1e-10, maxdim=64)
#     end
#
#     return sigma_current
# end
#
# """
#     get_mpo_eigenpairs(mpo::MPO; tol::Float64=1e-8, max_states::Int=20)
#
# Extracts the dominant eigenpairs of a positive semi-definite MPO by running deflated
# DMRG on the negative MPO. Returns eigenvalues and eigenvectors (as MPS).
# """
# function get_mpo_eigenpairs(mpo::MPO; tol::Float64=1e-8, max_states::Int=20)
#     sites = [filter(hasplev(0), inds(mpo[i]; tags="Site"))[1] for i in eachindex(1:length(mpo))]
#
#     # DMRG minimizes H, so we set H = -mpo to find largest eigenvalues
#     minus_mpo = MPO(length(mpo))
#     for i in eachindex(1:length(mpo))
#         minus_mpo[i] = mpo[i] * (i == 1 ? -1.0 : 1.0)
#     end
#
#     eigenvals = Float64[]
#     eigenstates = MPS[]
#
#     sweeps = Sweeps(4)
#     setmaxdim!(sweeps, 10, 20, 50, 100)
#     setcutoff!(sweeps, 1e-12)
#
#     for s in 1:max_states
#         psi0 = randomMPS(sites; linkdims=2)
#
#         # Use weight penalty orthogonalization to find sequential excited states
#         energy, psi = if isempty(eigenstates)
#             dmrg(minus_mpo, psi0, sweeps; outputlevel=0)
#         else
#             dmrg(minus_mpo, eigenstates, psi0, sweeps; weight=10.0, outputlevel=0)
#         end
#
#         val = -energy
#         if val < tol
#             break
#         end
#
#         push!(eigenvals, val)
#         push!(eigenstates, psi)
#     end
#
#     return eigenvals, eigenstates
# end
#
# """
#     join_mps(mpsA::MPS, mpsB::MPS) -> MPS
#
# Concatenates two disjoint MPSs into a single product-state MPS connected by a
# dummy link of dimension 1.
# """
# function join_mps(mpsA::MPS, mpsB::MPS)::MPS
#     N_A = length(mpsA)
#     N_B = length(mpsB)
#     tensors = Vector{ITensor}(undef, N_A + N_B)
#
#     for i in 1:N_A
#         tensors[i] = mpsA[i]
#     end
#     for i in 1:N_B
#         tensors[N_A + i] = mpsB[i]
#     end
#
#     l_dummy = Index(1, "Link,dummy")
#     t_dummy_A = ITensor(l_dummy)
#     t_dummy_A[l_dummy => 1] = 1.0
#     tensors[N_A] *= t_dummy_A
#
#     t_dummy_B = ITensor(l_dummy)
#     t_dummy_B[l_dummy => 1] = 1.0
#     tensors[N_A+1] *= t_dummy_B
#
#     return MPS(tensors)
# end
#
# """
#     combine_states(σ::MPO, τ::MPO; tol::Float64=1e-8) -> MPO
#
# Applies Dalzell & Brandão's Lemma 5 algorithm to two reduced density operators (MPOs)
# on disjoint subsystems A and B, exploiting the static eigenbases property to operate
# via a scalable scalar loop over eigenvalues.
# """
# function combine_states(σ::MPO, τ::MPO; tol::Float64=1e-8)::MPO
#     sites_A = [filter(hasplev(0), inds(σ[i]; tags="Site"))[1] for i in eachindex(1:length(σ))]
#     sites_B = [filter(hasplev(0), inds(τ[i]; tags="Site"))[1] for i in eachindex(1:length(τ))]
#     sites_AB = vcat(sites_A, sites_B)
#     N_AB = length(sites_AB)
#
#     println("  Extracting static eigenpairs for Subsystem A...")
#     lambda, s_mps = get_mpo_eigenpairs(σ; tol=tol)
#     println("  Extracting static eigenpairs for Subsystem B...")
#     mu, r_mps = get_mpo_eigenpairs(τ; tol=tol)
#
#     sigma_idx = collect(1:length(lambda))
#     tau_idx = collect(1:length(mu))
#
#     u_mpos = MPO[]
#     round_j = 1
#
#     while sum(lambda) > tol && sum(mu) > tol
#         # 1. Sort eigenvalues in non-decreasing order and apply permutation
#         p_A = sortperm(lambda)
#         lambda = lambda[p_A]
#         sigma_idx = sigma_idx[p_A]
#
#         p_B = sortperm(mu)
#         mu = mu[p_B]
#         tau_idx = tau_idx[p_B]
#
#         n_j = min(length(lambda), length(mu))
#         n_j == 0 && break
#
#         u_j_components = MPS[]
#
#         # 2. Scalar loop
#         for i in 1:n_j
#             c = min(lambda[i], mu[i])
#             lambda[i] -= c^2
#             mu[i] -= c^2
#
#             if c > 0
#                 prod_mps = join_mps(s_mps[sigma_idx[i]], r_mps[tau_idx[i]])
#                 prod_mps[1] *= c
#                 push!(u_j_components, prod_mps)
#             end
#         end
#
#         # Remove eigenvalues below tolerance
#         keep_A = lambda .>= tol
#         lambda = lambda[keep_A]
#         sigma_idx = sigma_idx[keep_A]
#
#         keep_B = mu .>= tol
#         mu = mu[keep_B]
#         tau_idx = tau_idx[keep_B]
#
#         isempty(u_j_components) && break
#
#         # 3. Build |u_j> as an MPS by accumulating scaled product states
#         u_j_mps = u_j_components[1]
#         for i in 2:length(u_j_components)
#             u_j_mps = add(u_j_mps, u_j_components[i]; cutoff=1e-12)
#         end
#
#         # 4. Build MPO for |u_j><u_j|
#         u_j_bra = dag(prime(u_j_mps)) # Primes both site and link indices
#         tensors = ITensor[]
#         for i in 1:N_AB
#             push!(tensors, u_j_mps[i] * u_j_bra[i])
#         end
#
#         # Fuse contiguous double-links into standard MPO single links
#         for i in 1:(N_AB - 1)
#             shared_inds = commoninds(tensors[i], tensors[i+1])
#             if length(shared_inds) > 1
#                 C = combiner(shared_inds; tags="Link,l=$i")
#                 tensors[i] *= C
#                 tensors[i+1] *= dag(C)
#             end
#         end
#
#         push!(u_mpos, MPO(tensors))
#         round_j += 1
#     end
#
#     if isempty(u_mpos)
#         return 0.0 * MPO(sites_AB, "Id")
#     end
#
#     # 5. Accumulate final Output
#     sigma_AB = u_mpos[1]
#     for i in 2:length(u_mpos)
#         sigma_AB = add(sigma_AB, u_mpos[i]; cutoff=1e-8)
#     end
#
#     trace_error = abs(real(tr(sigma_AB)) - 1.0)
#     max_bd = maximum(linkdims(sigma_AB))
#
#     println("  -> Completed in $(round_j-1) rounds.")
#     println("  -> Final MPO Max Bond Dimension: $max_bd (Theoretical limit: O(χ^3))")
#     println("  -> Trace error ε: $trace_error \n")
#
#     return sigma_AB
# end
#
# """
#     purify_mpo_to_mps(mpo::MPO; cutoff=1e-10, maxdim=64) -> MPS
#
# Purifies an MPO into an MPS site-by-site, enforcing dense storage allocation
# to prevent Julia 1.13 LinearAlgebra.svd dispatch errors on empty tensors.
# """
# function purify_mpo_to_mps(mpo::MPO; cutoff::Float64=1e-12, maxdim::Int=10000)::MPS
#     N = length(mpo)
#     mps_tensors = Vector{ITensor}(undef, N)
#
#     env = ITensor(1.0)
#     for k in 1:N
#         W = mpo[k]
#
#         T = env * W
#
#         # Ensure T is dense to prevent Julia 1.13 EmptyTensor SVD algorithm error
#         if isempty(storage(T)) || norm(T) == 0.0
#             T = dense(T)
#         end
#
#         s0_inds = filter(hasplev(0), inds(T; tags="Site"))
#         left_link_inds = filter(i -> !hasplev(1)(i) && !hasplev(0)(i), inds(T))
#
#         split_inds = union(s0_inds, left_link_inds)
#
#         U, S, V = svd(T, split_inds; cutoff=cutoff, maxdim=maxdim)
#
#         mps_tensors[k] = U
#         env = S * V
#     end
#
#     res_mps = MPS(mps_tensors)
#     truncate!(res_mps; cutoff=cutoff, maxdim=maxdim)
#     return res_mps
# end
#
# """
#     purify_state(sigma_n_minus_2::MPO, target_mps::MPS, layout::AbstractMarkerLayout, j::Int) -> MPS
#
# Purifies the active subregion MPO directly into a global N-site MPS.
# """
# function purify_state(sigma_n_minus_2::MPO, target_mps::MPS, layout::AbstractMarkerLayout, j::Int; cutoff::Float64=1e-12, maxdim::Int=10000)::MPS
#     N = length(target_mps)
#
#     marker_vals = fill(0, N)
#     if layout isa MarkerLayout
#         l = layout.l
#         for jp in 0:(l-1)
#             jp == j && continue
#             site0 = layout.B[j+1, jp+1] + 1
#             site1 = layout.B[jp+1, j+1] + 1
#             marker_vals[site0] = 1
#             marker_vals[site1] = min(2, dim(siteind(target_mps, site1)))
#         end
#     elseif layout isa SharedMarkerLayout
#         for mk in layout.markers
#             s_idx = mk.site + 1
#             if j in mk.S0
#                 marker_vals[s_idx] = 1
#             elseif j in mk.S1
#                 marker_vals[s_idx] = min(2, dim(siteind(target_mps, s_idx)))
#             end
#         end
#     end
#
#     local_mps = purify_mpo_to_mps(sigma_n_minus_2; cutoff=cutoff, maxdim=maxdim)
#
#     full_mps = MPS(N)
#     active_ptr = 1
#
#     for s in 1:N
#         s_ind = siteind(target_mps, s)
#         if marker_vals[s] != 0
#             val = marker_vals[s]
#             t = ITensor(s_ind)
#             t[s_ind => val] = 1.0
#             full_mps[s] = t
#         else
#             if active_ptr <= length(local_mps)
#                 full_mps[s] = local_mps[active_ptr]
#                 active_ptr += 1
#             else
#                 t = ITensor(s_ind)
#                 t[s_ind => 1] = 1.0
#                 full_mps[s] = t
#             end
#         end
#     end
#
#     orthogonalize!(full_mps, 1)
#     normalize!(full_mps)
#
#     return full_mps
# end
#
# """
#     build_global_approximator(phi_states::Vector{MPS}; maxdim::Int=200, cutoff::Float64=1e-10) -> MPS
#
# Sums the shift states |φ_j⟩ to construct the complete local approximator |ψ̃⟩ strictly via
# native MPS addition functions.
# """
# function build_global_approximator(phi_states::Vector{MPS}; maxdim::Int=200, cutoff::Float64=1e-12)::MPS
#     l = length(phi_states)
#
#     psi_tilde = copy(phi_states[1])
#
#     for j in 2:l
#         psi_tilde = add(psi_tilde, phi_states[j]; cutoff=cutoff, maxdim=maxdim)
#     end
#
#     psi_tilde *= (1.0 / sqrt(l))
#     normalize!(psi_tilde)
#
#     return psi_tilde
# end
#
# """
#     run_demo(; k::Int=2, ε::Float64=0.9) -> MPS
# """
# function run_demo(; k::Int=2, ε::Float64=0.9)::MPS
#     println("=== StateConstruction Demo ===")
#     println("Parameters: k = $k, ε =$ε")
#
#     layout = MarkerLayout(k, ε)
#     println("MarkerLayout: l=$(layout.l) block size, N = $(layout.N) sites, n =$(layout.n) blocks")
#
#     sites = siteinds("S=1", layout.N)
#     println("Generating target state (AKLT ground state)...")
#     psi_target = aklt_ground_state(sites)
#
#     l = layout.l
#     println("Constructing $l shift states |φ_j⟩ (MPO-Native)...")
#     phi_states = Vector{MPS}(undef, l)
#     for j in 0:(l - 1)
#         sigma_j_mpo = construct_phi_j(psi_target, layout, j)
#         phi_states[j + 1] = purify_state(sigma_j_mpo, psi_target, layout, j)
#         println("  Shift j = $j constructed.")
#     end
#
#     println("Building global approximator state |ψ̃⟩...")
#     psi_tilde = build_global_approximator(phi_states; maxdim=200, cutoff=1e-12)
#
#     println("\n=== Construction Summary ===")
#     println("Shift states generated: ", length(phi_states))
#     println("Approximator norm: ", norm(psi_tilde))
#     println("Approximator max bond dimension: ", maximum(linkdims(psi_tilde)))
#
#     return psi_tilde
# end

# end # module StateConstruction

module StateConstruction

using LinearAlgebra
using ITensors
using ITensorMPS
using ..MarkerGeometry
using ..DensityMatrix
using ..AKLT

export construct_phi_j, combine_states, purify_state, build_global_approximator, run_demo

"""
    construct_phi_j(target_mps::MPS, layout::AbstractMarkerLayout, j::Int) -> MPO

Constructs the unnormalized state |φ_j⟩ for a given shift j as an MPO, applying
Lemma 5 iteratively across non-marker blocks using 1-based circular indexing (mod1).
"""
function construct_phi_j(target_mps::MPS, layout::AbstractMarkerLayout, j::Int)::MPO
    sb = build_blocks(layout, j)

    tau_blocks = MPO[]

    for block in sb.A
        isempty(block) && continue
        # Shift 0-based layout site indices to 1-based ITensor indices
        start_idx = minimum(block) + 1
        stop_idx = maximum(block) + 1

        # Store the reduced density matrix natively as an MPO
        mpo = reduced_density_matrix(target_mps, start_idx, stop_idx)
        push!(tau_blocks, mpo)
    end

    num_blocks = length(tau_blocks)
    if num_blocks < 2
        return isempty(tau_blocks) ? MPO() : tau_blocks[1]
    end

    # Use mod1 for strict 1-based Julia index wrapping (1..num_blocks)
    start_idx = mod1(j + 2, num_blocks)
    sigma_current = tau_blocks[start_idx]

    # Iterate through remaining blocks
    for p in 1:(num_blocks - 3)
        i = mod1(j + p + 2, num_blocks)
        sigma_current = combine_states(sigma_current, tau_blocks[i]; tol=1e-8)
    end

    return sigma_current
end

"""
    combine_states(σ::MPO, τ::MPO; tol::Float64=1e-8) -> MPO

Applies Dalzell & Brandão's Lemma 5 algorithm natively to two reduced density operators
(MPOs) on subsystems A and B.
"""
function combine_states(σ::MPO, τ::MPO; tol::Float64=1e-8)::MPO
    sites_A = [filter(hasplev(0), inds(σ[i]; tags="Site"))[1] for i in eachindex(1:length(σ))]
    sites_B = [filter(hasplev(0), inds(τ[i]; tags="Site"))[1] for i in eachindex(1:length(τ))]

    sites_AB = vcat(sites_A, sites_B)
    N_A = length(σ)
    N_B = length(τ)
    N_AB = N_A + N_B

    alpha = copy(σ)
    beta  = copy(τ)

    sweeps = Sweeps(4)
    setmaxdim!(sweeps, 10, 20, 50, 100)
    setcutoff!(sweeps, 1e-12)

    u_mpos = MPO[]

    for r in 1:20
        println("  round $r: length(alpha)=$(length(alpha)), maxlinkdim(alpha)=$(maximum(linkdims(alpha)))")
        minus_alpha = MPO(length(alpha))
        for i in eachindex(1:length(alpha))
            minus_alpha[i] = alpha[i] * (i == 1 ? -1.0 : 1.0)
        end

        minus_beta = MPO(length(beta))
        for i in eachindex(1:length(beta))
            minus_beta[i] = beta[i] * (i == 1 ? -1.0 : 1.0)
        end

        psi0_A = randomMPS(sites_A; linkdims=2)
        psi0_B = randomMPS(sites_B; linkdims=2)

        energy_A, psi_A = dmrg(minus_alpha, psi0_A, sweeps; outputlevel=0)
        energy_B, psi_B = dmrg(minus_beta, psi0_B, sweeps; outputlevel=0)

        lambda = -energy_A
        mu     = -energy_B

        if lambda < tol || mu < tol
            break
        end

        gamma = sqrt(min(lambda, mu))

        rho_A = outer(prime(psi_A, "Site"), psi_A)
        rho_A[1] *= (gamma^2)

        rho_B = outer(prime(psi_B, "Site"), psi_B)
        rho_B[1] *= (gamma^2)

        minus_rho_A = MPO(length(rho_A))
        for i in eachindex(1:length(rho_A))
            minus_rho_A[i] = rho_A[i] * (i == 1 ? -1.0 : 1.0)
        end
        alpha = add(alpha, minus_rho_A; cutoff=1e-8)

        minus_rho_B = MPO(length(rho_B))
        for i in eachindex(1:length(rho_B))
            minus_rho_B[i] = rho_B[i] * (i == 1 ? -1.0 : 1.0)
        end
        beta = add(beta, minus_rho_B; cutoff=1e-8)

        u_mps = MPS(N_AB)
        for i in 1:N_A
            u_mps[i] = psi_A[i]
        end
        for i in 1:N_B
            u_mps[N_A + i] = psi_B[i]
        end

        l_dummy = Index(1, "Link,dummy")
        u_mps[N_A]   *= setelt(l_dummy => 1)
        u_mps[N_A+1] *= setelt(l_dummy => 1)

        u_mpo = outer(prime(u_mps, "Site"), u_mps)
        u_mpo[1] *= (gamma^2)

        push!(u_mpos, u_mpo)
    end
    println("  trace: ", tr(alpha), " norm: ", norm(alpha), "\n")
    if isempty(u_mpos)
        # Return a properly formed zero MPO with link indices
        z_mpo = 0.0 * MPO(sites_AB, "Id")
        return z_mpo
    end

    sigma_AB = u_mpos[1]
    for i in 2:length(u_mpos)
        sigma_AB = add(sigma_AB, u_mpos[i]; cutoff=1e-8)
    end

    return sigma_AB
end

"""
    purify_mpo_to_mps(mpo::MPO; cutoff=1e-10, maxdim=64) -> MPS

Purifies an MPO into an MPS site-by-site, enforcing dense storage allocation
to prevent Julia 1.13 LinearAlgebra.svd dispatch errors on empty tensors.
"""
function purify_mpo_to_mps(mpo::MPO; cutoff::Float64=1e-12, maxdim::Int=10000)::MPS
    N = length(mpo)
    mps_tensors = Vector{ITensor}(undef, N)

    env = ITensor(1.0)
    for k in 1:N
        W = mpo[k]

        T = env * W

        # Ensure T is dense to prevent Julia 1.13 EmptyTensor SVD algorithm error
        if isempty(storage(T)) || norm(T) == 0.0
            T = dense(T)
        end

        s0_inds = filter(hasplev(0), inds(T; tags="Site"))
        left_link_inds = filter(i -> !hasplev(1)(i) && !hasplev(0)(i), inds(T))

        split_inds = union(s0_inds, left_link_inds)

        U, S, V = svd(T, split_inds; cutoff=cutoff, maxdim=maxdim)

        mps_tensors[k] = U
        env = S * V
    end

    res_mps = MPS(mps_tensors)
    truncate!(res_mps; cutoff=cutoff, maxdim=maxdim)
    return res_mps
end

"""
    purify_state(sigma_n_minus_2::MPO, target_mps::MPS, layout::AbstractMarkerLayout, j::Int) -> MPS

Purifies the active subregion MPO directly into a global N-site MPS.
"""
function purify_state(sigma_n_minus_2::MPO, target_mps::MPS, layout::AbstractMarkerLayout, j::Int; cutoff::Float64=1e-12, maxdim::Int=10000)::MPS
    N = length(target_mps)

    marker_vals = fill(0, N)
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

    local_mps = purify_mpo_to_mps(sigma_n_minus_2; cutoff=cutoff, maxdim=maxdim)

    full_mps = MPS(N)
    active_ptr = 1

    for s in 1:N
        s_ind = siteind(target_mps, s)
        if marker_vals[s] != 0
            val = marker_vals[s]
            t = ITensor(s_ind)
            t[s_ind => val] = 1.0
            full_mps[s] = t
        else
            if active_ptr <= length(local_mps)
                full_mps[s] = local_mps[active_ptr]
                active_ptr += 1
            else
                t = ITensor(s_ind)
                t[s_ind => 1] = 1.0
                full_mps[s] = t
            end
        end
    end

    orthogonalize!(full_mps, 1)
    normalize!(full_mps)

    return full_mps
end

"""
    build_global_approximator(phi_states::Vector{MPS}; maxdim::Int=200, cutoff::Float64=1e-10) -> MPS

Sums the shift states |φ_j⟩ to construct the complete local approximator |ψ̃⟩ strictly via
native MPS addition functions.
"""
function build_global_approximator(phi_states::Vector{MPS}; maxdim::Int=200, cutoff::Float64=1e-12)::MPS
    l = length(phi_states)

    psi_tilde = copy(phi_states[1])

    for j in 2:l
        psi_tilde = add(psi_tilde, phi_states[j]; cutoff=cutoff, maxdim=maxdim)
    end

    psi_tilde *= (1.0 / sqrt(l))
    normalize!(psi_tilde)

    return psi_tilde
end

"""
    run_demo(; k::Int=2, ε::Float64=0.9) -> MPS
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
    println("Constructing $l shift states |φ_j⟩ (MPO-Native)...")
    phi_states = Vector{MPS}(undef, l)
    for j in 0:(l - 1)
        sigma_j_mpo = construct_phi_j(psi_target, layout, j)
        phi_states[j + 1] = purify_state(sigma_j_mpo, psi_target, layout, j)
        println("  Shift j = $j constructed.")
    end

    println("Building global approximator state |ψ̃⟩...")
    psi_tilde = build_global_approximator(phi_states; maxdim=200, cutoff=1e-12)

    println("\n=== Construction Summary ===")
    println("Shift states generated: ", length(phi_states))
    println("Approximator norm: ", norm(psi_tilde))
    println("Approximator max bond dimension: ", maximum(linkdims(psi_tilde)))

    return psi_tilde
end

end # module StateConstruction
