module TheoremVerification

using LinearAlgebra
using ITensors
using ITensorMPS
using ..MarkerGeometry
using ..StateConstruction
using ..DenseStateConstruction
using ..MatrixProductOperators
using ..AKLT
using ..DensityMatrix

export verify_orthogonality, debug_phi_j_construction

"""
    verify_orthogonality(k::Int, ε::AbstractFloat)

Constructs the states |φ_0⟩ and |φ_1⟩ for the AKLT ground state and checks their inner product.
"""
function verify_orthogonality(k::Integer=2, ε::AbstractFloat=0.5)
    layout = SharedMarkerLayout(k,ε, max(k,2))

    # N is determined by the MarkerLayout (N = layout.N)
    sites = siteinds("S=1", layout.N)

    println("--- Generating Exact AKLT Target State ---")
    psi_target = aklt_ground_state(sites)

    println("--- Constructing |φ_0⟩ ---")
    sigma_0 = construct_phi_j(psi_target, layout, 0)
    phi_0   = purify_state(sigma_0, psi_target, layout, 0)

    println("--- Constructing |φ_1⟩ ---")
    sigma_1 = construct_phi_j(psi_target, layout, 1)
    phi_1   = purify_state(sigma_1, psi_target, layout, 1)

    println("--- Verifying Orthogonality ---")
    overlap = inner(phi_0, phi_1)

    println("Resulting Inner Product ⟨φ_0 | φ_1⟩ = ", overlap)

    if abs(overlap) < 1e-10
        println("SUCCESS: States are perfectly orthogonal.")
    else
        println("WARNING: States are not orthogonal!")
    end
end

"""
    debug_phi_j_construction(target_mps::MPS, layout::AbstractMarkerLayout, j::Int; verbose::Bool=true) -> MPS

Step-by-step diagnostic function for constructing |φ_j⟩.
Prints detailed reports on block decompositions, local density matrix ranks,
Lemma 5 intermediate merges, purification sink capacity, and final MPS properties.
"""
function debug_phi_j_construction(target_mps::MPS, layout::AbstractMarkerLayout, j::Int; verbose::Bool=true)
    println("==========================================================")
    println(" DEBUG: Constructing |φ_j⟩ for Shift j = $j")
    println(" Layout type : ", typeof(layout))
    println(" Block size l : ", layout.l)
    println(" Chain size N : ", layout.N)
    println(" Blocks n     : ", layout.n)
    println("==========================================================")

    # Disable ITensor order warnings for local debug inspection if needed
    ITensors.disable_warn_order()

    # Step 1: Shift Block Decomposition
    sb = build_blocks(layout, j)
    Aj_purif, Aj1_purif = purification_blocks(layout, sb)

    println("\n[Step 1] Block & Marker Decomposition:")
    println("  Marker set B_$j ($(length(sb.Bj)) sites) : ", sb.Bj)
    println("  Purification Block A_{$j,$j}          : ", Aj_purif)
    println("  Purification Block A_{$j,$(mod(j+1, layout.n))}          : ", Aj1_purif)
    println("  Total non-empty trimmed blocks A_{j,i} : ", length(sb.A))

    # Step 2: Extract Local Density Matrices (MPO format)
    println("\n[Step 2] Extracting Reduced Density Matrices ρ_{A_{j,i}}:")
    tau_blocks = MPO[]

    for (idx, block) in enumerate(sb.A)
        isempty(block) && continue

        # Convert 0-based site indices from layout to 1-based indices
        block_sites_1based = sort(block .+ 1)
        # start_site = block_sites_1based[1]
        # end_site = block_sites_1based[end]

        # mpo = get_block_rdm(target_mps, block_sites_1based)
        mpo = reduced_density_matrix(target_mps, block_sites_1based)
        print("   ✔")
        push!(tau_blocks, mpo)

        tr_val = real(mpo_trace(mpo))

        if length(mpo) <= 6
            dens = contract(mpo)
            u_inds = filter(hasplev(0), inds(dens))
            p_inds = filter(hasplev(1), inds(dens))
            c_row  = combiner(p_inds...)
            c_col  = combiner(u_inds...)
            mat    = matrix(dens * c_row * c_col, combinedind(c_row), combinedind(c_col))
            rk     = rank(mat, 1e-10)

            # D, _ = eigen(dens)
            # d = dim(D, 1)
            # evals = [D[n, n] for n in 1:d]
            # evals = sort(abs.(evals), rev=true)
            # println(evals[1:10])

            println("  A-Block $idx $(block_sites_1based) Length = $(length(mpo)), Rank = $rk, Tr(ρ) = $tr_val")
        else
            println("  A-Block $idx $(block_sites_1based): Length = $(length(mpo)), Tr(ρ) = $tr_val")
        end
    end

    # Step 3: Combine reduced density matrices using Lemma 5 from Dalzel and Brandão
    # to form the state σ_{n-2} on all A_j except A_{j,j} and A_{j,j+1}
    println("[Step 3] Combining blocks")
    sigma = tau_blocks[j+2]
    println("  Block $(j+2): ", sb.A[j+2])
    for p in 1:(layout.n-3)
        i = (j+p+1)%layout.n + 1
        i == j && i == j+1 && continue
        sigma = combine_states(sigma, tau_blocks[i])
        println("  ✔  Block $(i): ", sb.A[i])
    end
    println("  combined state: ", sigma)

    # Step 4: Purify combined states, handles the marker sites as well
    println("\n[Step 4] Purifying states over ρ_{A_{j,i}} into state |Q>_{A_j}")
    phi_j_mps = purify_state(sigma, target_mps, layout, j)

    # Step 5:
    println("\n[Step 5] Final MPS |φ_j⟩ Properties:")
    println("  Total MPS Sites        : ", length(phi_j_mps))
    println("  Max Link (Bond) Dim    : ", maximum(linkdims(phi_j_mps)))
    println("  State Norm ⟨φ_j|φ_j⟩^{1/2}: ", norm(phi_j_mps))
    println("==========================================================\n")

    ITensors.reset_warn_order()
    # return phi_j_mps
end

end # module TheoremVerification
