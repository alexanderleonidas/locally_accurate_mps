using ITensors
using ITensorMPS
using LinearAlgebra
using Plots

# Load framework and core modules
include("../LocallyAccurateMPS.jl")
using .LocallyAccurateMPS
using .LocallyAccurateMPS.MatrixProductStates
using .LocallyAccurateMPS.MatrixProductOperators
using .LocallyAccurateMPS.DensityMatrix
using .LocallyAccurateMPS.Measures
using .LocallyAccurateMPS.AKLT
using .LocallyAccurateMPS.TFIM
using .LocallyAccurateMPS.XXZ


"""
    benchmark_bond_dimension_vs_N(; N_list=[16, 32, 64, 128, 256], k=2, target_eps=0.01)

Benchmarks the minimal MPS bond dimension χ required to approximate the local k-body
reduced states of a high-precision ground state within a local trace distance error target_eps.
"""
function benchmark_bond_dimension_vs_N(;
    N_list::Vector{Int} = [16, 32, 64, 128, 256],
    k::Int = 2,
    target_eps::Float64 = 0.01,
    chi_grid::Vector{Int} = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
)
    println("==========================================================================")
    println(" Benchmarking Bond Dimension Saturation χ*(N) vs. System Size N")
    println(" Parameters: k = $k, Target Trace Error ε = $target_eps")
    println("==========================================================================")

    results = Dict{Int, Int}() # N => required_chi

    for N in N_list
        sites = siteinds("S=1/2", N)
        # 1. Build Transverse-Field Ising Model (TFIM) Hamiltonian in gapped phase (h = 1.5)
        energy, psi_exact = tfim_ground_state(sites; h=1.5, nsweeps=15, maxdim=200)

        # 2. Find minimal bond dimension χ that satisfies local trace error threshold
        found = false
        for chi in chi_grid
            # Truncate MPS natively using MatrixProductStates module wrapper
            psi_trunc = truncate_bond_dimension(psi_exact, chi)

            # Evaluate maximum k-local trace distance error across the chain
            # d1 = evaluate_max_local_trace_distance(psi_exact, psi_trunc, k)
            d1 = k_local_trace_distance(psi_exact, psi_trunc, k)

            if d1 <= target_eps
                results[N] = chi
                println("N = $(lpad(N, 3)) | Local Error Saturation: χ* = $(lpad(chi, 2)), Max Trace Distance d1 =$(round(d1, digits=6))")
                found = true
                break
            end
        end

        if !found
            println("N = $(lpad(N, 3)) | Error target ε = $target_eps not met within maximum test bond dimension$(maximum(chi_grid))")
        end
    end

    println("==========================================================================")
    return results
end

"""
    plot_bond_dimension_saturation(results_dict; k=2, target_eps=0.01, filename="figures/chi_saturation_vs_N.png")

Plots the minimal required bond dimension χ* vs. system length N.

# Arguments
- `results_dict`: Dictionary or Vector of Pairs mapping `N => required_chi`
  (e.g., Dict(16 => 4, 32 => 6, 64 => 6, 128 => 6, 256 => 6))
- `k`: Observation window length (default: 2)
- `target_eps`: Local trace distance threshold (default: 0.01)
- `filename`: Path to save the output plot
"""
function plot_bond_dimension_saturation(
    results_dict::Union{Dict{Int, Int}, Vector{Pair{Int, Int}}};
    k::Int = 2,
    target_eps::Float64 = 0.01,
    filename::String = "chi_saturation_vs_N.png"
)
    # Sort data by chain length N
    sorted_pairs = sort(collect(results_dict), by = x -> x[1])
    N_vals   = [p[1] for p in sorted_pairs]
    chi_vals = [p[2] for p in sorted_pairs]

    # Initialize plot canvas
    plt = plot(
        xlabel = "Chain Length N",
        ylabel = "Required Bond Dimension χ*",
        title  = "Bond Dimension Saturation (k = $k, ε =$target_eps)",
        legend = false,
        framestyle = :box,
        grid = true,
        gridalpha = 0.3,
        tickfont = font(10),
        guidefont = font(11, :bold),
        titlefont = font(12, :bold),
        size = (600, 400),
        dpi = 300
    )

    # Plot saturation curve
    plot!(
        plt,
        N_vals,
        chi_vals,
        mark = :circle,
        markersize = 6,
        markercolor = :crimson,
        linecolor = :crimson,
        linewidth = 2,
        label = "χ*(N)"
    )

    # Add a horizontal dashed reference line showing the asymptotic saturation level
    max_chi = chi_vals[end]
    hline!(
        plt,
        [max_chi],
        linestyle = :dash,
        linecolor = :black,
        alpha = 0.6,
        linewidth = 1.5
    )

    # Annotate saturation region
    annotate!(
        plt,
        N_vals[end - 1],
        max_chi + 0.8,
        text("Saturation χ* ≈ $max_chi", 9, :right, :italic)
    )

    # Ensure output directory exists if path is provided
    # outdir = dirname("figures/")
    # if !isempty(outdir) && !isdir(outdir)
    #     mkpath(outdir)
    # end

    savefig(plt, "figures/$filename")
    println("✔ Saturation plot saved successfully to figures/$filename")

    return plt
end

"""
    plot_multi_eps_saturation(multi_results; k=2, filename="figures/multi_chi_saturation.png")

Plots χ* vs N for multiple target trace distance thresholds ε.

# Arguments
- `multi_results`: Dict mapping `eps_target => Dict(N => chi)`
"""
function plot_multi_eps_saturation(
    multi_results::Dict{Float64, Dict{Int, Int}};
    k::Int = 2,
    filename::String = "multi_chi_saturation.png"
)
    plt = plot(
        xlabel = "Chain Length N",
        ylabel = "Required Bond Dimension χ*",
        title  = "k-Local Error Saturation vs. N (k = $k)",
        legend = :bottomright,
        framestyle = :box,
        grid = true,
        gridalpha = 0.3,
        size = (650, 420),
        dpi = 300
    )

    colors = [:navy, :crimson, :darkgreen, :darkorange]

    for (idx, (eps, res_dict)) in enumerate(sort(collect(multi_results), by = x -> x[1], rev=true))
        sorted_pairs = sort(collect(res_dict), by = x -> x[1])
        N_vals   = [p[1] for p in sorted_pairs]
        chi_vals = [p[2] for p in sorted_pairs]

        c = colors[mod1(idx, length(colors))]

        plot!(
            plt,
            N_vals,
            chi_vals,
            mark = :square,
            markersize = 5,
            linecolor = c,
            markercolor = c,
            linewidth = 2,
            label = "ε = $eps"
        )
    end

    savefig(plt, filename)
    println("✔ Multi-curve plot saved to $filename")
    return plt
end

function plot_bond_dimension_vs_N()

    # 1. Run single benchmark
    results = benchmark_bond_dimension_vs_N(
        N_list = [16, 32, 64, 128, 256],
        k = 2,
        target_eps = 0.01
    )

    # 2. Plot result
    plot_bond_dimension_saturation(results; k = 2, target_eps = 0.01)

    # --- Or for multiple target thresholds ---

    # multi_results = Dict{Float64, Dict{Int, Int}}()
    # for eps in [0.05, 0.01, 0.005]
    #     multi_results[eps] = benchmark_bond_dimension_vs_N(
    #         N_list = [16, 32, 64, 128, 256],
    #         k = 2,
    #         target_eps = eps
    #     )
    # end
    #
    # plot_multi_eps_saturation(multi_results; k = 2)

end
