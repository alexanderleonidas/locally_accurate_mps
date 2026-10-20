using Plots

"""
    benchmark_chain_lengths()
Benchmark and plot the scaling of chain lengths N₀ for different methods (Original, Dense, Shared Reuse)
as a function of k/ε for k={2, 4, and 8}. The results are saved to "chain_length_scaling.png".
"""
function benchmark_chain_lengths()
    epsilons = 10 .^ range(log10(0.5), log10(0.01), length=50)
    k_vals = [2, 4, 8]

    p1 = plot(title="Chain Length N₀ Scaling (k=4)", xlabel="1/ε", ylabel="N₀",
              xscale=:log10, yscale=:log10, legend=:topleft, framestyle=:box)

    for k in k_vals
        N0_paper  = [MarkerLayout(k, eps).N for eps in epsilons]
        N0_dense  = [MarkerLayout(k, eps; which=:dense).N for eps in epsilons]
        N0_share = [N0_shared(k, eps).N0 for eps in epsilons]

        if k == 4
            plot!(p1, 1 ./ epsilons, N0_paper,  label="Original O(k³/ε³)", lw=2, linestyle=:dash)
            plot!(p1, 1 ./ epsilons, N0_dense,  label="Dense O(k³/ε²)", lw=2, linestyle=:dot)
            plot!(p1, 1 ./ epsilons, N0_share, label="Shared Reuse O(k²/ε² + (k/ε)log k)", lw=2.5)
        end
    end
    savefig(p1, "figures/chain-length/chain_length_scaling.png")
    println("Saved chain_length_scaling.png")
end


"""
    benchmark_chain_lengths_per_method(;
        k_vals = [2, 4, 8],
        ε_min = 0.5, ε_max = 0.01, npoints = 50,
        inline_labels = false,
        outdir = "figures/chain-length")

Plot three separate figures, one for each construction (Original/paper,
Dense, Shared Reuse). Each figure shows the N₀ scaling as a function of
1/ε with one curve per k in `k_vals`.

If `inline_labels = true`, each curve is annotated with its k value next
to the line and the legend is suppressed (matching the earlier
`plot_budget_vs_epsilon_over_k` style). Otherwise a normal legend is used.

Returns `(p_paper, p_dense, p_shared)` and saves each figure to
`outdir/chain_length_<method>.png`.
"""
function benchmark_chain_lengths_per_method(;
        k_vals::AbstractVector{Int} = [2, 4, 8],
        ε_min::Float64 = 0.5, ε_max::Float64 = 0.01,
        npoints::Int = 50,
        inline_labels::Bool = false,
        outdir::String = "figures/chain-length")

    ε_values = 10 .^ range(log10(ε_min), log10(ε_max), length = npoints)
    inv_eps  = 1 ./ ε_values

    # --- helper to reduce repetition -------------------------------------
    function _add_curve!(plt, xs, ys, k, color_idx)
        if inline_labels
            plot!(plt, xs, ys; lw = 2, color = color_idx)
            # annotate near the right-hand end of each curve
            idx = max(1, round(Int, 0.85 * length(xs)))
            annotate!(plt, xs[idx], ys[idx], text("k=$k", 9, :left, :bottom))
        else
            plot!(plt, xs, ys; lw = 2, label = "k=$k", color = color_idx)
        end
    end

    # --- Original / paper -------------------------------------------------
    p_paper = plot(title = "Original (Paper) Construction",
                   xlabel = "1/ε", ylabel = "N₀",
                   xscale = :log10, yscale = :log10,
                   legend = inline_labels ? false : :topleft,
                   framestyle = :box)
    for (i, k) in enumerate(k_vals)
        N0 = [MarkerLayout(k, ε; which=:paper).N for ε in ε_values]
        _add_curve!(p_paper, inv_eps, N0, k, i)
    end
    savefig(p_paper, joinpath(outdir, "d&b.png"))

    # --- Dense ------------------------------------------------------------
    p_dense = plot(title = "Dense Construction",
                   xlabel = "1/ε", ylabel = "N₀",
                   xscale = :log10, yscale = :log10,
                   legend = inline_labels ? false : :topleft,
                   framestyle = :box)
    for (i, k) in enumerate(k_vals)
        N0 = [MarkerLayout(k, ε; which=:dense).N for ε in ε_values]
        _add_curve!(p_dense, inv_eps, N0, k, i)
    end
    savefig(p_dense, joinpath(outdir, "dense.png"))

    # --- Shared Reuse -----------------------------------------------------
    p_shared = plot(title = "Shared Reuse Construction",
                    xlabel = "1/ε", ylabel = "N₀",
                    xscale = :log10, yscale = :log10,
                    legend = inline_labels ? false : :topleft,
                    framestyle = :box)
    for (i, k) in enumerate(k_vals)
        N0 = [N0_shared(k, ε).N0 for ε in ε_values]
        _add_curve!(p_shared, inv_eps, N0, k, i)
    end
    savefig(p_shared, joinpath(outdir, "shared.png"))

    println("Saved d&b.png, dense.png, shared.png to $outdir")
    return (p_paper, p_dense, p_shared)
end
