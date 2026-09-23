include("../lamps/MarkerGeometry.jl")
using .MarkerGeometry
using Plots
using Printf

# --- 1. N(l) helpers for each construction ---------------------------------
"""
    N_of_l(l::Int)
Closed-form chain length required for block size l under this module's
concrete choice of N (= largest marker index): N(l) = 6l³ − 7l² + l.
Strictly increasing in l, so it's invertible — this is what makes
`max_feasible_l` below possible without constructing a full MarkerLayout
for every candidate l.
Caveat: this is *a* valid sufficient chain length (this module's specific
choice), not necessarily the tightest N₀ the theorem allows — see
`max_feasible_l`'s docstring.
"""
N_of_l(l::Int) = 6*l^3 - 7*l^2 + l

"""
    N_of_l_dense(l::Int, k::Int; margin::Int=0)
Chain length required by `dense_marker_layout(l, k; margin=margin)`.
The marker matrix entries are R + (l*j+jp - 1)*spacing with
R = 2l²  + margin, spacing = k+1; the last (max) entry is
B[l, l-1] = R + (l² - l - 1)*(k+1), and N = B[l, l-1] + 1.
"""
function N_of_l_dense(l::Int, k::Int; margin::Int=0)
    R = 2*l^2 + margin
    return R + (l^2 - l - 1)*(k + 1)
end

"""
    N_of_l_shared(l::Int, k::Int; t::Int=max(k,2), R::Union{Int,Nothing}=nothing)
Chain length required by `SharedMarkerLayout(l, k, t; R=R)`.
Mirrors the N computation in the SharedMarkerLayout constructor:
    N = Rval + (M-1)*(k+1) + 2l
where M = 2 * length(biclique_cover_rows(l, t)) and Rval = 2l² by default.
"""
function N_of_l_shared(l::Int, k::Int; t::Int=max(k, 2), R::Union{Int,Nothing}=nothing)
    rows = biclique_cover_rows(l, t)
    M = 2 * length(rows)
    Rval = R === nothing ? 2*l^2 : R
    return Rval + (M - 1)*(k + 1)
end

# --- 2. Invert: largest l with N(l) ≤ P, per construction -------------------
"""
    max_feasible_l(P::Int; l_min::Int=2)
Given a resource budget P (the largest chain length you can afford to
simulate), returns the largest block size l such that N_of_l(l) ≤ P, or
`nothing` if even l_min doesn't fit.
Because N_of_l(l) reflects this module's own construction choice rather
than the theorem's tightest possible N₀ = O(l³), this is a *lower bound*
on the true best-achievable l for a given P — i.e. `epsilon_bound` below
is an upper bound on what the theorem's own error bound could ideally
deliver at budget P. Tightening this gap is a separate research question,
not something this function resolves.
"""
function max_feasible_l(P::Int; l_min::Int=2)
    N_of_l(l_min) > P && return nothing
    l = l_min
    while N_of_l(l + 1) <= P
        l += 1
    end
    return l
end
"""
    max_l_for_budget(Nfun, P::Int; l_min::Int=2)
Largest l ≥ l_min with Nfun(l) ≤ P, or `nothing` if even l_min doesn't fit.
`Nfun` is any strictly increasing function of l.
"""
function max_l_for_budget(Nfun, P::Int; l_min::Int=2)
    Nfun(l_min) > P && return nothing
    l = l_min
    while Nfun(l + 1) <= P
        l += 1
    end
    return l
end

# --- 3. ε bound as a function of P, per construction ------------------------

"""
    epsilon_paper(P::Int, k::Int)
Best (k,ε) error guaranteed by the paper's marker layout at budget P,
using ε = (k+3)/l. Returns `missing` if no l ≥ k+4 fits.
"""
function epsilon_paper(P::Int, k::Int)
    l = max_l_for_budget(N_of_l, P; l_min=k + 4)
    l === nothing && return missing
    return (k + 3) / l
end

"""
    epsilon_dense(P::Int, k::Int; margin::Int=0)
Same as `epsilon_paper` but for the dense-packing layout. Note N_dense
does depend on k (through the spacing k+1), unlike the paper's N(l).
"""
function epsilon_dense(P::Int, k::Int; margin::Int=0)
    Nfun = l -> N_of_l_dense(l, k; margin=margin)
    l = max_l_for_budget(Nfun, P; l_min=k + 4)
    l === nothing && return missing
    return (k + 3) / l
end

"""
    epsilon_shared(P::Int, k::Int; t::Int=max(k,2), R=nothing)
Best (k,ε) error guaranteed by the shared-marker layout at budget P.
"""
function epsilon_shared(P::Int, k::Int; t::Int=max(k, 2),
                        R::Union{Int,Nothing}=nothing)
    Nfun = l -> N_of_l_shared(l, k; t=t, R=R)
    l = max_l_for_budget(Nfun, P; l_min=k + 4)
    l === nothing && return missing
    return (k + 3) / l
end

# --- 4. The plot ------------------------------------------------------------

"""
    plot_budget_vs_epsilon(k::Int, P_values::AbstractVector{Int};
                           which=:all, t_shared::Int=max(k,2), kwargs...)
Log-log plot of best achievable ε vs resource budget P, for each of the
three constructions. Missing values (budget too small) are dropped from
each curve so no spurious line is drawn to zero.

Use `which` to plot a subset: `:paper`, `:dense`, `:shared`, or `:all`.
"""
function plot_budget_vs_epsilon(k::Int, P_values::AbstractVector{Int};
                                which::Symbol=:all, t_shared::Int=max(k, 2),
                                kwargs...)
    which in (:all, :paper, :dense, :shared) ||
        throw(ArgumentError("which must be :all, :paper, :dense, or :shared"))

    Ps = collect(P_values)
    out = Dict{Symbol,Vector{Float64}}()
    for (sym, f) in ((:paper, epsilon_paper),
                     (:dense, epsilon_dense),
                     (:shared, (P,k) -> epsilon_shared(P, k; t=t_shared)))
        vals = [f(P, k) for P in Ps]
        # Keep only finite entries so lines don't jump through missing.
        kept_P = [Ps[i] for i in eachindex(Ps) if !(vals[i] === missing)]
        kept_ε = [Float64(vals[i]) for i in eachindex(Ps) if !(vals[i] === missing)]
        out[sym] = vcat(kept_ε)   # will rebuild P inside plotting below
        out[Symbol(sym, :_P)] = vcat(kept_P)
    end

    plt = plot(; xscale=:log10, yscale=:log10,
               xlabel="P (sites simulated)",
               ylabel="best achievable ε (k=$k)",
               title="Local approximation vs resource budget",
               legend=:bottomleft, kwargs...)

    if which in (:all, :paper)
        plot!(plt, out[:paper_P], out[:paper]; lw=1.5,
              label="paper (one-per-pair, O(l³) = O(k³/ε³))")
    end
    if which in (:all, :dense)
        plot!(plt, out[:dense_P], out[:dense]; lw=1.5,
              label="dense packing, O(k·l²) = O(k³/ε²)")
    end
    if which in (:all, :shared)
        plot!(plt, out[:shared_P], out[:shared]; lw=1.5,
              label="shared markers (t=$t_shared), O(k²/ε² + (k/ε)log k)")
    end
    return plt
end

"""
    plot_budget_vs_epsilon_over_k(k_values::AbstractVector{Int},
                                  P_values::AbstractVector{Int};
                                  which::Symbol=:paper,
                                  t_for::Function = k -> max(k, 2),
                                  label_at::Float64=0.85,
                                  kwargs...)

For a fixed construction (`which` ∈ `:paper`, `:dense`, `:shared`), draw one
ε-vs-P curve per value of k, all on the same log-log axes. Each curve is
annotated with its k value directly next to the line (at the right-hand end),
and the legend is suppressed.

`t_for(k)` gives the shared-marker load t used for each k (default `max(k,2)`).
`label_at` is the fractional position along each curve (0 = start, 1 = end)
where the "k = ..." annotation is placed.
"""
function plot_budget_vs_epsilon_over_k(k_values::AbstractVector{Int},
                                       P_values::AbstractVector{Int};
                                       which::Symbol=:paper,
                                       t_for::Function = k -> max(k, 2),
                                       label_at::Float64=0.85,
                                       kwargs...)
    which in (:paper, :dense, :shared) ||
        throw(ArgumentError("which must be :paper, :dense, or :shared"))

    Ps = collect(P_values)

    plt = plot(; xscale=:log10, yscale=:log10,
               xlabel="P (sites simulated)",
               ylabel="best achievable ε",
               title="Local approximation vs resource budget — $(which) construction",
               legend=false, kwargs...)

    for k in k_values
        # Pick the right ε function for this construction.
        f = if which === :paper
            P -> epsilon_paper(P, k)
        elseif which === :dense
            P -> epsilon_dense(P, k)
        else
            let t = t_for(k)
                P -> epsilon_shared(P, k; t=t)
            end
        end

        vals = [f(P) for P in Ps]
        kept_P = [Ps[i] for i in eachindex(Ps) if !(vals[i] === missing)]
        kept_ε = [Float64(vals[i]) for i in eachindex(Ps) if !(vals[i] === missing)]
        isempty(kept_P) && continue

        plot!(plt, kept_P, kept_ε; lw=1.5)

        # Place the k label at a fractional position along the curve.
        idx = clamp(round(Int, label_at * length(kept_P)), 1, length(kept_P))
        annotate!(plt, kept_P[idx], kept_ε[idx],text("k=$k", 9, :left, :bottom))
    end

    return plt
end

# --- 5. Quick numeric table for sanity-checking -----------------------------

"""
    compare_epsilon_table(k::Int, P_values::AbstractVector{Int}; t_shared=max(k,2))
Print ε at each budget for all three constructions side by side.
"""
function compare_epsilon_table(k::Int, P_values::AbstractVector{Int}; t_shared::Int=max(k, 2))
    println("Comparison table of achievable ε given a budget P for k=$k ")
    @printf("%12s %12s %12s %12s\n", "P", "ε_paper", "ε_dense", "ε_shared")
    for P in P_values
        ep = epsilon_paper(P, k)
        ed = epsilon_dense(P, k)
        es = epsilon_shared(P, k; t=t_shared)
        f(x) = x === missing ? "-" : @sprintf("%.4f", x)
        @printf("%12d %12s %12s %12s\n", P, f(ep), f(ed), f(es))
    end
end

# --- 6. Demo ---------------------------------------------------------------

function run_epsilon_plot_single(; k::Int=4, Pmin::Int=10^2, Pmax::Int=10^7, npoints::Int=200, t_shared::Int=max(k, 2), which::Symbol=:all)
    P_values = round.(Int, exp10.(range(log10(Pmin), log10(Pmax), length=npoints)))
    plt = plot_budget_vs_epsilon(k, P_values; which=which, t_shared=t_shared, title="k=$k, shared load t=$t_shared")
    savefig(plt, "figures/epsilon-vs-budget/$(which)_k=$k.png")
    println("Plot saved to figures/epsilon-vs-budget/$(which)_k=$k.png")
    compare_epsilon_table(k, [10^2, 500, 10^3, 10^4, 10^5, 10^6, 10^7]; t_shared=t_shared)
end

function run_epsilon_plot_over_k(; k::AbstractVector{Int}=4, Pmin::Int=10^2, Pmax::Int=10^7, npoints::Int=200, t_for::Function = k -> max(k, 2), which::Symbol=:paper)
    P_values = round.(Int, exp10.(range(log10(Pmin), log10(Pmax), length=npoints)))
    plt = plot_budget_vs_epsilon_over_k(k, P_values; which=which, t_for=t_for, title="Achievable ε vs budget P for $(which) construction")
    savefig(plt, "figures/epsilon-vs-budget/$(which)_over_k.png")
    println("Plot saved to figures/epsilon-vs-budget/$(which)_over_k.png")
    # compare_epsilon_table(k, [10^2, 500, 10^3, 10^4, 10^5, 10^6, 10^7]; t_shared=t_shared)
end
