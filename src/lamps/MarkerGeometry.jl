module MarkerGeometry
using Printf: Floats
using Base: rows_to_dimshape
using Printf, Plots
export AbstractMarkerLayout,
       MarkerLayout,
       ShiftBlocks,
       build_blocks,
       purification_blocks,
       verify_all_purifications,
       shared_marker_report,
       print_markers,
       dense_marker_layout,
       run_demo,
       N_of_l,
       max_feasible_l,
       epsilon_bound,
       p_vs_epsilon_curve,
       Marker,
       SharedMarkerLayout,
       biclique_cover_rows,
       verify_separating_rows,
       print_shared_markers,
       N0_shared,
       compare_all_constructions
# ---------------------------------------------------------------------------
# AbstractMarkerLayout: common interface shared by every marker construction
# in this file (the paper's original one-marker-per-pair layout, and the
# shared/reused-marker layout added below). Any concrete subtype only needs
# to provide `l`.
# ---------------------------------------------------------------------------
abstract type AbstractMarkerLayout end

# ---------------------------------------------------------------------------
# MarkerLayout: the l-dependent data shared across all shifts j = 0,...,l-1
# ---------------------------------------------------------------------------
"""
    MarkerLayout

Holds Theorem 1 Part 1's marker geometry for block size `l`:
- `k` : length of the local window (1 ≤ k < l)
- `ε` : desired trace distance (0 ≤ ε < 1)
- `l` : block length (choose l = ceil((k+3)/ε) per Eq. (37) of Dalzell & Brandão)
- `N` : chain length, set just past the largest marker index
- `n` : number of length-l blocks, ⌊N/l⌋
- `B` : l×l matrix; B[j+1, j'+1] is the site index of marker B_{j,j'}
        (site index = 3l²(j+j') + l(j−j') + 2l², Eq. before (32)), or -1
        on the diagonal where j = j' (no marker is defined there).
- `which` : :paper for the original one-marker-per-pair layout, or :dense
           for the dense-packing construction (see `dense_marker_layout`).

Note this struct only tracks *where* the O(l²−l) single-site markers and
the block boundaries sit — it is a combinatorial/geometric model of the
construction, not a simulation of |φ_j⟩ itself. Building the actual
quantum states (Lemma 5's iterative purification) requires manipulating
d^(2l)-dimensional reduced density matrices and belongs in the MPS module,
not here — see the note at the bottom of this file.
This is the "one dedicated marker per ordered pair" layout: l²−l markers
total, each used by exactly one pair (j,j'). See `SharedMarkerLayout`
below for the marker-reuse construction, where a single marker is shared
by many shifts at once.
"""
struct MarkerLayout <: AbstractMarkerLayout
    k::Integer
    ε::AbstractFloat
    l::Integer
    N::Integer
    n::Integer
    B::Matrix{Integer}
    which::Symbol
end

"""
    MarkerLayout(k::Integer, ε::AbstractFloat; which::Symbol=:paper)

Constructs a `MarkerLayout` for block size l = ceil((k+3)/ε) and the
"""
function MarkerLayout(k::Integer, ε::AbstractFloat; which::Symbol=:paper)
    @assert 0 <= ε < 1 "ε must satisfy 0 ≤ ε < 1"
    l = ceil(Int, (k + 3) / ε)
    # @assert l > 2 "l must be at least 2 (need j ≠ j' pairs)"
    # @assert 1 <= k < l "k must satisfy 1 ≤ k < l (requires a window smaller than block size)"
    B = which===:paper ? marker_matrix(l) : dense_marker_matrix(l, k, spacing=k+1, margin=0)
    # The largest marker index is always at (j, j') = (l-1, l-2): the 3l²(j+j')
    # term dominates so j+j' is maximized first (=2l-3, only achieved by
    # (l-1,l-2) and (l-2,l-1)), and between those two the l(j-j') term breaks
    # the tie in favor of j > j'.
    # N = B[l, l-1] + 2l + 1
    N = B[l, l-1]
    n = fld(N, l)
    return MarkerLayout(k, ε, l ,N, n, B, which)
end

"""
    marker_matrix(l::Int)

Computes the l×l marker matrix B for the original paper's construction (one dedicated marker per ordered pair).
The marker for the ordered pair (j, j') is placed at site index 3l²(j+j') + l(j−j') + 2l², and the diagonal entries
(where j = j') are set to -1 since no marker is defined there. This function returns the matrix B.
"""
function marker_matrix(l::Integer)
    B = Matrix{Int}(undef, l, l)
    for j in 0:l-1, jp in 0:l-1
        # The index of the B_{j,j'} marker is given by 3l²(j+ j′) + l(j−j′) + 2l²
        # we get the maximum value when j = l-1 and j' = l-2, which gives us 3l²(2l-3) + l(1) + 2l² = 6l³ - 7l² + l
        B[j+1, jp+1] = j == jp ? -1 : 3*l^2*(j + jp) + l*(j - jp) + 2*l^2
    end
    return B
end

"""
    dense_marker_matrix(l::Int, k::Int; spacing::Int=k+1, margin::Int=2l)

Alternative to `MarkerLayout(l)`. Motivation:
Eq. (35)-(37) only require that no length-k window contains two markers,
i.e. pairwise marker separation ≥ k (k+1 gives one site of margin) — NOT
separation ≥ l as the paper's formula happens to realize (its actual
minimum gap is 2l, checked directly on the l=3 case). Meanwhile every
shift's purification blocks M_{j,j}, M_{j,j+1} live within the first
~2l² sites regardless of where markers sit. So: reserve R ≈ 2l²+margin
sites to clear every shift's purification zone, then pack all l²−l
markers as a single arithmetic sequence with gap `spacing` right after.
This gives N ≈ R + (l²−l)(spacing) = O(k·l²) = O(k³/ε²) using l=(k+3)/ε,
versus the paper's O(l³) = O(k³/ε³) — a genuine asymptotic saving of one
factor of 1/ε.
"""
function dense_marker_matrix(l::Integer, k::Integer; spacing::Integer=k+1, margin::Integer=2l)
    @assert spacing > k  "spacing=$spacing <= k=$k: two markers could then share a k-site window, breaking the disjointness argument"
    B = Matrix{Int}(undef, l, l)
    # rank = 0
    for j in 0:l-1, jp in 0:l-1
        if j == jp
            B[j+1, jp+1] = -1
        else
            # B[j+1, jp+1] = R + 1 + rank * spacing
            # rank += 1
            B[j+1, jp+1] = 2*l^2 + (l*j+jp -1) * spacing + margin
        end
    end
    # N = R + (rank - 1) * spacing + 1
    # N = B[l, l-1]
    return B
end
"""
    compare_packings(l::Int, k::Int)

Prints N required by the paper's formula vs. the dense-packing candidate,
for the same (l, k). Does not itself validate the dense packing's
correctness — run `verify_all_purifications` on both layouts separately.
"""
function compare_packings(l::Integer, k::Integer)
    paper = MarkerLayout(l)
    dense = dense_marker_layout(l, k, margin=0)
    println("l=$l, k=$k:")
    println("  paper's N₀  = $(paper.N)   (formula-derived spacing >= l)")
    println("  dense N₀    = $(dense.N)   (spacing = k+1 = $(k+1))")
    println("  ratio       = $(round(paper.N / dense.N, digits=1))×")
end

"""
    print_markers(layout::MarkerLayout)

Displays the marker matrix B in a human-readable format, showing the site indices of each marker B_{j,j'} for j ≠ j'.
The diagonal entries (where j = j') are marked with a dash (-) since no marker is defined there.
It also prints the total number of distinct markers and the chain length N along with the number of blocks n.
"""
function print_markers(layout::MarkerLayout)
    l, B = layout.l, layout.B
    println("=== Single-site markers B_{j,j'} (l = $l) ===")
        # Determine a suitable column width (at least 5, and one extra space for positive numbers)
        max_num = maximum(B)                 # ignore -1; it's fine
        width = max(5, length(string(max_num)) + 1)
        # Build header: "j'= X" with X right‑aligned to make total length = width
        header_cols = [string("j'=", lpad(c, width - 3)) for c in 0:l-1]
        println("      " * join(header_cols, " "))
        # Print each row
        for j in 0:l-1
            row = [B[j+1, jp+1] for jp in 0:l-1]
            cells = [r == -1 ? lpad("-", width) : @sprintf("%*d", width, r) for r in row]
            println("j=$j   " * join(cells, " "))
        end
        num_markers = l^2 - l
        println("\nDistinct markers: $num_markers  (l²−l, one per ordered pair j≠j')")
        println("Chain length N = $(layout.N), blocks n = ⌊N/l⌋ = $(layout.n)")
end
# ---------------------------------------------------------------------------
# ShiftBlocks: the per-shift decomposition used to build |φ_j⟩
# ---------------------------------------------------------------------------
"""
    ShiftBlocks
For one fixed shift j:
- `M[i]` for i = 1,...,n-1 is block M_{j,i} = [j+l(i-1), j+li-1]
- `M[n]` is the wrap-around block M_{j,0} = [j+ l(n−1),N−1] and [0,j−1]
- `A[i]` is the trimmed block A_{j,i} = M_{j,i} \\ B_j
- `Bj`   is the marker set B_j = ∪_{j'≠j} {B_{j,j'}, B_{j',j}}
"""
struct ShiftBlocks
    j::Integer
    M::Vector{Vector{Integer}}
    A::Vector{Vector{Integer}}
    Bj::Vector{Integer}
end

"""
    marker_sites_for_shift(layout::AbstractMarkerLayout, j::Int)

Returns the physical sites of every marker that constrains shift j
(the set B_j, Eq. (21)/(38)). `build_blocks` is generic over this one
method, so it works unchanged for `MarkerLayout` (one marker per pair)
and `SharedMarkerLayout` (one marker shared by many shifts) alike — add
a method here for any future marker construction instead of touching
`build_blocks` itself.
"""
function marker_sites_for_shift(layout::MarkerLayout, j::Integer)
    l, B = layout.l, layout.B
    Bj = Int[]
    for jp in 0:l-1
        jp == j && continue
        push!(Bj, B[j+1, jp+1])
        push!(Bj, B[jp+1, j+1])
    end
    return Bj
end

"""
    build_blocks(layout, j)

Returns the per-shift decomposition used to build |φ_j⟩: the blocks M_{j,i} and the trimmed blocks
A_{j,i} = M_{j,i}!=B_j, along with the marker set B_j.
"""
function build_blocks(layout::AbstractMarkerLayout, j::Integer)
    l, N, n = layout.l, layout.N, layout.n
    0 <= j < l || throw(ArgumentError("shift j must satisfy 0 ≤ j < l"))
    M = Vector{Vector{Integer}}()
    for i in 1:n-1
        push!(M, collect(j + l*(i-1) : j + l*i - 1))
        # println("Shift j=$j: blocks M_{$j,$i} = $(M[i])")
    end
    start_wrap = j + l*(n - 1)
    wrap = vcat(collect(start_wrap:N-1), j > 0 ? collect(0:j-1) : Int[])
    push!(M, wrap)   # position n <-> block index i = 0
    Bj = marker_sites_for_shift(layout, j)
    A = [filter(site -> !(site in Bj), block) for block in M]
    # println("B_j = $Bj")
    return ShiftBlocks(j, M, A, Bj)
end

"""
    purification_blocks(layout, sb)

Returns (A_{j,j}, A_{j,j+1}) — mod-n conversion to the 1-based `M`/`A`
vectors, where position n stores the wrap-around block M_{j,0}/A_{j,0}.
These are the two blocks used to purify σ_{n-2} into |Q_j⟩ (see the
paragraph following Eq. (32) in the paper).
"""
function purification_blocks(layout::AbstractMarkerLayout, sb::ShiftBlocks)
    n = layout.n
    to_pos(i) = i == 0 ? n : i     # block index i=0 lives at vector position n
    pos0 = to_pos(mod(sb.j, n))
    pos1 = to_pos(mod(sb.j + 1, n))
    return sb.M[pos0], sb.M[pos1]
end
"""
    verify_all_purifications(layout)

Sanity check from the paper's construction: since the markers in B_j are
excluded from A_{j,i} by definition, no marker of B_j should ever land in
the purification blocks M_{j,j} or M_{j,j+1}. This is guaranteed by
marker formula having a margin of 2l^2, so this check should always pass — it
verifies the *implementation*, not the theorem.
"""
function verify_all_purifications(layout::AbstractMarkerLayout)
    println("=== Verifying purification blocks A_{j,j}, A_{j,j+1} avoid B_j ===")
    all_ok = true
    for j in 0:layout.l-1
        sb = build_blocks(layout, j)
        Aj, Aj1 = purification_blocks(layout, sb)
        bad = union(intersect(Aj, sb.Bj), intersect(Aj1, sb.Bj))
        ok = isempty(bad)
        all_ok &= ok
        status = ok ? "OK" : "FAIL: markers $bad leaked into purification blocks"
        println("j=$j: A_{j,j}=[$(Aj[1]),...,$(Aj[end])], A_{j,j+1}=[$(Aj1[1]),...,$(Aj1[end])] => $status")
    end
    println(all_ok ? "\nAll shifts verified." : "\nWARNING: at least one shift failed.")
    return all_ok
end
"""
    shared_marker_report(layout)

For each unordered pair (j, j'), the site B_{j,j'} sits inside *both*
B_j and B_j' by construction (that's what makes ⟨φ_j|φ_j'⟩ = 0 work:
the same site is forced to |0⟩ for φ_j and |1⟩ for φ_j'). This isn't a
discovered inefficiency — it's structurally required by the proof, since
each ordered-pair marker is definitionally shared between exactly two
shifts' marker sets.
"""
function shared_marker_report(layout::AbstractMarkerLayout)
    l = layout.l
    Bj_sets = [Set(build_blocks(layout, j).Bj) for j in 0:l-1]
    println("=== Marker sharing between shifts (structural, by construction) ===")
    for j in 0:l-1, jp in j+1:l-1
        shared = sort(collect(intersect(Bj_sets[j+1], Bj_sets[jp+1])))
        println("B_$j ∩ B_$jp = $shared (shared $(length(shared)) markers)")
    end
end
# ===========================================================================
# SharedMarkerLayout: the marker-reuse construction (bounded biclique cover)
#
# This is the analytical track's second improvement (after `dense_marker_
# layout`'s "free win"). Instead of one dedicated marker per ordered pair
# (l²−l markers), a single marker here is a *shared* bipartition of a subset
# of shifts: it forces every shift in S0 to |0⟩ and every shift in S1 to |1⟩
# at once. A carefully chosen family of O(l²/t² + (l/t)log t) such markers
# (a two-fold bounded biclique cover of K_l, load ≤ t) still separates every
# pair of shifts — with far fewer markers than one-per-pair. Every marker
# site still needs to be seen by only one length-k window at a time, so
# markers are spaced by k+1 and duplicated onto two physical copies to keep
# two independent separating witnesses per pair (one can always fall inside
# a window X, the other cannot).
#
# Net effect: with load t = Θ(k), this cuts the chain-length threshold from
# N_0 = O(k³/ε²) (the "free win") down to N_0 = O(k²/ε²) — see `N0_shared`.
# ===========================================================================
"""
    Marker

One physical marker: a single chain site that forces every shift in `S0`
to |0⟩ and every shift in `S1` to |1⟩ (Eq. (1)). Shifts not listed in
either set are unconstrained at this site (the `*` symbol) and stay part
of the ordinary A_j system. `site` is the physical index on the chain.
"""
struct Marker
    S0::Vector{Integer}
    S1::Vector{Integer}
    site::Int
end
"""
    biclique_cover_rows(l::Integer, t::Integer)

Builds a one-fold bounded biclique cover of K_l: a list of (S0, S1)
shift-bipartitions ("rows"), each of load |S0|+|S1| ≤ t, such that every
unordered pair of shifts in [l] is separated (put on opposite sides) by
at least one row. This is the deterministic grouping construction
(groups + inter-group markers + intra-group binary addressing),
generalized to any integer l > t ≥ 2 via s = ⌊t/2⌋ (not only even t):
1. Split [l] into groups of size ≤ 2s.
2. Split each group into two halves of size ≤ s.
3. For every pair of groups (a, b), add the 4 rows pairing each half of
   group a against each half of group b (load ≤ 2s ≤ t). This separates
   every cross-group pair.
4. Inside each group, give each member a unique binary address of
   ⌈log2(group size)⌉ bits, and add one row per bit position (the
   members with bit=0 vs bit=1). This separates every within-group pair,
   since any two distinct addresses differ in some bit.
The result is only a *one*-fold cover (each pair separated by ≥1 row,
not ≥2): `SharedMarkerLayout` gets multiplicity two afterward by
duplicating every row onto two distinct physical sites, not by finding
extra combinatorial rows here — see its docstring.
Caveat (odd t): the equal-half split caps every marker's load at
2s = t−1 when t is odd (one unit of budget unused), and each inter-group
marker then covers ⌊t/2⌋² pairs instead of the achievable ⌊t²/4⌋ — e.g.
for t=5, 4 pairs instead of 6, a 50% loss for that one marker. This is a
genuine, acknowledged constant-factor slack (at most 4×, shrinking to 1×
as t→∞); it changes the constant in P = Θ(l²/t² + (l/t)log t), not the
asymptotic order. Gu (arXiv:1511.02845) gives the fully tight
unequal-split construction; this function deliberately keeps the simpler
equal-split version, per the same design choice as the rest of this
module (cite Gu for tightness rather than re-derive it).
"""
function biclique_cover_rows(l::Integer, t::Integer)
    @assert l > 2 "l must be at least 2"
    @assert 2 <= t < l "t must be at least 2 and less than l"
    s = fld(t, 2)                      # ⌊t/2⌋: per-half budget, so 2s ≤ t
    group_size = 2*s
    g = cld(l, group_size)             # number of groups
    groups = Vector{Vector{Int}}(undef, g)
    for a in 1:g
        lo = (a-1)*group_size
        hi = min(a*group_size, l) - 1
        groups[a] = collect(lo:hi)
    end
    # println("groups (size ≤ $group_size):")
    # for a in 1:g
    #     println("  group $a: $(groups[a])")
    # end
    rows = Tuple{Vector{Integer},Vector{Int}}[]
    # --- inter-group rows: split each group into two halves, 4 rows per group pair ---
    half0 = Vector{Vector{Int}}(undef, g)
    half1 = Vector{Vector{Int}}(undef, g)
    for a in 1:g
        m = length(groups[a])
        mid = fld(m, 2)
        half0[a] = groups[a][1:mid]
        half1[a] = groups[a][mid+1:end]
    end
    # println("half-groups (size ≤ $s):")
    # for a in 1:g
    #     println("  group $a: half0=$(half0[a]), half1=$(half1[a])")
    # end
    for a in 1:g, b in a+1:g
        # Handle empty set when dealing with groups of size one
        isempty(half0[a]) || isempty(half0[b]) || push!(rows, (half0[a], half0[b]))
        isempty(half0[a]) || isempty(half1[b]) || push!(rows, (half0[a], half1[b]))
        isempty(half1[a]) || isempty(half0[b]) || push!(rows, (half1[a], half0[b]))
        isempty(half1[a]) || isempty(half1[b]) || push!(rows, (half1[a], half1[b]))
    end
    # println("inter-group rows (4 per group pair, $(g*(g-1)/2) pairs)")
    # println(rows)
    # --- intra-group rows: one row per bit of a binary address within the group ---
    for grp in groups
        m = length(grp)
        m <= 1 && continue
        h = ceil(Int, log2(m))
        for bit in 0:h-1
            S0 = [grp[i+1] for i in 0:m-1 if ((i >> bit) & 1) == 0]
            S1 = [grp[i+1] for i in 0:m-1 if ((i >> bit) & 1) == 1]
            push!(rows, (S0, S1))
        end
    end
    # println("intra-group rows (1 per bit of each group, total $(sum(ceil(Int, log2(length(grp))) for grp in groups)) rows)")
    # println(rows)
    return rows
end
"""
    verify_separating_rows(rows, l::Int)

Checks that every unordered pair {x,y} ⊂ [l] is separated by
at least one row of `rows`. Returns `(ok::Bool, unseparated_pairs)`. Run
this on the output of `biclique_cover_rows` before trusting it — it is
the empirical check that the construction, not just its asymptotics, is
correct for the given (l, t).
"""
function verify_separating_rows(rows, l::Integer)
    mult = zeros(Int, l, l)
    for (S0, S1) in rows
        for x in S0, y in S1
            mult[x+1, y+1] += 1
            mult[y+1, x+1] += 1
        end
    end
    bad = Tuple{Int,Int}[]
    for x in 0:l-1, y in x+1:l-1
        mult[x+1, y+1] == 0 && push!(bad, (x, y))
    end
    return isempty(bad), bad
end
"""
    SharedMarkerLayout

The marker-reuse (shared marker) construction. In contrast to
`MarkerLayout` (one dedicated physical marker per ordered pair, l²−l of
them), here a small family of shared markers is reused across many
shifts:
- `l`       : number of shifts
- `t`       : marker load bound (max |S0|+|S1| over all markers)
- `s`       : ⌊t/2⌋, the per-half group size used by the construction
- `k`       : window size the layout protects against (markers are
              spaced by k+1 so any length-k window meets at most one)
- `N`       : chain length (one past the last marker site)
- `n`       : number of length-l blocks, ⌊N/l⌋
- `markers` : the M physical `Marker`s (already duplicated and assigned
              a physical site each)
- `Bj`      : per-shift marker-site sets; `Bj[j+1]` lists every physical
              site that constrains shift j (the set B_j of Eq. (38))
"""
struct SharedMarkerLayout <: AbstractMarkerLayout
    k::Integer
    ε::AbstractFloat
    t::Integer
    l::Integer
    s::Integer
    N::Integer
    n::Integer
    markers::Vector{Marker}
    Bj::Vector{Vector{Integer}}
end
"""
    SharedMarkerLayout(k::Integer, ε::AbstractFloat, t::Integer; R::Union{Integer,Nothing}=nothing)

Builds the marker-reuse layout for block size `l`, window size `k`, and
marker load `t`. Two useful choices of `t` (Section 3 of the marker-reuse
note): `t = max(k, 2)` ("supervisor scale", simplest, gives N₀ =
O(k²/ε²)) or `t = max(ceil(Int, sqrt(k)), 2)` (smaller error constant,
same asymptotic order).
Construction:
1. Build the combinatorial rows with `biclique_cover_rows` and verify
   they separate every pair with `verify_separating_rows` — throws if
   any pair is left unseparated (this should never trigger; it is a
   correctness guard on the construction, not a tuning knob).
2. Duplicate every row onto two physical sites at
   p_r = 2l^2 + r·(k+1), r = 0,...,P-1. Consecutive physical
   markers are then always ≥ k+1 apart, so any length-k window contains
   at most one, and both copies of a row's certificate can never fall in
   the same window — this is exactly what promotes "≥1 row separates the
   pair" to "≥2 spaced-apart certificates separate the pair", which is
   what the orthogonality/partial-trace argument actually needs.
3. `R` defaults to 2l², the purification reserve adopted in the
   thesis draft — large enough that both purification blocks of every
   shift (which live within the first l²+l sites, see `verify_all_
   purifications`) stay entirely marker-free.
4. A trailing buffer of 2l sites is left *after* the last marker before
   setting N. Reason: shift j=0's wrap-around purification block
   M_{0,0} = [l(n-1), N-1] sits at the very end of the chain (only j=0
   hits this, since `to_pos` only remaps block-index 0 for j=0 — every
   other shift's purification blocks are the ordinary, non-wrapping
   ones). Without the buffer, that tail block would overlap the last
   few markers, since they are otherwise packed right up to N. This is
   exactly the "endpoint and rollover" bookkeeping flagged, but not
   resolved, in the marker-reuse note's rigour checklist; the O(l)
   buffer added here is asymptotically irrelevant next to the O(l²)
   head reserve `R`.
"""
function SharedMarkerLayout(k::Integer, ε::AbstractFloat, t::Integer; R::Union{Integer,Nothing}=nothing)
    l = ceil(Int, (k + t + 1) / ε)
    rows = biclique_cover_rows(l, t)
    ok, bad = verify_separating_rows(rows, l)
    ok || throw(ErrorException("biclique_cover_rows(l=$l, t=$t) left $(length(bad)) pair(s) unseparated: $bad"))
    Rval = R === nothing ? 2*l^2 : R
    markers = Marker[]
    r = 0
    for (S0, S1) in rows   # duplicate: two physical copies per row
        push!(markers, Marker(S0, S1, Rval + r*(k+1)))
        r += 1
        push!(markers, Marker(S1, S0, Rval + r*(k+1)))
        r += 1
    end
    P = length(markers)
    N = Rval + (P - 1)*(k + 1) + 2*l   # +2l: tail buffer, see point 4 above
    n = fld(N, l)
    n <= l && @warn "n=$n ≤ l=$l: wrap-around block may collide with the " *
                    "purification blocks assumed clear below; increase N or check manually."
    Bj = [Int[] for _ in 1:l]
    for mk in markers
        for j in mk.S0
            push!(Bj[j+1], mk.site)
        end
        for j in mk.S1
            push!(Bj[j+1], mk.site)
        end
    end
    foreach(v -> (sort!(v); unique!(v)), Bj)
    s = fld(t, 2)
    return SharedMarkerLayout(k, ε, t, l, s, N, n, markers, Bj)
end
marker_sites_for_shift(layout::SharedMarkerLayout, j::Int) = layout.Bj[j+1]

"""
    print_shared_markers(layout::SharedMarkerLayout)

Prints every physical marker (site, S0, S1, load) and the resulting
per-shift marker sets B_j. Meant for small demo instances — for
realistic (k,ε) the marker count M is large; use `layout.markers` and
`layout.Bj` directly for anything programmatic.
"""
function print_shared_markers(layout::SharedMarkerLayout)
    println("=== Shared marker layout (bounded biclique cover, l=$(layout.l), " *
            "t=$(layout.t), s=$(layout.s), k=$(layout.k)) ===")
    println("Physical markers P = $(length(layout.markers)) " *
            "(combinatorial rows × 2 for the multiplicity-two duplication)")
    println("Chain length N = $(layout.N), blocks n = ⌊N/l⌋ = $(layout.n)")
    for (r, mk) in enumerate(layout.markers)
        load = length(mk.S0) + length(mk.S1)
        println("marker $r @ site $(mk.site): S⁰=$(mk.S0), S¹=$(mk.S1)  (load $load)")
    end
    println("\nPer-shift marker-site sets B_j:")
    for j in 0:layout.l-1
        println("  B_$j = $(layout.Bj[j+1])")
    end
end

"""
    N0_shared(k::Integer, ε::AbstractFloat; t::Union{Integer,Nothing}=nothing)

Chain length N₀ required by the marker-reuse construction for target
(k, ε). Defaults to the load t = max(k, 2), giving N₀ = O(k²/ε²),
pass `t` to try another load. Returns a named tuple (l, t, M, N0). This
only computes the sizes, it does not build the full `SharedMarkerLayout`
(which is O(P) work and unnecessary just to read off N₀).
"""
function N0_shared(k::Integer, ε::AbstractFloat; t::Union{Integer,Nothing}=nothing)
    tt = t === nothing ? max(k, 2) : t
    l = ceil(Int, (k + tt + 1) / ε)
    rows = biclique_cover_rows(l, tt)
    P = 2*length(rows)
    N0 = l^2 + (P - 1)*(k + 1) + 2*l # +2l: tail buffer, see point 4 above
    return (l=l, t=tt, P=P, N0=N0)
end

"""
    compare_all_constructions(k::Integer, ε::AbstractFloat)
Prints N₀ under all three marker constructions in this module, for the
same target (k, ε):
  1. paper's original layout   (`MarkerLayout`,        N₀ = O(k³/ε³))
  2. denser packing / free win (`dense_marker_layout`,  N₀ = O(k³/ε²))
  3. marker reuse / shared     (`SharedMarkerLayout`,   N₀ = O(k²/ε² + (k/ε)log k))
Constructions 1-2 use l = ⌈(k+3)/ε⌉ (their own error bound, Eq. (37));
construction 3 uses its own, larger l = ⌈(k+t+1)/ε⌉ (Eq. (26)) since its
error bound has an extra +t term from the shared marker's load — so the
three N₀ values are not at identical l, but each is the correct N₀ for
the *same* target (k, ε), which is the comparison that matters.
"""
function compare_all_constructions(k::Integer, ε::AbstractFloat)
    paper = MarkerLayout(k, ε; which=:paper)
    dense = MarkerLayout(k, ε; which=:dense)
    shared = N0_shared(k, ε)
    println("Target: k=$k, ε=$ε")
    println("  1. paper's original   : l=$(paper.l),  P=$(paper.l*(paper.l-1)),  N₀=$(paper.N)")
    println("  2. denser packing     : l=$(dense.l),  P=$(dense.l*(dense.l-1)),  N₀=$(dense.N)")
    println("  3. marker reuse       : l=$(shared.l),  t=$(shared.t),  P=$(shared.P),  N₀=$(shared.N0)")
    println("  reuse vs paper ratio  : $(round(paper.N / shared.N0, digits=1))×")
    println("  reuse vs dense ratio  : $(round(dense.N / shared.N0, digits=1))×")
end

# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------
function run_demo(; k::Integer=2, ε::AbstractFloat=0.5)
    println("=== Dalzel and Brandão marker construction (:paper) ===")
    layout = MarkerLayout(k, ε; which=:paper)
    println("k=$k, ε=$ε  →  l =  ⌈(k+3)/ε⌉ = $(layout.l)\n")
    print_markers(layout)
    println()
    verify_all_purifications(layout)
    println()
    shared_marker_report(layout)
    println()
    println("=== Dense marker spacing construction (:dense) ===")
    dense_layout =  MarkerLayout(k, ε; which=:dense)
    println("k=$k, ε=$ε  →  l = ⌈(k+3)/ε⌉ = $(layout.l)\n")
    print_markers(dense_layout)
    println()
    shared_marker_report(dense_layout)
    println()
    verify_all_purifications(dense_layout)
    println()
    # compare_packings(l, k)
    # println()

    println("=== Load bounded marker construction (:shared) ===")
    t = max(k, 2)
    shared_layout = SharedMarkerLayout(k, ε, t)
    println("k=$k, ε=$ε, t=k=$t →  l = ⌈(k+t+1)/ε⌉ = $(shared_layout.l)\n")
    print_shared_markers(shared_layout)
    println()
    verify_all_purifications(shared_layout)
    println()
    shared_marker_report(shared_layout)
    println()

    println("=== Comparison of constructions ===")
    compare_all_constructions(k, ε)
    # return layout
end
end # module
