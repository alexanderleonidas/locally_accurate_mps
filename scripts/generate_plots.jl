include("../src/plotting/chain_length.jl")
include("../src/plotting/plot_constructions.jl")


benchmark_chain_lengths_per_method(k_vals = [2, 4,6,8,10,12], ε_min = 0.9, ε_max = 0.01)


# Plot the scaling of ε over k for different methods (Original, Dense, Shared Reuse)
# k = [2,4,6,8,10,12]
# for sym in [:paper, :dense, :shared]
#     run_epsilon_plot_over_k(k=k, Pmin=10^2, Pmax=10^7, npoints=200, which=sym)
# end
