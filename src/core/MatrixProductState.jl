module MatrixProductStates

using ITensors
using ITensorMPS

export truncate_bond_dimension, get_bond_dimensions, analyze_bond_dimension_spectrum

"""
    truncate_bond_dimension(mps::MPS, maxdim::Int) -> MPS

Truncate an MPS to a maximum bond dimension using ITensor's truncation sweep.
"""
function truncate_bond_dimension(mps::MPS, maxdim::Int)::MPS
    mps_trunc = copy(mps)
    truncate!(mps_trunc; maxdim=maxdim)
    return mps_trunc
end

"""
    get_bond_dimensions(mps::MPS) -> Vector{Int}

Extract the bond dimension at each virtual link in the MPS.
"""
function get_bond_dimensions(mps::MPS)::Vector{Int}
    N = length(mps)
    dims = Vector{Int}(undef, N + 1)

    # Boundary bond dimensions are 1
    dims[1] = 1
    dims[N + 1] = 1

    # Extract interior virtual bond dimensions
    for i in 1:N-1
        l = linkind(mps, i)
        dims[i + 1] = (l === nothing) ? 1 : dim(l)
    end

    return dims
end

"""
    analyze_bond_dimension_spectrum(mps::MPS) -> Dict

Analyze the spectrum of bond dimensions across the chain.
"""
function analyze_bond_dimension_spectrum(mps::MPS)::Dict
    dims = get_bond_dimensions(mps)

    return Dict(
        "min_chi" => minimum(dims),
        "max_chi" => maximum(dims),
        "mean_chi" => sum(dims) / length(dims),
        "dims" => dims
    )
end

end # module MatrixProductStates
