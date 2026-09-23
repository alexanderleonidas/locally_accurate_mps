module LocallyAccurateMPS
using ITensors
using ITensorMPS

include("core/MatrixProductState.jl")
using .MatrixProductStates

include("core/MatrixProductOperators.jl")
using .MatrixProductOperators

include("quantum/DensityMatrix.jl")
using .DensityMatrix

include("quantum/Measures.jl")
using .Measures

include("models/AKLT.jl")
using .AKLT

include("lamps/MarkerGeometry.jl")
using .MarkerGeometry

include("lamps/DenseStateConstruction.jl")
using .DenseStateConstruction

include("lamps/StateConstruction.jl")
using .StateConstruction

include("lamps/TheoremVerification.jl")
using .TheoremVerification

export greet, main

function greet()
    println("Welcome to the LocallyAccurateMPS module!")
    println("Free memory: ", Base.Sys.free_memory() / 2^30, " GB")
    GC.gc()
end

function main()
    # LearningQudits.greet()
    # LearningQudits.show_example()

    # LearningTensors.greet()
    # LearningTensors.main()

    # MarkerGeometry.run_demo(;k=4, ε=0.9)

    # StateConstruction.run_demo(; k=2, epsilon=0.9)

    # TheoremVerification.verify_orthogonality(2, 0.9)

    # 1. Setup small parameters for debugging
    k = 2
    ε = 0.9

    # Works with both MarkerLayout or SharedMarkerLayout!
    layout = SharedMarkerLayout(k, ε, max(k, 2))
    sites = siteinds("S=1", layout.N)
    psi_target = aklt_ground_state(sites)
    # 2. Run step-by-step diagnostic for shift j = 0
    debug_phi_j_construction(psi_target, layout, 5)
end

end # module locally_accurate_mps
