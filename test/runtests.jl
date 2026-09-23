using Test
using ITensors
using ITensorMPS
using LinearAlgebra

include("../src/LocallyAccurateMPS.jl")
using .LocallyAccurateMPS
using .LocallyAccurateMPS.MatrixProductStates
using .LocallyAccurateMPS.MatrixProductOperators
using .LocallyAccurateMPS.AKLT
using .LocallyAccurateMPS.StateConstruction
using .LocallyAccurateMPS.DenseStateConstruction
using .LocallyAccurateMPS.DensityMatrix
using .LocallyAccurateMPS.Measures

@testset "All MPS/MPO Tests" begin
    @testset "AKLT Model Tests" begin
        include("models/test_aklt.jl")
    end

    @testset "State Construction Tests" begin
        include("lamps/test_lemma5.jl")
    end

    @testset "Quantum Density Matrix & Measures Tests" begin
        include("quantum/test_density_matrix.jl")
        include("quantum/test_measures.jl")
    end
end
