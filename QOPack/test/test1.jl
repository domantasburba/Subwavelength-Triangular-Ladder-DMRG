using QOPack
using Test

@testset "QOPack.jl" begin
    # Write your tests here.

    # Math.jl
    @test 1/2*Pauli_x() == create_S_x(1/2)
    @test 1/2*Pauli_y() == create_S_y(1/2)
    @test 1/2*Pauli_z() == create_S_z(1/2)
end
