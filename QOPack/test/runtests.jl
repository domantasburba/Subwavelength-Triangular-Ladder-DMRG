using QOPack
using Test

@testset "QOPack.jl" begin
    ##### Math.jl #####
    @test 1/2*Pauli_x() == create_S_x(1/2)
    @test 1/2*Pauli_y() == create_S_y(1/2)
    @test 1/2*Pauli_z() == create_S_z(1/2)

    ### sqrt_Gaussian and Gaussian ###
    # Randomized input test
    N_rand = 1000
    σ_range = 0.01:0.01:10
    μ_range = -3:0.01:3
    for i ∈ 1:N_rand
        σ = rand(σ_range)
        μ = rand(μ_range)
        x = rand((μ-5σ:0.01:μ+5σ))

        sqrt_Gaussian_xμσ = sqrt_Gaussian(x, μ, σ)
        Gaussian_xμσ = Gaussian(x, μ, σ)

        # sqrt_Gaussian^2 == Gaussian for any inputs
        @test isapprox(sqrt_Gaussian_xμσ^2, Gaussian_xμσ, atol=1e-7)
        # Always positive for all real inputs
        @test sqrt_Gaussian_xμσ > 0.0
        @test Gaussian_xμσ > 0.0
    end # for

    # Edge cases
    # Results should only depend on x-μ
    ### END OF sqrt_Gaussian and Gaussian ###
    ##### END OF Math.jl #####
end

# using Jive
# runtests(@__DIR__)
