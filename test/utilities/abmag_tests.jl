using CrowdPhot: abmag, abmag_err
using Test

@testset "abmag" begin
    # The AB system is defined so that 3631 Jy is exactly magnitude 0.
    @test abmag(3631.0) == 0.0
    # A factor of 100 in flux is exactly 5 magnitudes, by construction.
    @test abmag(36.31) ≈ 5.0 atol = 1.0e-12
    @test abmag(363100.0) ≈ -5.0 atol = 1.0e-12
    # 2.5 * log10(2) for a factor of two.
    @test abmag(3631.0 / 2) ≈ 2.5 * log10(2) atol = 1.0e-12

    @testset "non-positive flux" begin
        # Return NaN rather than throwing for non-positive flux.
        @test isnan(abmag(0.0))
        @test isnan(abmag(-1.0))
        @test isnan(abmag(-1.0f0))
    end

    @testset "element type" begin
        @test abmag(3631.0f0) isa Float32
        @test abmag(3631.0) isa Float64
        # Integers promote rather than erroring on `float`.
        @test abmag(3631) isa Float64
        @test abmag(3631) == 0.0
    end
end

@testset "abmag_err" begin
    @test abmag_err(100.0, 10.0) ≈ 2.5 / log(10) * 0.1 atol = 1.0e-12
    @test abmag_err(100.0, 0.0) == 0.0

    @testset "scale invariance" begin
        # The whole point: the ratio cancels units, so no calibration is needed
        # before calling this.  A caller passing raw fitted fluxes and a caller
        # passing janskys must agree exactly.
        f, e = 137.0, 4.25
        for s in (1.0e-7, 1.0e3, 3631.0)
            @test abmag_err(f * s, e * s) ≈ abmag_err(f, e) rtol = 1.0e-12
        end
    end

    @testset "non-positive flux" begin
        # Matches `abmag`, so a source is NaN in both columns or neither.
        @test isnan(abmag_err(0.0, 1.0))
        @test isnan(abmag_err(-1.0, 1.0))
        @test isnan(abmag(-1.0)) && isnan(abmag_err(-1.0, 1.0))
    end

    @testset "element type" begin
        @test abmag_err(100.0f0, 10.0f0) isa Float32
        @test abmag_err(100.0, 10.0) isa Float64
        # Mixed precision promotes to the wider type.
        @test abmag_err(100.0f0, 10.0) isa Float64
    end
end
