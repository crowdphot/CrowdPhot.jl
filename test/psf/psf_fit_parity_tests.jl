using CrowdPhot.PSF
using CrowdPhot.PSF: AbstractPSFModel, fit_star
using Test

@testset "fit CircularGaussianPSF specialized parity" begin
    # Compare circular-Gaussian specialized accumulation against the generic fitter.
    inds = (1:30, 1:30)
    truth = CircularGaussianPSF(x = 13.5, y = 12.3, fwhm = 4.0, flux = 200.0, bkg = 5.0)
    img = evaluate.(truth, inds[1], inds[2]')
    init = CircularGaussianPSF(x = 14.1, y = 11.8, fwhm = 4.3, flux = 190.0, bkg = 5.5)
    generic_sig = Tuple{AbstractPSFModel{Float64}, AbstractMatrix, Any}

    best_generic, result_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-6,
    )
    best_specialized, result_specialized = fit_star(
        init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-6,
    )

    @test result_specialized.converged == result_generic.converged
    @test result_specialized.minimizer ≈ result_generic.minimizer rtol = 1.0e-12 atol = 1.0e-12
    @test result_specialized.minimum ≈ result_generic.minimum rtol = 1.0e-12 atol = 1.0e-12
    @test best_specialized.x ≈ best_generic.x rtol = 1.0e-12 atol = 1.0e-12
    @test best_specialized.y ≈ best_generic.y rtol = 1.0e-12 atol = 1.0e-12
    @test best_specialized.fwhm ≈ best_generic.fwhm rtol = 1.0e-12 atol = 1.0e-12
    @test best_specialized.flux ≈ best_generic.flux rtol = 1.0e-12 atol = 1.0e-12
    @test best_specialized.bkg ≈ best_generic.bkg rtol = 1.0e-12 atol = 1.0e-12

    # Exercise the projected-parameter accumulator used when fields are fixed.
    fixed = (bkg = truth.bkg,)
    best_fixed_generic, result_fixed_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        fixed,
        x_tol = 1.0e-6,
    )
    best_fixed_specialized, result_fixed_specialized = fit_star(
        init, img, inds;
        fixed,
        x_tol = 1.0e-6,
    )

    @test result_fixed_specialized.minimizer ≈ result_fixed_generic.minimizer rtol = 1.0e-12 atol = 1.0e-12
    @test result_fixed_specialized.minimum ≈ result_fixed_generic.minimum rtol = 1.0e-12 atol = 1.0e-12
    @test best_fixed_specialized.x ≈ best_fixed_generic.x rtol = 1.0e-12 atol = 1.0e-12
    @test best_fixed_specialized.y ≈ best_fixed_generic.y rtol = 1.0e-12 atol = 1.0e-12
    @test best_fixed_specialized.fwhm ≈ best_fixed_generic.fwhm rtol = 1.0e-12 atol = 1.0e-12
    @test best_fixed_specialized.flux ≈ best_fixed_generic.flux rtol = 1.0e-12 atol = 1.0e-12
    @test best_fixed_specialized.bkg ≈ best_fixed_generic.bkg rtol = 1.0e-12 atol = 1.0e-12
end

@testset "fit GaussianPSF specialized parity" begin
    # Compare elliptical-Gaussian specialized accumulation against the generic fitter.
    inds = (1:21, 1:21)
    truth = GaussianPSF(x = 10.4, y = 9.8, x_fwhm = 3.0, y_fwhm = 4.2, theta = 28.0, flux = 250.0, bkg = 4.0)
    img = evaluate.(truth, inds[1], inds[2]')
    init = GaussianPSF(x = 10.8, y = 9.4, x_fwhm = 3.3, y_fwhm = 3.9, theta = 24.0, flux = 230.0, bkg = 4.4)
    generic_sig = Tuple{AbstractPSFModel{Float64}, AbstractMatrix, Any}

    best_generic, result_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_specialized, result_specialized = fit_star(
        init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_specialized.converged == result_generic.converged
    @test result_specialized.minimizer ≈ result_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_specialized.minimum ≈ result_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x ≈ best_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y ≈ best_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x_fwhm ≈ best_generic.x_fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y_fwhm ≈ best_generic.y_fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.theta ≈ best_generic.theta rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.flux ≈ best_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.bkg ≈ best_generic.bkg rtol = 1.0e-11 atol = 1.0e-11

    # Exercise the projected-parameter accumulator for fixed structural fields.
    fixed = (x_fwhm = truth.x_fwhm, y_fwhm = truth.y_fwhm, theta = truth.theta, bkg = truth.bkg)
    best_fixed_generic, result_fixed_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_fixed_specialized, result_fixed_specialized = fit_star(
        init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_fixed_specialized.minimizer ≈ result_fixed_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_fixed_specialized.minimum ≈ result_fixed_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x ≈ best_fixed_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y ≈ best_fixed_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x_fwhm ≈ best_fixed_generic.x_fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y_fwhm ≈ best_fixed_generic.y_fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.theta ≈ best_fixed_generic.theta rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.flux ≈ best_fixed_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.bkg ≈ best_fixed_generic.bkg rtol = 1.0e-11 atol = 1.0e-11
end

@testset "fit CircularGaussianPRF specialized parity" begin
    # Compare circular-Gaussian PRF specialized accumulation against the generic fitter.
    inds = (1:21, 1:21)
    truth = CircularGaussianPRF(x = 10.4, y = 9.8, fwhm = 3.0, flux = 250.0, bkg = 4.0)
    img = evaluate.(truth, inds[1], inds[2]')
    init = CircularGaussianPRF(x = 10.8, y = 9.4, fwhm = 3.3, flux = 230.0, bkg = 4.4)
    generic_sig = Tuple{AbstractPSFModel{Float64}, AbstractMatrix, Any}

    best_generic, result_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_specialized, result_specialized = fit_star(
        init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_specialized.converged == result_generic.converged
    @test result_specialized.minimizer ≈ result_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_specialized.minimum ≈ result_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x ≈ best_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y ≈ best_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.fwhm ≈ best_generic.fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.flux ≈ best_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.bkg ≈ best_generic.bkg rtol = 1.0e-11 atol = 1.0e-11

    # Exercise the projected-parameter accumulator for fixed structural fields.
    fixed = (fwhm = truth.fwhm, bkg = truth.bkg)
    best_fixed_generic, result_fixed_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_fixed_specialized, result_fixed_specialized = fit_star(
        init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_fixed_specialized.minimizer ≈ result_fixed_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_fixed_specialized.minimum ≈ result_fixed_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x ≈ best_fixed_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y ≈ best_fixed_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.fwhm ≈ best_fixed_generic.fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.flux ≈ best_fixed_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.bkg ≈ best_fixed_generic.bkg rtol = 1.0e-11 atol = 1.0e-11
end

@testset "fit GaussianPRF specialized parity" begin
    # Compare elliptical-Gaussian PRF specialized accumulation against the generic fitter.
    inds = (1:21, 1:21)
    truth = GaussianPRF(x = 10.4, y = 9.8, x_fwhm = 3.0, y_fwhm = 4.2, theta = 28.0, flux = 250.0, bkg = 4.0)
    img = evaluate.(truth, inds[1], inds[2]')
    init = GaussianPRF(x = 10.8, y = 9.4, x_fwhm = 3.3, y_fwhm = 3.9, theta = 24.0, flux = 230.0, bkg = 4.4)
    generic_sig = Tuple{AbstractPSFModel{Float64}, AbstractMatrix, Any}

    best_generic, result_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_specialized, result_specialized = fit_star(
        init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_specialized.converged == result_generic.converged
    @test result_specialized.minimizer ≈ result_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_specialized.minimum ≈ result_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x ≈ best_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y ≈ best_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x_fwhm ≈ best_generic.x_fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y_fwhm ≈ best_generic.y_fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.theta ≈ best_generic.theta rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.flux ≈ best_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.bkg ≈ best_generic.bkg rtol = 1.0e-11 atol = 1.0e-11

    # Exercise the projected-parameter accumulator for fixed structural fields.
    fixed = (x_fwhm = truth.x_fwhm, y_fwhm = truth.y_fwhm, theta = truth.theta, bkg = truth.bkg)
    best_fixed_generic, result_fixed_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_fixed_specialized, result_fixed_specialized = fit_star(
        init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_fixed_specialized.minimizer ≈ result_fixed_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_fixed_specialized.minimum ≈ result_fixed_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x ≈ best_fixed_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y ≈ best_fixed_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x_fwhm ≈ best_fixed_generic.x_fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y_fwhm ≈ best_fixed_generic.y_fwhm rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.theta ≈ best_fixed_generic.theta rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.flux ≈ best_fixed_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.bkg ≈ best_fixed_generic.bkg rtol = 1.0e-11 atol = 1.0e-11
end

@testset "fit CircularMoffatPSF specialized parity" begin
    # Compare circular-Moffat specialized accumulation against the generic fitter.
    inds = (1:21, 1:21)
    truth = CircularMoffatPSF(x = 10.4, y = 9.8, α = 3.0, β = 3.2, flux = 250.0, bkg = 4.0)
    img = evaluate.(truth, inds[1], inds[2]')
    init = CircularMoffatPSF(x = 10.8, y = 9.4, α = 3.3, β = 2.9, flux = 230.0, bkg = 4.4)
    generic_sig = Tuple{AbstractPSFModel{Float64}, AbstractMatrix, Any}

    best_generic, result_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_specialized, result_specialized = fit_star(
        init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_specialized.converged == result_generic.converged
    @test result_specialized.minimizer ≈ result_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_specialized.minimum ≈ result_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x ≈ best_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y ≈ best_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.α ≈ best_generic.α rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.β ≈ best_generic.β rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.flux ≈ best_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.bkg ≈ best_generic.bkg rtol = 1.0e-11 atol = 1.0e-11

    # Exercise the projected-parameter accumulator for fixed structural fields.
    fixed = (α = truth.α, β = truth.β, bkg = truth.bkg)
    best_fixed_generic, result_fixed_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_fixed_specialized, result_fixed_specialized = fit_star(
        init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_fixed_specialized.minimizer ≈ result_fixed_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_fixed_specialized.minimum ≈ result_fixed_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x ≈ best_fixed_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y ≈ best_fixed_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.α ≈ best_fixed_generic.α rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.β ≈ best_fixed_generic.β rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.flux ≈ best_fixed_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.bkg ≈ best_fixed_generic.bkg rtol = 1.0e-11 atol = 1.0e-11
end

@testset "fit MoffatPSF specialized parity" begin
    # Compare elliptical-Moffat specialized accumulation against the generic fitter.
    inds = (1:21, 1:21)
    truth = MoffatPSF(x = 10.4, y = 9.8, x_α = 3.0, y_α = 4.2, theta = 28.0, β = 3.2, flux = 250.0, bkg = 4.0)
    img = evaluate.(truth, inds[1], inds[2]')
    init = MoffatPSF(x = 10.8, y = 9.4, x_α = 3.3, y_α = 3.9, theta = 24.0, β = 2.9, flux = 230.0, bkg = 4.4)
    generic_sig = Tuple{AbstractPSFModel{Float64}, AbstractMatrix, Any}

    best_generic, result_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_specialized, result_specialized = fit_star(
        init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_specialized.converged == result_generic.converged
    @test result_specialized.minimizer ≈ result_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_specialized.minimum ≈ result_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x ≈ best_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y ≈ best_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x_α ≈ best_generic.x_α rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y_α ≈ best_generic.y_α rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.theta ≈ best_generic.theta rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.β ≈ best_generic.β rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.flux ≈ best_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.bkg ≈ best_generic.bkg rtol = 1.0e-11 atol = 1.0e-11

    # Exercise the projected-parameter accumulator for fixed structural fields.
    fixed = (x_α = truth.x_α, y_α = truth.y_α, theta = truth.theta, β = truth.β, bkg = truth.bkg)
    best_fixed_generic, result_fixed_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_fixed_specialized, result_fixed_specialized = fit_star(
        init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_fixed_specialized.minimizer ≈ result_fixed_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_fixed_specialized.minimum ≈ result_fixed_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x ≈ best_fixed_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y ≈ best_fixed_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x_α ≈ best_fixed_generic.x_α rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y_α ≈ best_fixed_generic.y_α rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.theta ≈ best_fixed_generic.theta rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.β ≈ best_fixed_generic.β rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.flux ≈ best_fixed_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.bkg ≈ best_fixed_generic.bkg rtol = 1.0e-11 atol = 1.0e-11
end

@testset "fit AiryPSF specialized parity" begin
    # Compare Airy specialized accumulation against the generic fitter.
    inds = (1:21, 1:21)
    truth = AiryPSF(x = 10.4, y = 9.8, radius = 4.5, flux = 250.0, bkg = 4.0)
    img = evaluate.(truth, inds[1], inds[2]')
    init = AiryPSF(x = 10.8, y = 9.4, radius = 4.8, flux = 230.0, bkg = 4.4)
    generic_sig = Tuple{AbstractPSFModel{Float64}, AbstractMatrix, Any}

    best_generic, result_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_specialized, result_specialized = fit_star(
        init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_specialized.converged == result_generic.converged
    @test result_specialized.minimizer ≈ result_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_specialized.minimum ≈ result_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x ≈ best_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y ≈ best_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.radius ≈ best_generic.radius rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.flux ≈ best_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.bkg ≈ best_generic.bkg rtol = 1.0e-11 atol = 1.0e-11

    # Exercise the projected-parameter accumulator for fixed structural fields.
    fixed = (radius = truth.radius, bkg = truth.bkg)
    best_fixed_generic, result_fixed_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_fixed_specialized, result_fixed_specialized = fit_star(
        init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_fixed_specialized.minimizer ≈ result_fixed_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_fixed_specialized.minimum ≈ result_fixed_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x ≈ best_fixed_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y ≈ best_fixed_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.radius ≈ best_fixed_generic.radius rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.flux ≈ best_fixed_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.bkg ≈ best_fixed_generic.bkg rtol = 1.0e-11 atol = 1.0e-11
end

@testset "fit ImagePSF specialized parity" begin
    # Compare ImagePSF specialized accumulation against the generic fitter.
    inds = (1:16, 1:16)
    grid_model = CircularGaussianPRF(x = 8.0, y = 8.0, fwhm = 2.4, flux = 1.0, bkg = 0.0)
    psf_data = evaluate.(grid_model, inds[1], inds[2]')
    truth = ImagePSF(psf_data; x = 8.35, y = 7.75, flux = 300.0, bkg = 4.0, origin = (8.0, 8.0), normalize = true)
    img = evaluate.(truth, inds[1], inds[2]')
    init = ImagePSF(psf_data; x = 8.0, y = 8.1, flux = 260.0, bkg = 3.5, origin = (8.0, 8.0), normalize = true)
    generic_sig = Tuple{AbstractPSFModel{Float64}, AbstractMatrix, Any}

    best_generic, result_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_specialized, result_specialized = fit_star(
        init, img, inds;
        inv_var = fill(1.0, size(img)),
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_specialized.converged == result_generic.converged
    @test result_specialized.minimizer ≈ result_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_specialized.minimum ≈ result_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.x ≈ best_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.y ≈ best_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.flux ≈ best_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_specialized.bkg ≈ best_generic.bkg rtol = 1.0e-11 atol = 1.0e-11

    # Exercise the projected-parameter accumulator used for fixed ImagePSF fields.
    fixed = (x = truth.x, y = truth.y, bkg = truth.bkg)
    best_fixed_generic, result_fixed_generic = invoke(
        fit_star, generic_sig, init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )
    best_fixed_specialized, result_fixed_specialized = fit_star(
        init, img, inds;
        fixed,
        x_tol = 1.0e-7,
        max_iter = 100,
    )

    @test result_fixed_specialized.minimizer ≈ result_fixed_generic.minimizer rtol = 1.0e-11 atol = 1.0e-11
    @test result_fixed_specialized.minimum ≈ result_fixed_generic.minimum rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.x ≈ best_fixed_generic.x rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.y ≈ best_fixed_generic.y rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.flux ≈ best_fixed_generic.flux rtol = 1.0e-11 atol = 1.0e-11
    @test best_fixed_specialized.bkg ≈ best_fixed_generic.bkg rtol = 1.0e-11 atol = 1.0e-11
end
