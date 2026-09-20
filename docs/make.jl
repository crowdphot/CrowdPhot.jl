using CrowdPhot
using CrowdPhot.PSF
using CrowdPhot.Background
using Documenter
# using Documenter.Remotes: GitHub
using DocumenterCitations: CitationBibliography

setup = quote
    using CrowdPhot
    using CrowdPhot.PSF
    using CrowdPhot.Background
end

DocMeta.setdocmeta!(CrowdPhot, :DocTestSetup, setup; recursive = true)

const CI = get(ENV, "CI", "false") == "true"
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
    sitename = "CrowdPhot.jl",
    modules = [CrowdPhot, CrowdPhot.PSF, CrowdPhot.Background],
    format = Documenter.HTML(;
        prettyurls = CI,
        assets = String[],
    ),
    authors = "Chris Garling",
    # repo = GitHub("crowdphot/CrowdPhot.jl"),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Pixel Coordinate Conventions" => "pixel_conventions.md",
        ],
        "Tutorials" => [
            "HST ACS DRC" => "tutorials/HST-DRC.md",
        ],
        "Background Estimation" => "background.md",
        "Detection" => "detection.md",
        "Centroid Refinement and Morphology" => "morphology.md",
        # "Candidate Selection" => "picking.md",
        "PSFs" => [
            "PSF API" => "psf/psf_api.md",
            "Parametric PSF Models" => "psf/parametric_models.md",
            "Effective PSF Models" => [
                "ePSF Overview" => "psf/empirical/epsf_overview.md",
                "ImagePSF" => "psf/empirical/image_psf.md",
            ],
            "Spatially Varying PSF Models" => [
                "GriddedPSFModel" => "psf/variable/gridded_psf.md",
            ],
            "Picking PSF Stars" => "psf/picking.md",
        ],
        "Photometry" => [
            "PSF Fitting Photometry" => "photometry/psf_fitting.md",
            "Curves of Growth" => "photometry/curve_of_growth.md",
        ],
        "Observatories" => [
            "Roman" => "observatories/roman.md",
        ],
        "Levenberg-Marquardt Fitter" => "lm_fitter.md",
        "Simulation" => "simulation.md",
        "Utilities" => "utilities.md",
        "Index" => "doc_index.md",
        "References" => "refs.md",
        ],
    doctest = true,
    linkcheck = CI,
    warnonly = [:missing_docs, :linkcheck],
    plugins = [bib],
)

deploydocs(;
    repo = "github.com/crowdphot/CrowdPhot.jl.git",
    devbranch = "main",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#"], # Restrict to minor releases
)