# CrowdPhot.jl
<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://crowdphot.github.io/CrowdPhot.jl/stable/) -->
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://crowdphot.github.io/CrowdPhot.jl/dev/)
[![Build Status](https://github.com/crowdphot/CrowdPhot.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/crowdphot/CrowdPhot.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/github/crowdphot/CrowdPhot.jl/graph/badge.svg?token=L69R23H29M)](https://codecov.io/github/crowdphot/CrowdPhot.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Installation

### Julia

If you need to install Julia, it is recommended to do so through [Juliaup](https://github.com/julialang/juliaup),
which is Julia's version multiplexer (similar to how `uv` lets you manage multiple Python versions).

Once `juliaup` is installed it should install the most recent stable version. You can install other versions if
needed with `juliaup add <X.X.X>` such as `juliaup add 1.13.0`. To set a default version that will be used when
you launch `julia`, set `juliaup default <X.X.X>`. To launch a non-default version of julia, execute, for example,
`julia +1.13.0`.

### Projects

Julia has built-in "projects" which serve the same purpose as virtual environments in Python, allowing for
self-contained dependency resolution on a per-project basis. When you start Julia as `julia`, it launches into
the **global** environment. Launching with `julia --project=my_project/` will start Julia using the Project.toml
dependency file located in the directory `my_project` (e.g., `julia --project=.` to use the
Project.toml in the current directory, or make a new one if none exists). The Project.toml is the user-facing file
where project dependencies are recorded and can have loose version specifiers for package dependencies, or no
version specifiers at all. The Manifest.toml is computer-generated and records the **exact versions** of the dependencies
that actually got installed. Only the Manifest.toml guarantees an **exactly** reproducible environment, but in practice
Project.toml is what people typically host and share.

We recommend installing CrowdPhot.jl into specific projects where you need it rather than into the global environment.

### Package

Start Julia (preferably pointing it to a project via the `julia --project=<path to project>` syntax) and enter
the package manager by hitting `]` at the REPL prompt. You will see the prompt change from `julia> ` to
`(my_project) pkg>` or similar, indicating you are in the package manager. Here you can add packages to your environment.

While CrowdPhot.jl is unregistered (i.e., not yet officially released) you have to add it from its git location; at the `pkg>` prompt, enter `add https://github.com/crowdphot/CrowdPhot.jl#main` to add CrowdPhot.jl as a dependency to this
project. It will be installed, with the installation linked to the main branch of the GitHub repo. You can update
CrowdPhot.jl and all other packages by issuing `update` from the package manager prompt. All of CrowdPhot.jl's dependencies
will be resolved and installed alongside it.

Return to the main Julia REPL prompt by hitting backspace when at the root level of the package manager prompt.
Type `using CrowdPhot` at the REPL prompt and it should complete successfully. You are ready to use CrowdPhot.jl
in your project.