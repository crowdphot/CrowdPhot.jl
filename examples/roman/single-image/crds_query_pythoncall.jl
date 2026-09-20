# crds_query_pythoncall.jl
#
# Resolving Roman reference files (ePSF, pixel area map, ...) by calling the
# real CRDS Python package from Julia through PythonCall.jl.
#
# There is no Julia implementation of CRDS, and picking the right ePSF by hand
# per (detector, filter) is not practical, so this example depends on Python.
# `CondaPkg.toml` installs `crds` into a private environment automatically; see
# README.md for how to point at an existing Python instead.

# Without this, every CRDS call fails with "CERTIFICATE_VERIFY_FAILED: unable to
# get local issuer certificate", even though the same interpreter reaches the
# same server fine when run from a shell.  The chain is:
#
#   1. Julia has already loaded its own `lib/julia/libssl.so.3` by the time
#      CPython dlopens `_ssl`.  A library with that SONAME is therefore already
#      in the process, so the loader reuses it instead of following the
#      interpreter's RPATH to its own copy.  `/proc/self/maps` confirms it:
#      embedded, only Julia's libssl and libcrypto are mapped.
#   2. OpenSSL bakes its default certificate locations in at compile time, and
#      Julia's is built by BinaryBuilder inside a container, so they come out as
#      `/workspace/destdir/ssl/cert.pem` and `/workspace/destdir/ssl/certs`.
#      Those paths do not exist on any machine that runs Julia.
#   3. CPython's `ssl.get_default_verify_paths()` returns `None` for any default
#      that is not an existing file or directory, so the trust store is empty
#      and verification cannot succeed.
#
# Julia's own HTTPS is unaffected, which is why this is not a well-known
# problem: Julia never relies on those compiled-in defaults, it hands libcurl a
# bundle from NetworkOptions explicitly.  Only an embedded interpreter, which
# does rely on them, gets caught.  So we do the same thing here.
#
# This has to happen before `using PythonCall`, and it is skipped if the caller
# has already chosen a bundle.
import NetworkOptions
get!(ENV, "SSL_CERT_FILE", NetworkOptions.ca_roots_path())

using PythonCall

const crds = pyimport("crds")
const data_file = pyimport("crds.data_file")
const client_api = pyimport("crds.client.api")

const OBSERVATORY = "roman"

# CRDS needs a cache directory and a server to talk to.  Neither has a usable
# default, and without them the failure surfaces much later as a confusing
# Python traceback, so check up front.
for var in ("CRDS_PATH", "CRDS_SERVER_URL")
    haskey(ENV, var) || error("$var is not set; CRDS cannot resolve reference files. " *
                              "See README.md in this directory.")
end

# crds.data_file.get_free_header() is the same header flattening CRDS uses
# internally (crds/io/abstract.py::AbstractFile.to_simple_types): the nested
# ASDF tree becomes dotted, upper-cased, stringified keys, e.g.
# "ROMAN.META.INSTRUMENT.DETECTOR" => "WFI01".  Calling the real extractor
# guarantees the header matches whatever CRDS itself would read from the file.
crds_header(asdf_path::AbstractString) =
    data_file.get_free_header(asdf_path, (), nothing, OBSERVATORY)

# Resolve the pipeline context once and reuse it for every lookup.
# Re-resolving "latest" per call risks a lookup silently switching rulesets
# mid-session if the server takes a delivery while this process is running.
const CONTEXT = pyconvert(String, client_api.get_default_context(OBSERVATORY))

"""
    lookup_references(asdf_path, reftypes) -> Dict{String, String}

Reference *basenames* for `asdf_path`, without downloading anything.  Mirrors
`crds.getrecommendations`: returns a `"NOT FOUND ..."` string rather than
raising for a reftype that does not apply or has no match.
"""
function lookup_references(asdf_path::AbstractString, reftypes::Vector{String})
    header = crds_header(asdf_path)
    result = crds.getrecommendations(header; reftypes = pylist(reftypes),
                                     context = CONTEXT, observatory = OBSERVATORY)
    return Dict(pyconvert(String, k) => pyconvert(String, v) for (k, v) in result.items())
end

"""
    get_references(asdf_path, reftypes) -> Dict{String, String}

Local *file paths* for `asdf_path`'s reference files, downloading into
`\$CRDS_PATH` if they are not already cached.  Mirrors `crds.getreferences`.

Unlike [`lookup_references`](@ref) this raises a Python exception if a
requested reftype has no match at all (as opposed to "not applicable", which is
skipped silently).  Reference files are large: one ePSF is roughly 180 MB.
"""
function get_references(asdf_path::AbstractString, reftypes::Vector{String})
    header = crds_header(asdf_path)
    result = crds.getreferences(header; reftypes = pylist(reftypes),
                                context = CONTEXT, observatory = OBSERVATORY)
    return Dict(pyconvert(String, k) => pyconvert(String, v) for (k, v) in result.items())
end
