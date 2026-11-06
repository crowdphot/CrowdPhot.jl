import CrowdPhot
using CrowdPhot.Roman: projection_regions, projection_region, region_to_world, skycells, SKYCELL_NXY, SKYCELL_BORDER, SKYCELL_SCALE
using CRDSClient: best_references, fetch_reference
using Test

# The bundled projection-region table and the skycell rules, against values read from the SKYCELLS reference
# (roman_wfi_skycells_0001.asdf).  With CROWDPHOT_TEST_SKYCELLS=true, CRDS must still select that file, and every
# one of its 8,045,034 skycells is checked against it (fetched with CRDSClient into $CRDS_PATH: 1.3 GB, once).

@testset "bundled projection regions" begin
    P = projection_regions()
    @test length(P) == 4058
    @test [r.index for r in P] == 0:4057
    @test first(P).skycell_start == 0 && last(P).skycell_end == 8045034
    @test all(P[k].skycell_end == P[k + 1].skycell_start for k in 1:(length(P) - 1))
    @test all(r -> r.orientat == 0 && r.x_tangent == (r.nx - 1) / 2 && r.y_tangent == (r.ny - 1) / 2, P)
    @test count(r -> r.dec_max == 90 || r.dec_min == -90, P) == 2
end

@testset "skycells of the region holding 1054 pointing 1" begin
    r = projection_region(268.5872, -28.5918)
    @test r.index == 2991 && r.ra_tangent == 270.0
    @test region_to_world(r, r.x_tangent, r.y_tangent) == (r.ra_tangent, r.dec_tangent)
    c = skycells(r)
    @test length(c) == r.skycell_end - r.skycell_start == 1978
    @test c.name[1:2] == ["270m27x27y29", "270m27x27y30"]
    k = findfirst(==("270m27x50y29"), c.name)
    @test k == 990    # 989 0-based in the file's region slice
    @test (c.x_tangent[k], c.y_tangent[k]) == (103299.5, 2499.5)
    @test c.ra_center[k] ≈ 268.26457673661304 atol = 1e-11
    @test c.dec_center[k] ≈ -27.475665122431217 atol = 1e-11
    @test all(isapprox.((c.ra_corn1[k], c.dec_corn1[k]), (268.2209461776189, -27.513301091531943); atol = 1e-11))
    @test all(isapprox.((c.ra_corn3[k], c.dec_corn3[k]), (268.30817905357026, -27.43801421428674); atol = 1e-11))
    @test c.orientat[k] == 1.73542320728302f0
    # +x east, +y north
    @test c.ra_corn2[k] > c.ra_corn1[k] && c.dec_corn4[k] > c.dec_corn1[k]
    @test skycells(projection_regions()[1]).name[1:2] == ["135p90x25y49", "135p90x25y50"]
end

# --- the full check against the reference file ---

# ASDF block header at byte `i` (0-based) of `b`: data start (0-based), allocated and used sizes, compression.
function block_header(b, i)
    b[(i + 1):(i + 4)] == [0xd3, UInt8('B'), UInt8('L'), UInt8('K')] || error("no ASDF block at byte $i")
    be(T, o) = ntoh(reinterpret(T, b[(o + 1):(o + sizeof(T))])[1])
    hs = be(UInt16, i + 4)
    return (; data = i + 6 + Int(hs), alloc = Int(be(UInt64, i + 14)), used = Int(be(UInt64, i + 22)), compression = b[(i + 11):(i + 14)])
end

const SC_RECORD = 164   # name (16 UCS4), ra_center, dec_center (f8), orientat (f4), x_tangent, y_tangent, 8 corner coordinates (f8)

le(T, b, o) = ltoh(reinterpret(T, b[(o + 1):(o + sizeof(T))])[1])
ucs4(b, o) = String([Char(u) for u in reinterpret(UInt32, b[(o + 1):(o + 64)]) if u != 0])

if get(ENV, "CROWDPHOT_TEST_SKYCELLS", "false") == "true"
    @testset "every skycell against the SKYCELLS reference" begin
        # a far-future date selects the newest delivery: a new skycells file fails here, before the comparison
        sel = best_references(Dict("ROMAN.META.INSTRUMENT.NAME" => "WFI", "ROMAN.META.EXPOSURE.START_TIME" => "2100-01-01 00:00:00");
                              reftypes = ["skycells"], observatory = "roman")
        @test sel["skycells"] == "roman_wfi_skycells_0001.asdf"
        path = fetch_reference("roman_wfi_skycells_0001.asdf")
        b = read(path)
        first_block = findfirst([0xd3, UInt8('B'), UInt8('L'), UInt8('K')], b)
        tree = String(b[1:(first(first_block) - 1)])
        @test occursin("nxy_skycell: $SKYCELL_NXY", tree) && occursin("skycell_border_pixels: $SKYCELL_BORDER", tree)
        @test occursin("pixel_scale: $SKYCELL_SCALE", tree)
        @test occursin("shape: [4058]", tree) && occursin("shape: [8045034]", tree)
        h0 = block_header(b, first(first_block) - 1)
        h1 = block_header(b, h0.data + h0.alloc)
        @test all(iszero, h0.compression) && all(iszero, h1.compression)
        @test h1.used == 8045034 * SC_RECORD
        bundled = read(joinpath(pkgdir(CrowdPhot), "data", "roman_wfi_skycells_0001_projection_regions.bin"))
        @test b[(h0.data + 1):(h0.data + h0.used)] == bundled

        P = projection_regions()
        n_count = Threads.Atomic{Int}(0); n_name = Threads.Atomic{Int}(0); n_tangent = Threads.Atomic{Int}(0); n_orient = Threads.Atomic{Int}(0)
        err_center = zeros(length(P)); err_corner = zeros(length(P)); err_orient = zeros(length(P))
        sep(ra1, dec1, ra2, dec2) = max(abs(rem(ra1 - ra2, 360, RoundNearest)) * cosd(dec2), abs(dec1 - dec2))
        Threads.@threads for k in eachindex(P)
            r = P[k]; c = skycells(r)
            if length(c) != r.skycell_end - r.skycell_start
                Threads.atomic_add!(n_count, 1); continue
            end
            for (m, s) in enumerate(r.skycell_start:(r.skycell_end - 1))
                o = h1.data + s * SC_RECORD
                ucs4(b, o) == c.name[m] || Threads.atomic_add!(n_name, 1)
                (le(Float64, b, o + 84), le(Float64, b, o + 92)) == (c.x_tangent[m], c.y_tangent[m]) || Threads.atomic_add!(n_tangent, 1)
                err_center[k] = max(err_center[k], sep(c.ra_center[m], c.dec_center[m], le(Float64, b, o + 64), le(Float64, b, o + 72)))
                ours = (c.ra_corn1[m], c.dec_corn1[m], c.ra_corn2[m], c.dec_corn2[m], c.ra_corn3[m], c.dec_corn3[m], c.ra_corn4[m], c.dec_corn4[m])
                for q in 1:4
                    err_corner[k] = max(err_corner[k], sep(ours[2q - 1], ours[2q], le(Float64, b, o + 100 + 16(q - 1)), le(Float64, b, o + 108 + 16(q - 1))))
                end
                # equal modulo 360 to Float32 rounding: the file holds floating-point residues (~1e-14) where the offset
                # is 0, and one polar cell at -225 for 135
                t = le(Float32, b, o + 80); d = abs(rem(Float64(c.orientat[m]) - t, 360, RoundNearest))
                err_orient[k] = max(err_orient[k], d)
                d <= eps(abs(t)) + 1e-12 || Threads.atomic_add!(n_orient, 1)
            end
        end
        println("skycells vs the reference: max center error $(maximum(err_center)) deg, corner $(maximum(err_corner)) deg, orientat $(maximum(err_orient)) deg")
        @test n_count[] == 0
        @test n_name[] == 0
        @test n_tangent[] == 0
        @test n_orient[] == 0
        @test maximum(err_center) < 1e-11
        @test maximum(err_corner) < 1e-11
    end
else
    @info "Skipping the check of every skycell against the SKYCELLS reference (1.3 GB); set CROWDPHOT_TEST_SKYCELLS=true to run it"
end
