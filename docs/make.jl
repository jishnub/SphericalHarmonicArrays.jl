using SphericalHarmonicArrays
using Documenter

DocMeta.setdocmeta!(SphericalHarmonicArrays, :DocTestSetup, :(using SphericalHarmonicArrays); recursive=true)

makedocs(;
    modules=[SphericalHarmonicArrays],
    authors="Jishnu Bhattacharya",
    repo="https://github.com/jishnub/SphericalHarmonicArrays.jl/blob/{commit}{path}#L{line}",
    sitename="SphericalHarmonicArrays",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jishnub.github.io/SphericalHarmonicArrays.jl",
        assets=String[],
    ),
    pages=[
        "Reference" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jishnub/SphericalHarmonicArrays.jl",
)
