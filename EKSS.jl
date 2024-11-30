### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 4059337b-d0c5-4948-94b6-c47c229625da
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 15d6d724-e3ab-443c-99c6-3521976533d8
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod

# ╔═╡ 1ca8b538-af39-11ef-2006-2778f9ccd5c2
html"""<style>
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 96%;
    margin-left: 0%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 46afcea3-a754-4731-81f9-e2a4c880d61d


# ╔═╡ Cell order:
# ╠═1ca8b538-af39-11ef-2006-2778f9ccd5c2
# ╠═4059337b-d0c5-4948-94b6-c47c229625da
# ╠═15d6d724-e3ab-443c-99c6-3521976533d8
# ╠═46afcea3-a754-4731-81f9-e2a4c880d61d
