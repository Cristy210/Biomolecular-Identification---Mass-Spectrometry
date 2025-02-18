### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 708a7911-63a7-428c-b1a2-553fdba5d062
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 84fd9933-e6a4-45c4-9978-67025be8ba5e
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering, Metal

# ╔═╡ fac27de6-ee21-11ef-2fdd-ad1c8eabd940
html"""<style>
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 94%;
    margin-left: 0%;
    margin-right: 4% !important;
}
"""

# ╔═╡ d0aa0ded-4984-40c9-85c7-237ddd7b0180
md"""
### Activate Project Directory
"""

# ╔═╡ 2183ef8e-c162-4a16-a313-8ffb66a1143f
begin
	dir = joinpath(@__DIR__, "data", "DataFiles", "Biological Data")
	file_names = ["A.npy", "B.npy", "C.npy", "D.npy"]
	file_paths = [joinpath(dir, file_name) for file_name in file_names]
end

# ╔═╡ 2e4b1b21-73e1-4f6c-b854-2e6f4eb3e429
md"
**Choose the preprocessing method in the drop-down menu below**
"

# ╔═╡ 3490485f-17fa-4c75-8325-b785df4674c1
@bind Preprocessing Select(["Original", "Absolute", "Squaring"])

# ╔═╡ 2e6b1b10-d3d3-470d-b4c9-3509f66c39a8
Data = [permutedims(npzread(path)) for path in file_paths]

# ╔═╡ 54a3ee32-c6b8-45b3-883a-088a45d8bdfc
md"""
**Data Matrix for all the .npy files from the agents A,B,C,D**
"""

# ╔═╡ d303de49-5f7c-4227-bc78-fad22062f599
D_Preprocessing = Dict("Original" => hcat(Data...), "Absolute" =>abs.(hcat(Data...)), "Squaring" => hcat(Data...).^2)

# ╔═╡ 6d5acb5c-c4a8-488e-8cfa-3ef1e1b90312
D = D_Preprocessing[Preprocessing]

# ╔═╡ d682d7fb-60ea-416d-96f4-77bd5ac67652
gpu_data = MtlMatrix{Float32}(D)

# ╔═╡ ed972bda-039e-4d6f-8224-203082a37127
gpu_d_t = MtlMatrix(transpose(gpu_data))

# ╔═╡ f230b889-7815-4b02-9c58-5120f16a5eba
md"""
### Spectral Clustering
"""

# ╔═╡ 3f97063e-2a02-4375-aacd-45eb6457dbd3
gpu_data * gpu_d_t

# ╔═╡ b2ad69ec-a66b-4c34-990f-82c37952d190
begin 
	col_norms = [norm(gpu_data[:, i]) for i in 1:size(gpu_data, 2)]
	Norm_vec = [gpu_data[:, i] ./ col_norms[i] for i in 1:size(gpu_data, 2)]
	Norm_mat = hcat(Norm_vec...);
	n_clusters = 4;
	A = transpose(Norm_mat) * Norm_mat
end

# ╔═╡ 0ce50623-9892-4715-8457-2f6c9115a773
S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

# ╔═╡ 8d929a6a-1ad4-4958-960e-b18959fbcb9a
diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ 0621d031-0379-4197-aa6d-16a33ddabea8
function embedding(A, k)
	
	S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

	# Compute node degrees and form Laplacian
	diag_mat = Diagonal(vec(sum(S, dims=2)))
	D_sqrinv = sqrt(inv(diag_mat))
	L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

	# Compute eigenvectors
	decomp, history = partialschur(L_sym; nev=k, which=:SR)
	@info history
	return mapslices(normalize, decomp.Q; dims=2)
end

# ╔═╡ 072edacd-e76c-4955-8071-32505d6a0103
V = embedding(A, n_clusters)

# ╔═╡ Cell order:
# ╟─fac27de6-ee21-11ef-2fdd-ad1c8eabd940
# ╟─d0aa0ded-4984-40c9-85c7-237ddd7b0180
# ╠═708a7911-63a7-428c-b1a2-553fdba5d062
# ╠═84fd9933-e6a4-45c4-9978-67025be8ba5e
# ╠═2183ef8e-c162-4a16-a313-8ffb66a1143f
# ╟─2e4b1b21-73e1-4f6c-b854-2e6f4eb3e429
# ╠═3490485f-17fa-4c75-8325-b785df4674c1
# ╠═2e6b1b10-d3d3-470d-b4c9-3509f66c39a8
# ╟─54a3ee32-c6b8-45b3-883a-088a45d8bdfc
# ╠═d303de49-5f7c-4227-bc78-fad22062f599
# ╠═6d5acb5c-c4a8-488e-8cfa-3ef1e1b90312
# ╠═d682d7fb-60ea-416d-96f4-77bd5ac67652
# ╠═ed972bda-039e-4d6f-8224-203082a37127
# ╟─f230b889-7815-4b02-9c58-5120f16a5eba
# ╠═3f97063e-2a02-4375-aacd-45eb6457dbd3
# ╠═b2ad69ec-a66b-4c34-990f-82c37952d190
# ╠═0ce50623-9892-4715-8457-2f6c9115a773
# ╠═8d929a6a-1ad4-4958-960e-b18959fbcb9a
# ╠═0621d031-0379-4197-aa6d-16a33ddabea8
# ╠═072edacd-e76c-4955-8071-32505d6a0103
