### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ fa0106eb-8113-4893-931b-2aa653fa1b9e
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 11f3b1a8-127a-47ca-9d46-c3199a34275e
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

# ╔═╡ 9ddb97c4-7336-41e3-ab9d-3b0b3da3bda2
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

# ╔═╡ e4d0245e-4c3b-4e71-b705-940a6f1459e5
data_dir = joinpath(@__DIR__, "data", "K-Subspaces")

# ╔═╡ 6e67c7e6-aae4-40f3-9ad9-bd909e0062f4
file_names = ["A.npy", "B.npy", "C.npy", "D.npy", "Noise.npy"]

# ╔═╡ 353a6a00-0ecd-4e02-ba39-4b430af9c6dc
file_paths = [joinpath(data_dir, file_name) for file_name in file_names]

# ╔═╡ 6043c32e-96b3-40b9-a5f9-687bf2ea486b
Data = [permutedims(npzread(path)) for path in file_paths]

# ╔═╡ 2ebae7f8-3324-449b-af3c-3885bdeabd7b
md"""
#### Data Matrix for all the .npy files from the agents A,B,C,D, and Noise
"""

# ╔═╡ bcfed245-106a-41c8-8a2e-137cd71d48b7
D = hcat(Data...)

# ╔═╡ 596dfd07-92c8-4c7a-a47b-4a694533d66c
D_pos = abs.(D)

# ╔═╡ 06651850-0b40-46f4-8b50-c5c7d18a4a57
col_norms = [norm(D_pos[:, i]) for i in 1:size(D_pos, 2)]

# ╔═╡ 90898f21-da41-4d25-81b8-a5218cb19759
Norm_vec = [D_pos[:, i] ./ col_norms[i] for i in 1:size(D_pos, 2)]

# ╔═╡ 50891e05-2f8d-434d-8f8e-21c7a1a6454d
Norm_mat = hcat(Norm_vec...)

# ╔═╡ 3c546422-1670-4ffb-80d6-34d0d3fe9b9d
A = transpose(Norm_mat) * Norm_mat

# ╔═╡ 8b5c82c5-9a1a-4f0c-8c07-6fbc1ab861fe
S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

# ╔═╡ b8d8af7d-2af1-4e3f-b7fc-ea9e475f27ff
diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ a1ff46e9-c3eb-4d2d-bec7-62910de91f55
D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ 2af8c30b-b9e3-4860-ac19-ffcdb84c445f
L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ 75ae6322-6716-4719-b9bc-04a0811a279c
n_clusters = 5

# ╔═╡ 236ccf42-d337-47e5-8066-cc69495945ef
decomp, history = partialschur(L_sym; nev=n_clusters, which=:SR)

# ╔═╡ 8fb8ac68-a2f3-49b3-8376-2c2ac406029c
V = mapslices(normalize, decomp.Q; dims=2)

# ╔═╡ fa6628d9-80c9-4189-b655-fdc4cf3b5b25
function batchkmeans(X, k, args...; nruns=100, kwargs...)
	runs = @withprogress map(1:nruns) do idx
		# Run K-means
		Random.seed!(idx)  # set seed for reproducibility
		result = with_logger(NullLogger()) do
			kmeans(X, k, args...; kwargs...)
		end

		# Log progress and return result
		@logprogress idx/nruns
		return result
	end

	# Print how many converged
	nconverged = count(run -> run.converged, runs)
	@info "$nconverged/$nruns runs converged"

	# Return runs sorted best to worst
	return sort(runs; by=run->run.totalcost)
end

# ╔═╡ 1e86cacd-e424-4709-9cf3-3bf1a35f9c38
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 4f5b0d31-5bb1-445e-864d-c4392560af35
EKSS_Results = spec_clusterings[1].assignments

# ╔═╡ f972177c-cee0-4ba2-9ab5-1933cce3587d
A_EKSS = EKSS_Results[1:500]

# ╔═╡ 221d472f-e0dd-4d28-92fa-effd0a465f01
B_EKSS = EKSS_Results[501:1000]

# ╔═╡ 421b6805-d5b5-4d2e-839e-f2c5ad4f11ae
C_EKSS = EKSS_Results[1001:1500]

# ╔═╡ 34a49d7c-0779-401f-863b-8552f9a5266c
D_EKSS = EKSS_Results[1501:2000]

# ╔═╡ 3561b2fd-80d6-48be-9611-251fefad2ae0
N_EKSS = EKSS_Results[2001:2500]

# ╔═╡ 72ed2bbf-7fcb-48a4-bcdb-0d191f5fb862
A_label_count_EKSS = [count(x -> (x==i), A_EKSS) / length(A_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ 16ae6f49-25e7-4a75-a773-a3ecfa5b2629
B_label_count_EKSS = [count(x -> (x==i), B_EKSS) / length(B_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ 7d1a34fc-b4d5-4737-b43f-44f188ea70ca
C_label_count_EKSS = [count(x -> (x==i), C_EKSS) / length(C_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ f996ab44-07e5-4b48-9284-6227e750f33c
D_label_count_EKSS = [count(x -> (x==i), D_EKSS) / length(D_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ 9741bc70-bd73-4fe0-9956-b63f63720639
N_label_count_EKSS = [count(x -> (x==i), N_EKSS) / length(N_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ Cell order:
# ╠═9ddb97c4-7336-41e3-ab9d-3b0b3da3bda2
# ╠═fa0106eb-8113-4893-931b-2aa653fa1b9e
# ╠═11f3b1a8-127a-47ca-9d46-c3199a34275e
# ╠═e4d0245e-4c3b-4e71-b705-940a6f1459e5
# ╠═6e67c7e6-aae4-40f3-9ad9-bd909e0062f4
# ╠═353a6a00-0ecd-4e02-ba39-4b430af9c6dc
# ╠═6043c32e-96b3-40b9-a5f9-687bf2ea486b
# ╟─2ebae7f8-3324-449b-af3c-3885bdeabd7b
# ╠═bcfed245-106a-41c8-8a2e-137cd71d48b7
# ╠═596dfd07-92c8-4c7a-a47b-4a694533d66c
# ╠═06651850-0b40-46f4-8b50-c5c7d18a4a57
# ╠═90898f21-da41-4d25-81b8-a5218cb19759
# ╠═50891e05-2f8d-434d-8f8e-21c7a1a6454d
# ╠═3c546422-1670-4ffb-80d6-34d0d3fe9b9d
# ╠═8b5c82c5-9a1a-4f0c-8c07-6fbc1ab861fe
# ╠═b8d8af7d-2af1-4e3f-b7fc-ea9e475f27ff
# ╠═a1ff46e9-c3eb-4d2d-bec7-62910de91f55
# ╠═2af8c30b-b9e3-4860-ac19-ffcdb84c445f
# ╠═75ae6322-6716-4719-b9bc-04a0811a279c
# ╠═236ccf42-d337-47e5-8066-cc69495945ef
# ╠═8fb8ac68-a2f3-49b3-8376-2c2ac406029c
# ╠═fa6628d9-80c9-4189-b655-fdc4cf3b5b25
# ╠═1e86cacd-e424-4709-9cf3-3bf1a35f9c38
# ╠═4f5b0d31-5bb1-445e-864d-c4392560af35
# ╠═f972177c-cee0-4ba2-9ab5-1933cce3587d
# ╠═221d472f-e0dd-4d28-92fa-effd0a465f01
# ╠═421b6805-d5b5-4d2e-839e-f2c5ad4f11ae
# ╠═34a49d7c-0779-401f-863b-8552f9a5266c
# ╠═3561b2fd-80d6-48be-9611-251fefad2ae0
# ╠═72ed2bbf-7fcb-48a4-bcdb-0d191f5fb862
# ╠═16ae6f49-25e7-4a75-a773-a3ecfa5b2629
# ╠═7d1a34fc-b4d5-4737-b43f-44f188ea70ca
# ╠═f996ab44-07e5-4b48-9284-6227e750f33c
# ╠═9741bc70-bd73-4fe0-9956-b63f63720639
