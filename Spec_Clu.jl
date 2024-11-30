### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 7c568c74-1c0f-4ab5-aaa6-e1271ea1db4a
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 3564c39a-1def-4613-bc81-87698bc1374a
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

# ╔═╡ a5400cdb-5e6b-4ab7-856e-10520556773b
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

# ╔═╡ 500d545d-63d8-4781-82d1-c76134ba5d7b
data_dir = joinpath(@__DIR__, "data")

# ╔═╡ 9e3318f6-ae50-4f2c-a9e1-c713e474ac2a
md"""
#### Reading all the the first .npy files in the agents A,B,C,D
"""

# ╔═╡ b9a1ea68-bab2-4dea-a5da-9adf297eb3d4
A_1_Path = joinpath(data_dir, "A", "A_1.npy")

# ╔═╡ 4ff135ab-4766-4171-966f-9d1ec0a9fd85
B_1_Path = joinpath(data_dir, "B", "B_1.npy")

# ╔═╡ d47e4774-0643-42bf-8f9e-90623734f928
C_1_Path = joinpath(data_dir, "C", "C_1.npy")

# ╔═╡ 82582279-cd77-4421-babb-00ec2878556e
D_1_Path = joinpath(data_dir, "D", "D_1.npy")

# ╔═╡ 8d492fd8-db84-48d4-b008-ec60aa9a4589
A_1 = permutedims(npzread(A_1_Path))

# ╔═╡ 01a6a4d7-6cfe-4f02-94f3-d97fe40fa513
B_1 = permutedims(npzread(B_1_Path))

# ╔═╡ 588cab78-bea3-4786-8061-eccfd0467fbf
C_1 = permutedims(npzread(C_1_Path))

# ╔═╡ 13928f14-d86f-4b02-b133-08b1f4f8d7c8
D_1 = permutedims(npzread(D_1_Path))

# ╔═╡ d60a1be3-6294-4458-bf7c-0f62587c9f5a
md"""
#### Data Matrix for all the the first .npy files in the agents A,B,C,D
"""

# ╔═╡ b86ec7f7-38f2-4c03-a2e2-4da88efcc306
D = hcat(A_1, B_1, C_1, D_1)

# ╔═╡ d28bfcc8-76d9-49ed-87fa-44a4aa4e4cdf
CACHEDIR = splitext(relpath(@__FILE__))[1]

# ╔═╡ 414bc403-ac65-436c-92a6-bb6186aea596
col_norms = [norm(D[:, i]) for i in 1:size(D, 2)]

# ╔═╡ f06fe3bb-5252-4faf-8844-180bb14d9ad8
Norm_vec = [D[:, i] ./ col_norms[i] for i in 1:size(D, 2)]

# ╔═╡ 4c1b921c-3e0d-44a2-bb16-b28fcc68d466
Norm_mat = hcat(Norm_vec...)

# ╔═╡ edb6bb20-9d90-41d7-8f6f-855963a82882
A = transpose(Norm_mat) * Norm_mat

# ╔═╡ 2752b767-4e48-4854-8a07-cd67dee00efa
S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

# ╔═╡ 434d7a9c-60aa-4714-ab7e-8737b6104b04
diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ 41f76e88-e6b6-4102-bce5-0ef8a1020781
D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ 8ff868a6-2529-43b1-bfa5-843718e34d1d
L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ 86e0e245-ec92-4c04-8d83-0e494157bac5
decomp, history = partialschur(L_sym; nev=4, which=:SR)

# ╔═╡ f2be6e06-bed0-465c-94e8-017bdc288fb6
V = mapslices(normalize, decomp.Q; dims=2)

# ╔═╡ 0d71ecff-86d4-4b4c-8250-4ff47c6dbb34
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

# ╔═╡ 80cc2bb5-d43d-41c1-94ed-9ae9eeeadc48
spec_clusterings = batchkmeans(permutedims(V), 4; maxiter=1000)

# ╔═╡ 31604328-6a70-4d6a-8d0d-dbc373365484
SC_Results = spec_clusterings[1].assignments

# ╔═╡ 053ab09c-bc43-4c44-ab17-d876d895c2fa
A1_Res = SC_Results[1:240]

# ╔═╡ 00ec7477-e1ca-4484-b317-e0fdfe98c1bd
B1_Res = SC_Results[241:513]

# ╔═╡ f78c1cd3-12ab-4665-b029-f596d9b383d1
C1_Res = SC_Results[514:795]

# ╔═╡ 08c39ca5-8f6a-44e3-acf5-61caf0c8a6be
D1_Res = SC_Results[796:1676]

# ╔═╡ 15c4a97c-22c0-439f-920b-39373002cded
A_label_count = [count(x -> (x==i), A1_Res) / length(A1_Res) * 100 for i in 1:4]

# ╔═╡ 52836235-973e-4c52-8571-c4aeb83a93ca
B_label_count = [count(x -> (x==i), B1_Res) / length(B1_Res) * 100 for i in 1:4]

# ╔═╡ 58b584e3-3117-45ea-b7f4-8bddbf8ad9ca
C_label_count = [count(x -> (x==i), C1_Res) / length(C1_Res) * 100 for i in 1:4]

# ╔═╡ 65e6dbf3-3914-492a-a755-6884ed9c05a6
D_label_count = [count(x -> (x==i), D1_Res) / length(D1_Res) * 100 for i in 1:4]

# ╔═╡ 45496921-7e0a-47e2-9cad-7eb631fba2ba
md"""
### Relabeled the clusters to plot the confusion matrix
"""

# ╔═╡ dba08ba0-efb4-461b-80e5-9da2b2afc8f8
A_relabel_map = Dict(1 => 1,
					2 => 2,
					3 => 3, 
					4 => 4)

# ╔═╡ 1aa21c70-6180-4b2a-b8a6-8e8c20e01cec
A_Results_relabel = [A_relabel_map[label] for label in A1_Res]

# ╔═╡ cf85b772-560a-400f-8fb7-51fd1a03957b
B_relabel_map = Dict(1 => 1,
					2 => 4,
					3 => 3, 
					4 => 2)

# ╔═╡ 8eec45b3-62f7-462f-9a79-81301c68b1af
B_Results_relabel = [B_relabel_map[label] for label in B1_Res]

# ╔═╡ 6ed2564b-5cef-4cc8-a4cf-102722b24e8a
C_relabel_map = Dict(1 => 1,
					2 => 3,
					3 => 2, 
					4 => 4)

# ╔═╡ 9e9d4fad-5c67-46d8-abea-8f7f564ba4b1
C_Results_relabel = [C_relabel_map[label] for label in C1_Res]

# ╔═╡ 7cac556a-3623-4ec0-a29c-b4748029d2da
D_relabel_map = Dict(1 => 1,
					2 => 2,
					3 => 4, 
					4 => 3)

# ╔═╡ 20ad2a2e-e993-48fe-b47f-d40632b383e9
D_Results_relabel = [D_relabel_map[label] for label in D1_Res]

# ╔═╡ 6a18b286-5711-484f-94df-40ab17441d77
A_relabel_count = [count(x -> (x==i), A_Results_relabel) / length(A_Results_relabel) * 100 for i in 1:4]

# ╔═╡ 817d0d5d-c68f-4865-b5ae-7bd8942cbf6f
B_relabel_count = [count(x -> (x==i), B_Results_relabel) / length(B_Results_relabel) * 100 for i in 1:4]

# ╔═╡ 88baf541-d100-4e0c-9d68-e8a55ffa48e7
C_relabel_count = [count(x -> (x==i), C_Results_relabel) / length(C_Results_relabel) * 100 for i in 1:4]

# ╔═╡ 3ab5368d-0270-4eac-9a4b-91a9d5d11c52
D_relabel_count = [count(x -> (x==i), D_Results_relabel) / length(D_Results_relabel) * 100 for i in 1:4]

# ╔═╡ 80dca74c-a648-4270-8d8c-562f698a82f3
md"""
### Plot the confusion matrix
"""

# ╔═╡ 06ce4d77-bc6a-400c-8997-4528682c2c24
Conf_mat = hcat(A_relabel_count, B_relabel_count, C_relabel_count, D_relabel_count)

# ╔═╡ 8a4c0f84-03bb-49a5-935b-f2f3bb52d003
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xticks=(1:4, ["A", "B", "C", "D"]), yticks=(1:4, ["A", "B", "C", "D"]), xlabel="True Agents", ylabel="Predicted Agents", title="Confusion Matrix")
	hm = heatmap!(ax, Conf_mat, colormap=:viridis)
	for j in 1:4, i in 1:4
		value = round(Conf_mat[i, j], digits=1)
		text!(ax, i-0.01, j - 0.10, text = "$value", color=:black, align = (:center, :center), fontsize=13)
	end
	
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ Cell order:
# ╟─a5400cdb-5e6b-4ab7-856e-10520556773b
# ╠═7c568c74-1c0f-4ab5-aaa6-e1271ea1db4a
# ╠═3564c39a-1def-4613-bc81-87698bc1374a
# ╠═500d545d-63d8-4781-82d1-c76134ba5d7b
# ╟─9e3318f6-ae50-4f2c-a9e1-c713e474ac2a
# ╠═b9a1ea68-bab2-4dea-a5da-9adf297eb3d4
# ╠═4ff135ab-4766-4171-966f-9d1ec0a9fd85
# ╠═d47e4774-0643-42bf-8f9e-90623734f928
# ╠═82582279-cd77-4421-babb-00ec2878556e
# ╠═8d492fd8-db84-48d4-b008-ec60aa9a4589
# ╠═01a6a4d7-6cfe-4f02-94f3-d97fe40fa513
# ╠═588cab78-bea3-4786-8061-eccfd0467fbf
# ╠═13928f14-d86f-4b02-b133-08b1f4f8d7c8
# ╟─d60a1be3-6294-4458-bf7c-0f62587c9f5a
# ╠═b86ec7f7-38f2-4c03-a2e2-4da88efcc306
# ╠═d28bfcc8-76d9-49ed-87fa-44a4aa4e4cdf
# ╠═414bc403-ac65-436c-92a6-bb6186aea596
# ╠═f06fe3bb-5252-4faf-8844-180bb14d9ad8
# ╠═4c1b921c-3e0d-44a2-bb16-b28fcc68d466
# ╠═edb6bb20-9d90-41d7-8f6f-855963a82882
# ╠═2752b767-4e48-4854-8a07-cd67dee00efa
# ╠═434d7a9c-60aa-4714-ab7e-8737b6104b04
# ╠═41f76e88-e6b6-4102-bce5-0ef8a1020781
# ╠═8ff868a6-2529-43b1-bfa5-843718e34d1d
# ╠═86e0e245-ec92-4c04-8d83-0e494157bac5
# ╠═f2be6e06-bed0-465c-94e8-017bdc288fb6
# ╠═0d71ecff-86d4-4b4c-8250-4ff47c6dbb34
# ╠═80cc2bb5-d43d-41c1-94ed-9ae9eeeadc48
# ╠═31604328-6a70-4d6a-8d0d-dbc373365484
# ╠═053ab09c-bc43-4c44-ab17-d876d895c2fa
# ╠═00ec7477-e1ca-4484-b317-e0fdfe98c1bd
# ╠═f78c1cd3-12ab-4665-b029-f596d9b383d1
# ╠═08c39ca5-8f6a-44e3-acf5-61caf0c8a6be
# ╠═15c4a97c-22c0-439f-920b-39373002cded
# ╠═52836235-973e-4c52-8571-c4aeb83a93ca
# ╠═58b584e3-3117-45ea-b7f4-8bddbf8ad9ca
# ╠═65e6dbf3-3914-492a-a755-6884ed9c05a6
# ╟─45496921-7e0a-47e2-9cad-7eb631fba2ba
# ╠═dba08ba0-efb4-461b-80e5-9da2b2afc8f8
# ╠═1aa21c70-6180-4b2a-b8a6-8e8c20e01cec
# ╠═cf85b772-560a-400f-8fb7-51fd1a03957b
# ╠═8eec45b3-62f7-462f-9a79-81301c68b1af
# ╠═6ed2564b-5cef-4cc8-a4cf-102722b24e8a
# ╠═9e9d4fad-5c67-46d8-abea-8f7f564ba4b1
# ╠═7cac556a-3623-4ec0-a29c-b4748029d2da
# ╠═20ad2a2e-e993-48fe-b47f-d40632b383e9
# ╠═6a18b286-5711-484f-94df-40ab17441d77
# ╠═817d0d5d-c68f-4865-b5ae-7bd8942cbf6f
# ╠═88baf541-d100-4e0c-9d68-e8a55ffa48e7
# ╠═3ab5368d-0270-4eac-9a4b-91a9d5d11c52
# ╟─80dca74c-a648-4270-8d8c-562f698a82f3
# ╠═06ce4d77-bc6a-400c-8997-4528682c2c24
# ╠═8a4c0f84-03bb-49a5-935b-f2f3bb52d003
