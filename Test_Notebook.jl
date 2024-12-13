### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ bff7151f-6dc8-4ee2-a07d-41c8392aa5de
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 7400d58a-5a45-4797-b22c-8cbec7182a6f
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

# ╔═╡ 9dabf654-9566-4d37-92fe-6ac668d940bc
data_dir = joinpath(@__DIR__, "data")

# ╔═╡ 57cf671f-8906-45ad-acd3-7df0dfce1529
md"""
#### Reading all the the first .npy files in the agents A,B,C,D
"""

# ╔═╡ 46efc977-f274-40fd-b8c2-77091345d0c6
A_1_Path = joinpath(data_dir, "A", "A_1.npy")

# ╔═╡ 239e5d85-a796-44cd-a756-4c4fd0c270b0
B_1_Path = joinpath(data_dir, "B", "B_1.npy")

# ╔═╡ 4ec79044-0c8e-486d-9d86-fca2e767c3ea
C_1_Path = joinpath(data_dir, "C", "C_1.npy")

# ╔═╡ 582b9e55-2d3b-4210-a672-1f9512637fec
D_1_Path = joinpath(data_dir, "D", "D_1.npy")

# ╔═╡ 0f3227e8-1adc-466c-bbd5-f1a244bd596a
Noise_path = joinpath(data_dir, "Noise", "Noise_1.npy")

# ╔═╡ a8f90076-83f8-4839-b3ed-f353b89be943
A_1 = permutedims(npzread(A_1_Path))

# ╔═╡ d8ee7bf6-1e77-4595-aebc-b24d1bdd5553
B_1 = permutedims(npzread(B_1_Path))

# ╔═╡ 21d18cd2-a998-444b-82d0-73c921cd13e0
C_1 = permutedims(npzread(C_1_Path))

# ╔═╡ 7f5d3189-c3a6-4c14-9889-9efcf0736d7a
D_1 = permutedims(npzread(D_1_Path))

# ╔═╡ c1fb119d-d39f-4f56-923d-a9135151a343
N = permutedims(npzread(Noise_path))

# ╔═╡ 95a9736c-a4af-4386-b8a0-67ee3c069766
md"""
#### Data Matrix for all the the first .npy files in the agents A,B,C,D and Noise
"""

# ╔═╡ d51c9771-f91a-4214-a76c-aeefdf0bdd01
D = hcat(A_1, B_1, C_1, D_1, N)

# ╔═╡ 8f796b29-8b19-45a4-8e22-a0e0100bae7d
begin
	
	
	# Initialize a figure
	fig = Figure(resolution = (900, 400))
	
	# Create a 1x3 grid layout
	ax1 = Axis(fig[1, 1], title = "A")
	ax2 = Axis(fig[1, 2], title = "B")
	ax3 = Axis(fig[1, 3], title = "Background")

	ax4 = Axis(fig[3, 1], title = "A")
	ax5 = Axis(fig[3, 2], title = "B")
	ax6 = Axis(fig[3, 3], title = "Background")
	vmin,vmax=0,3
	# Plot heatmaps in each axis
	hm1=heatmap!(ax1, A_1, colormap = :viridis, colorrange = (vmin, vmax))
	hm2=heatmap!(ax2, D_1, colormap = :viridis, colorrange = (vmin, vmax))
	hm3=heatmap!(ax3, N, colormap = :viridis, colorrange = (vmin, vmax))
	lines!(ax4,mean(A_1,dims=2)[:,1])
	lines!(ax5,mean(D_1,dims=2)[:,1])
	lines!(ax6,mean(N,dims=2)[:,1])

	# Add individual color bars for each heatmap
	Colorbar(fig[2, 1], hm1, label = "", vertical = false)
	Colorbar(fig[2, 2], hm2, label = "", vertical = false)
	Colorbar(fig[2, 3], hm3, label = "", vertical = false)
	
	# Adjust layout
	#fig.layoutgap[] = 10  # Optional: Adjust gaps between plots for clarity
	
	# Display the figure
	fig
end

# ╔═╡ 67f52670-8e97-4665-b5cc-3911837c4ef2
CACHEDIR = splitext(relpath(@__FILE__))[1]

# ╔═╡ 7ad80aa5-801a-49e5-9ec6-ec9f26dd1140
col_norms = [norm(D[:, i]) for i in 1:size(D, 2)]

# ╔═╡ 4dd70e78-aab2-4b15-ba0b-b67b859c958b
Norm_vec = [D[:, i] ./ col_norms[i] for i in 1:size(D, 2)]

# ╔═╡ f2d75638-c775-4051-ad26-d9a1d1b49063
Norm_mat = hcat(Norm_vec...)

# ╔═╡ 8d2cba0d-b8e8-4ef4-85ba-73809026b35d
A = transpose(Norm_mat) * Norm_mat

# ╔═╡ 42f03dc3-d9ca-4819-97f6-dff3b25dde35
S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

# ╔═╡ 2696302f-29fe-4f13-b9a4-592a6bd08dd7
diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ 6d982138-9ab9-4828-90c0-4e250260bf25
D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ b2c175f2-6f02-461f-8322-2f58f4de8186
L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ 05e56e52-ff6b-4ed6-a6e0-3bf6b486e9ed
decomp, history = partialschur(L_sym; nev=5, which=:SR)

# ╔═╡ 4d14844e-7bb6-4620-a5b9-69b005961069
V = mapslices(normalize, decomp.Q; dims=2)

# ╔═╡ f6b518e1-8779-4186-9de0-b0c21a9930bf
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

# ╔═╡ 159b178a-74c9-4410-ad28-0d3124febf99
spec_clusterings = batchkmeans(permutedims(V), 5; maxiter=1000)

# ╔═╡ 68dc77f5-8d72-4e87-8f45-95bd1d8756b9
SC_Results = spec_clusterings[1].assignments

# ╔═╡ ad58dfd6-d86e-40b8-9d34-ad0c0b448fe6
A1_Res = SC_Results[1:240]

# ╔═╡ 9c351134-a5a6-46c5-a858-7abb83320e71
B1_Res = SC_Results[241:513]

# ╔═╡ 4ad29a9e-0a4a-42a0-9a7b-81a8e703bcd7
C1_Res = SC_Results[514:795]

# ╔═╡ 1b502002-0600-44fd-bdaa-cabeb9ba5a29
D1_Res = SC_Results[796:1676]

# ╔═╡ 868e0b8b-93df-411c-9fd6-711099b88230
N_Res = SC_Results[1677:1970]

# ╔═╡ a23a9876-438a-4dad-95ff-8dfdd0d3ab20
A_label_count = [count(x -> (x==i), A1_Res) / length(A1_Res) * 100 for i in 1:5]

# ╔═╡ 03bf1a77-f0cb-47f7-a309-6833faa45da1
B_label_count = [count(x -> (x==i), B1_Res) / length(B1_Res) * 100 for i in 1:5]

# ╔═╡ 5bede177-4522-4f8e-817c-881375556d74
C_label_count = [count(x -> (x==i), C1_Res) / length(C1_Res) * 100 for i in 1:5]

# ╔═╡ d6ca5c4d-fe2c-47fc-8ea5-d1d9462bdc6c
D_label_count = [count(x -> (x==i), D1_Res) / length(D1_Res) * 100 for i in 1:5]

# ╔═╡ a1f77009-bf4b-4ba9-8ae3-c96c0f6368f6
N_label_count = [count(x -> (x==i), N_Res) / length(N_Res) * 100 for i in 1:5]

# ╔═╡ Cell order:
# ╠═bff7151f-6dc8-4ee2-a07d-41c8392aa5de
# ╠═7400d58a-5a45-4797-b22c-8cbec7182a6f
# ╠═9dabf654-9566-4d37-92fe-6ac668d940bc
# ╠═57cf671f-8906-45ad-acd3-7df0dfce1529
# ╠═46efc977-f274-40fd-b8c2-77091345d0c6
# ╠═239e5d85-a796-44cd-a756-4c4fd0c270b0
# ╠═4ec79044-0c8e-486d-9d86-fca2e767c3ea
# ╠═582b9e55-2d3b-4210-a672-1f9512637fec
# ╠═0f3227e8-1adc-466c-bbd5-f1a244bd596a
# ╠═a8f90076-83f8-4839-b3ed-f353b89be943
# ╠═d8ee7bf6-1e77-4595-aebc-b24d1bdd5553
# ╠═21d18cd2-a998-444b-82d0-73c921cd13e0
# ╠═7f5d3189-c3a6-4c14-9889-9efcf0736d7a
# ╠═c1fb119d-d39f-4f56-923d-a9135151a343
# ╠═95a9736c-a4af-4386-b8a0-67ee3c069766
# ╠═d51c9771-f91a-4214-a76c-aeefdf0bdd01
# ╠═8f796b29-8b19-45a4-8e22-a0e0100bae7d
# ╠═67f52670-8e97-4665-b5cc-3911837c4ef2
# ╠═7ad80aa5-801a-49e5-9ec6-ec9f26dd1140
# ╠═4dd70e78-aab2-4b15-ba0b-b67b859c958b
# ╠═f2d75638-c775-4051-ad26-d9a1d1b49063
# ╠═8d2cba0d-b8e8-4ef4-85ba-73809026b35d
# ╠═42f03dc3-d9ca-4819-97f6-dff3b25dde35
# ╠═2696302f-29fe-4f13-b9a4-592a6bd08dd7
# ╠═6d982138-9ab9-4828-90c0-4e250260bf25
# ╠═b2c175f2-6f02-461f-8322-2f58f4de8186
# ╠═05e56e52-ff6b-4ed6-a6e0-3bf6b486e9ed
# ╠═4d14844e-7bb6-4620-a5b9-69b005961069
# ╠═f6b518e1-8779-4186-9de0-b0c21a9930bf
# ╠═159b178a-74c9-4410-ad28-0d3124febf99
# ╠═68dc77f5-8d72-4e87-8f45-95bd1d8756b9
# ╠═ad58dfd6-d86e-40b8-9d34-ad0c0b448fe6
# ╠═9c351134-a5a6-46c5-a858-7abb83320e71
# ╠═4ad29a9e-0a4a-42a0-9a7b-81a8e703bcd7
# ╠═1b502002-0600-44fd-bdaa-cabeb9ba5a29
# ╠═868e0b8b-93df-411c-9fd6-711099b88230
# ╠═a23a9876-438a-4dad-95ff-8dfdd0d3ab20
# ╠═03bf1a77-f0cb-47f7-a309-6833faa45da1
# ╠═5bede177-4522-4f8e-817c-881375556d74
# ╠═d6ca5c4d-fe2c-47fc-8ea5-d1d9462bdc6c
# ╠═a1f77009-bf4b-4ba9-8ae3-c96c0f6368f6
