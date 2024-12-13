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

# ╔═╡ 1b187d2e-958f-42a5-a266-6f0dd1847a6b
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 323391e7-2911-455e-813c-4f1b09aa02f6
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

# ╔═╡ 4b89c903-e5d9-4066-b5aa-e6ba8712e056
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

# ╔═╡ 57783977-c0f5-43b0-b43c-01d51e5a37cf
md"""
### Activate Project Directory
"""

# ╔═╡ 2fde9133-c6c0-4b13-bb76-b7dcf3f91c59
begin
	dir = joinpath(@__DIR__, "DataFiles", "Data")
	file_names = ["A.npy", "B.npy", "C.npy", "D.npy"]
	file_paths = [joinpath(dir, file_name) for file_name in file_names]
end

# ╔═╡ 3d02e7c2-809b-4fea-b6dd-2b57e80bd9fd
@bind Preprocessing Select(["Original", "Absolute", "Squaring"])

# ╔═╡ 0192f614-0739-4a64-94eb-b168b9b27023
Data = [permutedims(npzread(path)) for path in file_paths]

# ╔═╡ d07d2a4f-95e3-4cbe-92cd-8eb785ae08ba
md"""
### Data Visualizations
"""

# ╔═╡ 6acf8b8d-77dc-4da9-9fa1-32d31f2d17be
md"""
#### Data Matrix for all the .npy files from the agents A,B,C,D
"""

# ╔═╡ a33479ce-23be-450f-86e3-943a895116fe
D_Preprocessing = Dict("Original" => hcat(Data...), "Absolute" =>abs.(hcat(Data...)), "Squaring" => hcat(Data...).^2)

# ╔═╡ 783f183b-f14a-44d5-a22f-dd53b0667f2e
D = D_Preprocessing[Preprocessing]

# ╔═╡ c3038511-149c-4235-b95a-d157ecfb4394
with_theme() do
	fig = Figure(; size=(1300, 700))

	#Setting up the Gridlayouts one for each Agent
	grid1 = GridLayout(fig[1, 1]; nrow=3, ncol=1)
    grid2 = GridLayout(fig[1, 2]; nrow=3, ncol=1)
	grid3 = GridLayout(fig[1, 3]; nrow=3, ncol=1)
	grid4 = GridLayout(fig[1, 4]; nrow=3, ncol=1)

	vmin,vmax=0,3

	#Setting up the axis for heatmaps
	ax11 = Axis(grid1[1, 1], title = "Protein A")
	ax21 = Axis(grid2[1, 1], title = "Protein B")
	ax31 = Axis(grid3[1, 1], title = "Bacteria C")
	ax41 = Axis(grid4[1, 1], title = "Bacteria D")

	#Setting up the axis for plotting the peaks
	ax13 = Axis(grid1[3, 1])
	ax23 = Axis(grid2[3, 1])
	ax33 = Axis(grid3[3, 1])
	ax43 = Axis(grid4[3, 1])


	#Plotting the heatmaps for Agents A, B, C, and D
	hm1=heatmap!(ax11, D[:, 1:500], colormap = :viridis, colorrange = (vmin, vmax))
	hm2=heatmap!(ax21, D[:, 501:1000], colormap = :viridis, colorrange = (vmin, vmax))
	hm3=heatmap!(ax31, D[:, 1001:1500], colormap = :viridis, colorrange = (vmin, vmax))
	hm4=heatmap!(ax41, D[:, 1501:2000], colormap = :viridis, colorrange = (vmin, vmax))

	lines!(ax13,mean(D[:, 1:500],dims=2)[:,1])
	lines!(ax23,mean(D[:, 501:1000],dims=2)[:,1])
	lines!(ax33,mean(D[:, 1001:1500],dims=2)[:,1])
	lines!(ax43,mean(D[:, 1501:2000],dims=2)[:,1])

	Colorbar(grid1[2, 1], hm1, label = "", vertical = false)
	Colorbar(grid2[2, 1], hm2, label = "", vertical = false)
	Colorbar(grid3[2, 1], hm3, label = "", vertical = false)
	Colorbar(grid4[2, 1], hm4, label = "", vertical = false)

	fig
end

# ╔═╡ 65320198-78e1-442d-9ca3-5029393e20c2
with_theme() do
	fig = Figure(; size = (950,400))

	supertitle = Label(fig[0, 1:4], "Log Singular Values", fontsize=20, halign=:center, valign=:top, )

	#Setting up the Gridlayouts one for each Agent
	grid1 = GridLayout(fig[1, 1]; nrow=1, ncol=1)
    grid2 = GridLayout(fig[1, 2]; nrow=1, ncol=1)
	grid3 = GridLayout(fig[1, 3]; nrow=1, ncol=1)
	grid4 = GridLayout(fig[1, 4]; nrow=1, ncol=1)
	
	# Compute the singular values
	singular_values1 = svd(D[:, 1:500]').S
	singular_values2 = svd(D[:, 501:1000]').S
	singular_values3 = svd(D[:, 1001:1500]').S
	singular_values4 = svd(D[:, 1501:2000]').S
	
	# Compute the log of the singular values
	log_singular_values1 = log10.(singular_values1)
	log_singular_values2 = log10.(singular_values2)
	log_singular_values3 = log10.(singular_values3)
	log_singular_values3 = log10.(singular_values4)
	
	# Initialize a figure with size
	
	
	# Create axes for each subplot
	ax1 = Axis(grid1[1, 1], title = "Protein A", xlabel = "Index", ylabel = "Log10(Singular Value)")
	ax2 = Axis(grid2[1, 1], title = "Protein B", xlabel = "Index")
	ax3 = Axis(grid3[1, 1], title = "Bacteria C", xlabel = "Index")
	ax4 = Axis(grid4[1, 1], title = "Bacteria D", xlabel = "Index")
	
	# Plot log singular values on each axis
	scatter!(ax1, 1:length(log_singular_values1), log_singular_values1, color = :blue)
	scatter!(ax2, 1:length(log_singular_values2), log_singular_values2, color = :red)
	scatter!(ax3, 1:length(log_singular_values3), log_singular_values3, color = :green)
	scatter!(ax4, 1:length(log_singular_values3), log_singular_values3, color = :black)
	
	# Display the figure
	fig
end

# ╔═╡ f83c400c-7c6a-40ec-9784-a690506367de
md"""
### Spectral Clustering
"""

# ╔═╡ 7efcff6a-630f-49aa-bfb9-64c242a5a526
col_norms = [norm(D[:, i]) for i in 1:size(D, 2)]

# ╔═╡ f6143724-4571-4b3d-86a4-6a9a0450bec9
Norm_vec = [D[:, i] ./ col_norms[i] for i in 1:size(D, 2)]

# ╔═╡ 074bd8ec-b53a-46a6-9e64-ebb177ff4708
Norm_mat = hcat(Norm_vec...);

# ╔═╡ 18ebd983-9811-4652-9047-efb3d60b9722
A = transpose(Norm_mat) * Norm_mat

# ╔═╡ d448f5a8-7dd7-4db2-afff-3436e23b338f
n_clusters = 4

# ╔═╡ e0bec9ff-0442-4134-bf4e-636ec9bf6eeb
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

# ╔═╡ 3ffbeab0-c815-43f1-97d4-441969bea6b1
V = embedding(A, n_clusters)

# ╔═╡ cf041b97-409f-4c30-96bb-40ed0f1ec441
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

# ╔═╡ d1e78fd8-3403-40b2-9acf-43204caf5bf1
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 1192b8a0-7d4f-45a2-a506-70846ae0d206
SC_Results = spec_clusterings[1].assignments

# ╔═╡ 73139f21-4785-4948-97d1-4d4db9272b2f
A1_Res = SC_Results[1:500]

# ╔═╡ 05984c1c-2c35-4500-b99f-0ab627809198
B1_Res = SC_Results[501:1000]

# ╔═╡ c51136ce-9120-404c-8bc3-89ea250d1e62
C1_Res = SC_Results[1001:1500]

# ╔═╡ b89a1c6f-383d-4715-bd75-e6ae4da27623
D1_Res = SC_Results[1501:2000]

# ╔═╡ cb92b31d-f44d-4c6f-b181-4fa1cbdd1c52
A_label_count_SC = [count(x -> (x==i), A1_Res) / length(A1_Res) * 100 for i in 1:n_clusters]

# ╔═╡ 3990a38b-c85a-4115-9c47-6cb39a29b6e3
B_label_count_SC = [count(x -> (x==i), B1_Res) / length(B1_Res) * 100 for i in 1:n_clusters]

# ╔═╡ f041f163-5802-44b0-b417-f2f16575dfe0
C_label_count_SC = [count(x -> (x==i), C1_Res) / length(C1_Res) * 100 for i in 1:n_clusters]

# ╔═╡ 84a51ff3-72ff-4636-ba57-14e7a6bb5958
D_label_count_SC = [count(x -> (x==i), D1_Res) / length(D1_Res) * 100 for i in 1:n_clusters]

# ╔═╡ e17c9163-7b1d-4672-9846-b526c9f24343
md"""
### K-Subspaces
"""

# ╔═╡ 0dfa6d32-6689-4c05-8c25-9a2f6c932f63
function polar(X)
	U, _, V = svd(X)
	U*V'
end

# ╔═╡ 5843384d-ee11-4a04-9c52-a33efc527783
"""
	KSS(X, d; niters=100)

Run K-subspaces on the data matrix `X`
with subspace dimensions `d[1], ..., d[K]`,
treating the columns of `X` as the datapoints.
"""
function KSS(X, d; niters=100, Uinit=polar.(randn.(size(X, 1), collect(d))))
	K = length(d)
	D, N = size(X)

	# Initialize
	U = deepcopy(Uinit)
	c = [argmax(norm(U[k]' * view(X, :, i)) for k in 1:K) for i in 1:N]
	c_prev = copy(c)

	# Iterations
	@progress for t in 1:niters
		# Update subspaces
		for k in 1:K
			ilist = findall(==(k), c)
			# println("Cluster $k, ilist size: ", length(ilist))
			if isempty(ilist)
				# println("Initializing $k subspace")
				U[k] = polar(randn(D, d[k]))
			else
				A = view(X, :, ilist) * transpose(view(X, :, ilist))
				# decomp, history = partialschur(A; nev=d[k], which=:LR)
				# @show history
				# U[k] = tsvd(view(X, :, ilist), d[k])[1]
				U[k] = svd(view(X, :, ilist)).U[:,1:d[k]]
			end
		end

		# Update clusters
		for i in 1:N
			c[i] = argmax(norm(U[k]' * view(X, :, i)) for k in 1:K)
		end

		# Break if clusters did not change, update otherwise
		if c == c_prev
			@info "Terminated early at iteration $t"
			break
		end
		c_prev .= c
	end

	return U, c
end

# ╔═╡ a8c2cb19-febb-40ea-9766-d190c4ccda74
CACHEDIR = splitext(relpath(@__FILE__))[1]

# ╔═╡ 16a77379-f74d-4e7f-b175-8e9ed2a02965
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 077b8ae4-79ed-4806-8a78-63c58f07c882
function batch_KSS(X, d; niters=100, nruns=10)
	D, N = size(X)
	runs = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Int}, Float64}}(undef, nruns)
	@progress for idx in 1:nruns
		U, c = cachet(joinpath(CACHEDIR, "EKSS", "$Preprocessing", "run-$idx.bson")) do
			Random.seed!(idx)
			KSS(X, d; niters=niters)
		end

		total_cost = 0
		for i in 1:N
			cost = norm(U[c[i]]' * view(X, :, i))
			total_cost += cost
		end

		runs[idx] = (U, c, total_cost)

		
	end

	 return runs
end

# ╔═╡ 7a2b7ea0-3ae6-4c3d-b1ba-e4c1bd3828b3
KSS_runs = 300

# ╔═╡ 5c31fed2-e41a-4444-b886-22da7fecd7fc
KSS_Clustering = batch_KSS(D, fill(2, 4); niters=200, nruns=KSS_runs)

# ╔═╡ 689d4ad3-31a2-425c-b6fe-69983797fac7
min_idx_KSS = argmax(KSS_Clustering[i][3] for i in 1:KSS_runs)

# ╔═╡ 5f6480e8-36a5-4a6b-9aa3-4a5cddf0b6e8
KSS_Results = KSS_Clustering[min_idx_KSS][2]

# ╔═╡ 30918205-73f4-44a2-8f7d-6457378c2996
md"""
### Ensemble Clustering
"""

# ╔═╡ c2b1858d-c0a8-4ed7-8149-1e0634c1901e
cluster_labels = [KSS_Clustering[i][2] for i in 1:KSS_runs]

# ╔═╡ 73df4b78-a703-44cd-875c-536538225070
md"""
#### Co-Clustering Matrix
"""

# ╔═╡ 50367ac4-71d5-405a-b2c9-ebd9717e920f
N_points = length(cluster_labels[1])

# ╔═╡ 29af46ef-91fc-4fb9-8fa4-0cc02b311b30
length(cluster_labels)

# ╔═╡ 196a6ad0-13a3-4678-8f5e-9212d8be46a3
Co_Clustering = begin
	Aff = zeros(Float64, N_points, N_points)
	for labels in cluster_labels
		for i in 1:N_points, j in 1:N_points
			if labels[i] == labels[j]
				Aff[i, j] += 1
			end
		end
	end
	Aff ./ length(cluster_labels)
end

# ╔═╡ 8e653298-c048-4f90-8954-58e380f6fc2d
V_Ensemble = embedding(Co_Clustering, n_clusters)

# ╔═╡ 07999787-dfcb-41b6-9125-201d5b307923
EKSS_clusterings = batchkmeans(permutedims(V_Ensemble), n_clusters; maxiter=1000)

# ╔═╡ 6bf31f57-d627-42cc-af6a-d4088e87edfe
EKSS_Results = EKSS_clusterings[1].assignments

# ╔═╡ Cell order:
# ╟─4b89c903-e5d9-4066-b5aa-e6ba8712e056
# ╟─57783977-c0f5-43b0-b43c-01d51e5a37cf
# ╠═1b187d2e-958f-42a5-a266-6f0dd1847a6b
# ╠═323391e7-2911-455e-813c-4f1b09aa02f6
# ╠═2fde9133-c6c0-4b13-bb76-b7dcf3f91c59
# ╠═3d02e7c2-809b-4fea-b6dd-2b57e80bd9fd
# ╠═0192f614-0739-4a64-94eb-b168b9b27023
# ╟─d07d2a4f-95e3-4cbe-92cd-8eb785ae08ba
# ╟─6acf8b8d-77dc-4da9-9fa1-32d31f2d17be
# ╠═a33479ce-23be-450f-86e3-943a895116fe
# ╠═783f183b-f14a-44d5-a22f-dd53b0667f2e
# ╟─c3038511-149c-4235-b95a-d157ecfb4394
# ╟─65320198-78e1-442d-9ca3-5029393e20c2
# ╟─f83c400c-7c6a-40ec-9784-a690506367de
# ╠═7efcff6a-630f-49aa-bfb9-64c242a5a526
# ╠═f6143724-4571-4b3d-86a4-6a9a0450bec9
# ╠═074bd8ec-b53a-46a6-9e64-ebb177ff4708
# ╠═18ebd983-9811-4652-9047-efb3d60b9722
# ╠═d448f5a8-7dd7-4db2-afff-3436e23b338f
# ╠═e0bec9ff-0442-4134-bf4e-636ec9bf6eeb
# ╠═3ffbeab0-c815-43f1-97d4-441969bea6b1
# ╠═cf041b97-409f-4c30-96bb-40ed0f1ec441
# ╠═d1e78fd8-3403-40b2-9acf-43204caf5bf1
# ╠═1192b8a0-7d4f-45a2-a506-70846ae0d206
# ╠═73139f21-4785-4948-97d1-4d4db9272b2f
# ╠═05984c1c-2c35-4500-b99f-0ab627809198
# ╠═c51136ce-9120-404c-8bc3-89ea250d1e62
# ╠═b89a1c6f-383d-4715-bd75-e6ae4da27623
# ╠═cb92b31d-f44d-4c6f-b181-4fa1cbdd1c52
# ╠═3990a38b-c85a-4115-9c47-6cb39a29b6e3
# ╠═f041f163-5802-44b0-b417-f2f16575dfe0
# ╠═84a51ff3-72ff-4636-ba57-14e7a6bb5958
# ╟─e17c9163-7b1d-4672-9846-b526c9f24343
# ╠═0dfa6d32-6689-4c05-8c25-9a2f6c932f63
# ╠═5843384d-ee11-4a04-9c52-a33efc527783
# ╠═a8c2cb19-febb-40ea-9766-d190c4ccda74
# ╠═077b8ae4-79ed-4806-8a78-63c58f07c882
# ╠═16a77379-f74d-4e7f-b175-8e9ed2a02965
# ╠═7a2b7ea0-3ae6-4c3d-b1ba-e4c1bd3828b3
# ╠═5c31fed2-e41a-4444-b886-22da7fecd7fc
# ╠═689d4ad3-31a2-425c-b6fe-69983797fac7
# ╠═5f6480e8-36a5-4a6b-9aa3-4a5cddf0b6e8
# ╟─30918205-73f4-44a2-8f7d-6457378c2996
# ╠═c2b1858d-c0a8-4ed7-8149-1e0634c1901e
# ╟─73df4b78-a703-44cd-875c-536538225070
# ╠═50367ac4-71d5-405a-b2c9-ebd9717e920f
# ╠═29af46ef-91fc-4fb9-8fa4-0cc02b311b30
# ╠═196a6ad0-13a3-4678-8f5e-9212d8be46a3
# ╠═8e653298-c048-4f90-8954-58e380f6fc2d
# ╠═07999787-dfcb-41b6-9125-201d5b307923
# ╠═6bf31f57-d627-42cc-af6a-d4088e87edfe
