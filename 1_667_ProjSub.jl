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

# ╔═╡ 3d02e7c2-809b-4fea-b6dd-2b57e80bd9fd
dir = joinpath(@__DIR__, "data", "new_data")

# ╔═╡ 5cc1c590-7634-452e-9c9f-e3069eb84500
md"""
### Signals with different time steps
"""

# ╔═╡ 508ed756-4f36-409e-a19a-4977ad729c76
@bind Features Select(["size_2492", "size_3737", "size_7474", "size_14948", "size_74740"])

# ╔═╡ f11d6a5b-85f6-4595-8055-471aec249e8c
data_dir = joinpath(dir, "$Features")

# ╔═╡ 52cba473-073b-4b79-af5e-15cddba98aab
file_names = ["A.npy", "B.npy", "C.npy", "D.npy"]

# ╔═╡ e520e943-6231-48c1-81e3-046de58c60b3
file_paths = [joinpath(data_dir, file_name) for file_name in file_names]

# ╔═╡ 0192f614-0739-4a64-94eb-b168b9b27023
Data = [permutedims(npzread(path)) for path in file_paths]

# ╔═╡ 6acf8b8d-77dc-4da9-9fa1-32d31f2d17be
md"""
#### Data Matrix for all the .npy files from the agents A,B,C,D, and Noise
"""

# ╔═╡ 07cc7e63-5e5d-4937-b8e9-9f730d79a180
D_org = hcat(Data...)

# ╔═╡ 1b443b23-d3b5-48da-97ba-44f4ad498522
D = abs.(D_org.* (D_org .> 0))

# ╔═╡ d0c5ecaa-2f81-40af-8a36-9ce9732d1385
# CACHEDIR = splitext(relpath(@__FILE__))[1]

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

# ╔═╡ dcc8c1c8-14ed-4d01-8104-bcf11205b9cf
# V = mapslices(normalize, decomp.Q; dims=2)

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
A_label_count = [count(x -> (x==i), A1_Res) / length(A1_Res) * 100 for i in 1:n_clusters]

# ╔═╡ 3990a38b-c85a-4115-9c47-6cb39a29b6e3
B_label_count = [count(x -> (x==i), B1_Res) / length(B1_Res) * 100 for i in 1:n_clusters]

# ╔═╡ f041f163-5802-44b0-b417-f2f16575dfe0
C_label_count = [count(x -> (x==i), C1_Res) / length(C1_Res) * 100 for i in 1:n_clusters]

# ╔═╡ 84a51ff3-72ff-4636-ba57-14e7a6bb5958
D_label_count = [count(x -> (x==i), D1_Res) / length(D1_Res) * 100 for i in 1:n_clusters]

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
		U, c = cachet(joinpath(CACHEDIR, "$Features", "run-$idx.bson")) do
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

# ╔═╡ dedbc869-46b5-4bbe-9910-e2359e98fb52
KSS_Clustering = batch_KSS(D, fill(1, 5); niters=200, nruns=100)

# ╔═╡ Cell order:
# ╟─4b89c903-e5d9-4066-b5aa-e6ba8712e056
# ╟─57783977-c0f5-43b0-b43c-01d51e5a37cf
# ╠═1b187d2e-958f-42a5-a266-6f0dd1847a6b
# ╠═323391e7-2911-455e-813c-4f1b09aa02f6
# ╠═3d02e7c2-809b-4fea-b6dd-2b57e80bd9fd
# ╟─5cc1c590-7634-452e-9c9f-e3069eb84500
# ╠═508ed756-4f36-409e-a19a-4977ad729c76
# ╠═f11d6a5b-85f6-4595-8055-471aec249e8c
# ╠═52cba473-073b-4b79-af5e-15cddba98aab
# ╠═e520e943-6231-48c1-81e3-046de58c60b3
# ╠═0192f614-0739-4a64-94eb-b168b9b27023
# ╟─6acf8b8d-77dc-4da9-9fa1-32d31f2d17be
# ╠═07cc7e63-5e5d-4937-b8e9-9f730d79a180
# ╠═1b443b23-d3b5-48da-97ba-44f4ad498522
# ╠═d0c5ecaa-2f81-40af-8a36-9ce9732d1385
# ╟─f83c400c-7c6a-40ec-9784-a690506367de
# ╠═7efcff6a-630f-49aa-bfb9-64c242a5a526
# ╠═f6143724-4571-4b3d-86a4-6a9a0450bec9
# ╠═074bd8ec-b53a-46a6-9e64-ebb177ff4708
# ╠═18ebd983-9811-4652-9047-efb3d60b9722
# ╠═d448f5a8-7dd7-4db2-afff-3436e23b338f
# ╠═dcc8c1c8-14ed-4d01-8104-bcf11205b9cf
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
# ╠═dedbc869-46b5-4bbe-9910-e2359e98fb52
