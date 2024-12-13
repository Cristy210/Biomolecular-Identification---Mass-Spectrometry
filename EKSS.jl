### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 4059337b-d0c5-4948-94b6-c47c229625da
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 15d6d724-e3ab-443c-99c6-3521976533d8
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

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
data_dir = joinpath(@__DIR__, "data", "K-Subspaces")

# ╔═╡ 0ef5270a-6b13-44dc-a0ca-445cbac900e6
file_names = ["A.npy", "B.npy", "C.npy", "D.npy", "Noise.npy"]

# ╔═╡ 66771350-3cd8-4891-b77a-6f5ec7f328e0
file_paths = [joinpath(data_dir, file_name) for file_name in file_names]

# ╔═╡ 8d94d2c8-79e9-442e-a3b7-46703a05d16d
Data = [permutedims(npzread(path)) for path in file_paths]

# ╔═╡ 8694e758-e5c0-42c4-be53-bb6a7955fbe5
md"""
#### Data Matrix for all the .npy files from the agents A,B,C,D, and Noise
"""

# ╔═╡ bba58c1c-8a6f-4041-b294-2a0af2954f96
D = hcat(Data...)

# ╔═╡ 1a36bc2e-8f5b-4ba6-aecb-1f35955827b9
# positive_values = [row[row .> 0] for row in eachcol(D)]

# ╔═╡ f86a0699-396a-4f60-b940-d6af17526b68
D_pos = abs.(D .* (D .> 0))

# ╔═╡ 8f31b045-c9c1-44c7-96c1-3433c2188206
md"""
### K-Subspaces
"""

# ╔═╡ f6d0032b-00e9-4f2e-81b5-fd3646ea19cf
function polar(X)
	U, _, V = svd(X)
	U*V'
end

# ╔═╡ a268bde4-37b4-4729-b7f6-4ab339f1c2f4
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
				decomp, history = partialschur(A; nev=d[k], which=:LR)
				@show history
				# U[k] = tsvd(view(X, :, ilist), d[k])[1]
				U[k] = decomp.Q
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

# ╔═╡ 169eb774-2299-4a54-9d68-1b1f4ba3dab0
CACHEDIR = splitext(relpath(@__FILE__))[1]

# ╔═╡ a43d05d7-fd37-42c8-bc0c-d32a019b5f8e
function batch_KSS(X, d; niters=100, nruns=10)
	D, N = size(X)
	runs = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Int}, Float64}}(undef, nruns)
	@progress for idx in 1:nruns
		U, c = cachet(joinpath(CACHEDIR, "run-$idx.bson")) do
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

# ╔═╡ f91fc2c3-7cca-4eec-b91e-c3cbe2208c62
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end
<<<<<<< HEAD

# ╔═╡ 0ab76807-7edd-4585-a225-61ee3e439482
KSS_Clustering = batch_KSS(D, fill(1, 5); niters=200, nruns=271)
=======

# ╔═╡ 15e04665-1547-4ecb-87d1-9fae06c2639c
fill(2, 5)

# ╔═╡ 0ab76807-7edd-4585-a225-61ee3e439482

# KSS_Clustering = batch_KSS(D, fill(2, 5); niters=200, nruns=200)
KSS_Clustering = batch_KSS(D, fill(2, 5); niters=200, nruns=100)
>>>>>>> 949be196456e9a39d3623dfb9736ed3cb24d9604

# ╔═╡ 4fd0a2f9-93b7-4709-afb7-90e6ef61c889
KSS_Clustering[1]

# ╔═╡ 5df47f89-a6dd-492f-a810-5966d12e0340
N_runs = length(KSS_Clustering)

# ╔═╡ 2792f0c6-da3c-44ff-abf1-63c0c527c026
cluster_labels = [KSS_Clustering[i][2] for i in 1:N_runs]

# ╔═╡ 54977384-5eaf-4d7f-a072-a06e803aad2b
N_pixels = length(cluster_labels[1])

# ╔═╡ 0a227ba5-3000-4f91-9153-0458861e3b78
cluster_labels[1][1]

# ╔═╡ 59bce09f-3c01-40aa-b05e-55fcc6f3acc9
A = begin
	Aff = zeros(Float64, N_pixels, N_pixels)
	for labels in cluster_labels
		for i in 1:N_pixels, j in 1:N_pixels
			if labels[i] == labels[j]
				Aff[i, j] += 1
			end
		end
	end
	Aff ./ 100
end

# ╔═╡ 09805169-5fd0-4dd6-87ee-3ab66c499296
top_entries = 100

# ╔═╡ 7e968e6d-2a4f-4ff6-92b5-b045d51ff41b
# begin
	
# 	A_bar = zeros(size(A))
# 	sorted_indices = [sortperm(A[:, i], rev=true)[1:top_entries] for i in 1:size(A, 1)]
# 	[A_bar[i, 1:top_entries] .= [A[i, val] for (idx, val) in enumerate(sorted_indices[i])] for i in 1:size(A, 1)]
# end

# ╔═╡ c7e25010-014f-4e40-afa6-5a6a758c486e
A_bar

# ╔═╡ 8f133591-8f5e-4bb0-bc44-f06419da5c98
S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

# ╔═╡ a70f1bc8-b61b-4886-b231-c68eca53e580
diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ c39b8e19-cce5-481d-b844-d09e5e246fdb
D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ fc357b92-667c-4053-a259-d00f08e56cb5
L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ 8835255a-4fdc-4800-af50-210c5525ba0b
n_clusters = 5

# ╔═╡ fd9846a6-f2bd-4435-b63e-cbd5bd41f92c
decomp, history = partialschur(L_sym; nev=n_clusters, which=:SR)

# ╔═╡ 142b2164-312f-452c-88ae-4b98a1e4214b
V = mapslices(normalize, decomp.Q; dims=2)

# ╔═╡ 939eb048-4d50-4c45-9121-abb71666f4cd
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

# ╔═╡ 551a8d1b-a0dd-427e-8d86-da3433801f56
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ df981b8c-aefb-4245-bf53-592b784b031f
EKSS_Results = spec_clusterings[1].assignments

# ╔═╡ f8e930de-402b-4680-b5c5-c2c42297f610
A_EKSS = EKSS_Results[1:500]

# ╔═╡ bccba79f-ace7-4f1d-bb67-ddbcbb7d2d8b
B_EKSS = EKSS_Results[501:1000]

# ╔═╡ 4dcf2d1b-6c09-40fa-97d5-8bd05ae9e1bd
C_EKSS = EKSS_Results[1001:1500]

# ╔═╡ f82c22d9-2af8-45be-801b-749f59056059
D_EKSS = EKSS_Results[1501:2000]

# ╔═╡ b9c19b70-b660-4ee9-b0e4-392d5200ecd8
N_EKSS = EKSS_Results[2001:2500]

# ╔═╡ 85ef6270-1570-4145-9698-4c49e79a9d28
A_label_count_EKSS = [count(x -> (x==i), A_EKSS) / length(A_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ 43528cef-fa53-4c24-a988-2c0c93345d9d
B_label_count_EKSS = [count(x -> (x==i), B_EKSS) / length(B_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ 1dc75ac7-6148-443e-8bd4-d47c678ed9f9
C_label_count_EKSS = [count(x -> (x==i), C_EKSS) / length(C_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ d38078a8-df77-4694-9e3a-3fb9e25e9a8e
D_label_count_EKSS = [count(x -> (x==i), D_EKSS) / length(D_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ c7cddb89-610d-4a5a-a96e-b757f2813ca5
N_label_count_EKSS = [count(x -> (x==i), N_EKSS) / length(N_EKSS) * 100 for i in 1:n_clusters]

# ╔═╡ Cell order:
# ╠═1ca8b538-af39-11ef-2006-2778f9ccd5c2
# ╠═4059337b-d0c5-4948-94b6-c47c229625da
# ╠═15d6d724-e3ab-443c-99c6-3521976533d8
# ╠═46afcea3-a754-4731-81f9-e2a4c880d61d
# ╠═0ef5270a-6b13-44dc-a0ca-445cbac900e6
# ╠═66771350-3cd8-4891-b77a-6f5ec7f328e0
# ╠═8d94d2c8-79e9-442e-a3b7-46703a05d16d
# ╟─8694e758-e5c0-42c4-be53-bb6a7955fbe5
# ╠═bba58c1c-8a6f-4041-b294-2a0af2954f96
# ╠═1a36bc2e-8f5b-4ba6-aecb-1f35955827b9
# ╠═f86a0699-396a-4f60-b940-d6af17526b68
# ╟─8f31b045-c9c1-44c7-96c1-3433c2188206
# ╠═f6d0032b-00e9-4f2e-81b5-fd3646ea19cf
# ╠═a268bde4-37b4-4729-b7f6-4ab339f1c2f4
# ╠═169eb774-2299-4a54-9d68-1b1f4ba3dab0
# ╠═a43d05d7-fd37-42c8-bc0c-d32a019b5f8e
# ╠═f91fc2c3-7cca-4eec-b91e-c3cbe2208c62
# ╠═0ab76807-7edd-4585-a225-61ee3e439482
# ╠═4fd0a2f9-93b7-4709-afb7-90e6ef61c889
# ╠═5df47f89-a6dd-492f-a810-5966d12e0340
# ╠═2792f0c6-da3c-44ff-abf1-63c0c527c026
# ╠═54977384-5eaf-4d7f-a072-a06e803aad2b
# ╠═0a227ba5-3000-4f91-9153-0458861e3b78
# ╠═59bce09f-3c01-40aa-b05e-55fcc6f3acc9
# ╠═09805169-5fd0-4dd6-87ee-3ab66c499296
# ╠═7e968e6d-2a4f-4ff6-92b5-b045d51ff41b
# ╠═c7e25010-014f-4e40-afa6-5a6a758c486e
# ╠═8f133591-8f5e-4bb0-bc44-f06419da5c98
# ╠═a70f1bc8-b61b-4886-b231-c68eca53e580
# ╠═c39b8e19-cce5-481d-b844-d09e5e246fdb
# ╠═fc357b92-667c-4053-a259-d00f08e56cb5
# ╠═8835255a-4fdc-4800-af50-210c5525ba0b
# ╠═fd9846a6-f2bd-4435-b63e-cbd5bd41f92c
# ╠═142b2164-312f-452c-88ae-4b98a1e4214b
# ╠═939eb048-4d50-4c45-9121-abb71666f4cd
# ╠═551a8d1b-a0dd-427e-8d86-da3433801f56
# ╠═df981b8c-aefb-4245-bf53-592b784b031f
# ╠═f8e930de-402b-4680-b5c5-c2c42297f610
# ╠═bccba79f-ace7-4f1d-bb67-ddbcbb7d2d8b
# ╠═4dcf2d1b-6c09-40fa-97d5-8bd05ae9e1bd
# ╠═f82c22d9-2af8-45be-801b-749f59056059
# ╠═b9c19b70-b660-4ee9-b0e4-392d5200ecd8
# ╠═85ef6270-1570-4145-9698-4c49e79a9d28
# ╠═43528cef-fa53-4c24-a988-2c0c93345d9d
# ╠═1dc75ac7-6148-443e-8bd4-d47c678ed9f9
# ╠═d38078a8-df77-4694-9e3a-3fb9e25e9a8e
# ╠═c7cddb89-610d-4a5a-a96e-b757f2813ca5
