### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 18d70e9e-b154-4993-a3fb-baa1003f7eed
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ a64d86ab-c9f5-44b8-aadf-f66b1b9f92b2
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

# ╔═╡ 9c6a9a42-ae6d-11ef-1d5f-311cb3655916
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

# ╔═╡ cd5b3b47-96d5-4a56-9460-71893b5bd0bc
data_dir = joinpath(@__DIR__, "data")

# ╔═╡ 58f982b7-b8d2-41ca-aae5-198bf81759fb
md"""
#### Reading all the the first .npy files in the agents A,B,C,D
"""

# ╔═╡ c776f85f-8bef-4e86-a556-496ff4cc8d65
A_1_Path = joinpath(data_dir, "A", "A_1.npy")

# ╔═╡ 1bc03235-f0a3-4517-9be6-6329d8934ff4
B_1_Path = joinpath(data_dir, "B", "B_1.npy")

# ╔═╡ 0d5f401e-2c74-4c0c-a89e-3117710a5700
C_1_Path = joinpath(data_dir, "C", "C_1.npy")

# ╔═╡ 0434ce5d-ed1b-4746-a5a6-503fb04064f6
D_1_Path = joinpath(data_dir, "D", "D_1.npy")

# ╔═╡ c68812a7-0b64-4ad0-87c1-069ceb0e2a6a
Noise_path = joinpath(data_dir, "Noise", "Noise_1.npy")

# ╔═╡ fe5de852-e15b-418c-8052-c4a9a21b25a1
A_1 = permutedims(npzread(A_1_Path))

# ╔═╡ 7d50960d-082a-4c07-80ae-c7c09dfd32e2
B_1 = permutedims(npzread(B_1_Path))

# ╔═╡ b1c55a46-b277-4835-aadc-772a2bb0a88f
C_1 = permutedims(npzread(C_1_Path))

# ╔═╡ c28b4d5f-6bf1-4cbe-aa8c-dfb9e28a3caa
D_1 = permutedims(npzread(D_1_Path))

# ╔═╡ fb1e68ce-2647-4c1e-93da-be2b77b0c9bf
N = permutedims(npzread(Noise_path))

# ╔═╡ f5534483-1c56-41c5-a309-8708435b6389
md"""
#### Data Matrix for all the the first .npy files in the agents A,B,C,D
"""

# ╔═╡ c670fac6-a8be-4900-a35a-32fe8574afaa
D = hcat(A_1, B_1, C_1, D_1)

# ╔═╡ b9e1254d-125a-4be8-a01c-1a5a78c76ba2
md"""
### K-Subspaces
"""

# ╔═╡ 8642c0a4-9ef0-499d-b68e-176af46f6e2a
function polar(X)
	U, _, V = svd(X)
	U*V'
end

# ╔═╡ 3befa9a5-b34b-4da3-98d4-0976e3f7ca63
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

# ╔═╡ 9093b52f-6653-4e5c-96e6-123a9902a5fb
CACHEDIR = splitext(relpath(@__FILE__))[1]

# ╔═╡ cb7b25f2-273d-4a68-ab49-f79279462b05
fill(2, 4)

# ╔═╡ f4acff73-86ce-4561-96c4-bd71e3242582
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 017f233c-25a7-4c4b-858f-577e3654304d
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

# ╔═╡ 47ce0e35-573c-41e9-a7c9-226049379d0f
KSS_Clustering = batch_KSS(D, fill(2, 4); niters=200, nruns=100)

# ╔═╡ eb04b83e-5e40-4531-8f43-b3e9945fb762
KSS_Clustering[1]

# ╔═╡ 92f8441d-1426-4d23-841d-cf1e665b4bd5
N_runs = length(KSS_Clustering)

# ╔═╡ 06ed6eca-fbfc-4f66-bc0e-4de0eac3ef53
cluster_labels = [KSS_Clustering[i][2] for i in 1:N_runs]

# ╔═╡ 2101a8e1-700d-4d81-a3f9-15e283fc4a15
N_pixels = length(cluster_labels[1])

# ╔═╡ d68dd25a-79bd-46bc-90be-02e404d29265
cluster_labels[1][1]

# ╔═╡ 72478454-f83b-4e98-bc10-ff0444c1d770


# ╔═╡ 58464366-c9c2-4a5a-b3cf-bbcec13721b9
A = begin
	A = zeros(Float64, N_pixels, N_pixels)
	for labels in cluster_labels
		for i in 1:N_pixels, j in 1:N_pixels
			if labels[i] == labels[j]
				A[i, j] += 1
			end
		end
	end
	A ./ 100
end

# ╔═╡ 8f51c4cb-8fc9-4f2b-a621-fcd7d89a7244
S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

# ╔═╡ f679d461-ca23-462b-ac74-4365d4f2d2b4
diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ b6874048-8668-4927-b6aa-1e16a1063c75
D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ 8b68ef0a-97a3-4c28-b5bd-af3e2f294903
L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ 5578ef41-36eb-48e4-8c5b-18f00589d185
n_clusters = 4

# ╔═╡ 06d5b31c-03b5-4c4a-a9e8-dff6e1e470dc
decomp, history = partialschur(L_sym; nev=n_clusters, which=:SR)

# ╔═╡ 8d695307-0ddb-4f8d-b356-aeab8f00f90c
V = mapslices(normalize, decomp.Q; dims=2)

# ╔═╡ d7d5382c-1b7f-4a35-84c3-77ebb1e962ed
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

# ╔═╡ 0828d6c0-5a27-4b28-966f-e03d06724c3c
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ fbfd33e3-eaef-49f8-a07e-c1abba784e2b
SC_Results = spec_clusterings[1].assignments

# ╔═╡ dc15cb8a-04e9-4ef8-809d-fdcadab3e842
A1_Res_SC = SC_Results[1:240]

# ╔═╡ 96686b4e-ce37-4407-9979-d3094d543366
B1_Res_SC = SC_Results[241:513]

# ╔═╡ 826161f6-d5a2-4f05-9bde-eff31eb0ab58
C1_Res_SC = SC_Results[514:795]

# ╔═╡ a361d03f-6691-4a97-8368-82144cac020d


# ╔═╡ b6660c89-a96c-4540-bc3a-63157eb7f7b2
D1_Res_SC = SC_Results[796:1676]

# ╔═╡ bcb709ca-c1b2-40f3-9c9f-52d1db9a0048
A_label_count_EKSS = [count(x -> (x==i), A1_Res_SC) / length(A1_Res_SC) * 100 for i in 1:n_clusters]

# ╔═╡ 317a2be3-68ba-4c1f-bb73-d1ffc83bc45f
B_label_count_EKSS = [count(x -> (x==i), B1_Res_SC) / length(B1_Res_SC) * 100 for i in 1:n_clusters]

# ╔═╡ 5391d4d9-feee-41a5-aa60-a43e4b3f494c
C_label_count_EKSS = [count(x -> (x==i), C1_Res_SC) / length(C1_Res_SC) * 100 for i in 1:n_clusters]

# ╔═╡ 51082690-d147-41f3-930a-1d4f0b60b7ba
D_label_count_EKSS = [count(x -> (x==i), D1_Res_SC) / length(D1_Res_SC) * 100 for i in 1:n_clusters]

# ╔═╡ 3253c492-fc96-4083-92b7-6b341526f331
C_label_1 = findall(C1_Res_SC .==(1))

# ╔═╡ 7a8e84ec-d33e-49f5-9a35-d6325b88aa2d
C_label_3 = findall(C1_Res_SC .==(3))

# ╔═╡ c043ce09-a518-4341-9ec8-dcfea8b728ef
C_start_idx = 514

# ╔═╡ e05244e2-662f-4149-b5c8-e0a8c24bcf35
C_orig_idx = C_label_1 .+ (C_start_idx - 1)

# ╔═╡ 2dcfee65-45bc-45cf-8975-0c39c3a771f0
C3_orig_idx = C_label_3 .+ (C_start_idx - 1)

# ╔═╡ 5e3521cb-ae02-4f9b-a921-1bb11e42eb6c
C_1s = [D[:, idx] for idx in C_orig_idx]

# ╔═╡ 1e74ef70-2c84-4db7-851e-ac1829be1bfd
C_3s = [D[:, idx] for idx in C3_orig_idx]

# ╔═╡ 151a7f51-a773-476f-a119-8b688c51f076
C_l1 = hcat(C_1s...)

# ╔═╡ ad437c06-a8f0-4304-b5ad-f67b522129af
C_l3 = hcat(C_3s...)

# ╔═╡ d95c604e-f7f1-4c57-a870-134fda8d907e
md"""
### Plotting the signals labeled as 1
"""

# ╔═╡ 7fb05ff7-c40e-431b-8064-ebc5d7bdbfe2
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1])
	lines!(ax, vec(mean(C_l1, dims=1)))
	fig
end	

# ╔═╡ 2434fc83-648b-46df-a446-8c5e9409580b
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1])
	lines!(ax, vec(mean(C_l3, dims=1)))
	fig
end	

# ╔═╡ 935ce798-3823-4644-8537-ec82e7f8fdc9
min_idx_KSS = argmax(KSS_Clustering[i][3] for i in 1:100)

# ╔═╡ a6a34fb4-c1e9-41a1-8a96-905913b1aac2
KSS_Results = KSS_Clustering[min_idx_KSS][2]

# ╔═╡ 68eed153-b75a-439c-ba55-37aa22d842fb
A1_Res = KSS_Results[1:240]

# ╔═╡ 61de8c7f-1521-4082-b925-e30ae4c23d54
B1_Res = KSS_Results[241:513]

# ╔═╡ 66b2de17-c727-44b2-9681-b5c8b6203bb9
C1_Res = KSS_Results[514:795]

# ╔═╡ 40230085-bd95-47fb-9e9e-807fd4eeba95
D1_Res = KSS_Results[796:1676]

# ╔═╡ 885c18ad-17f8-4012-aaf8-8a7a78d1bc98


# ╔═╡ 1c3840a0-475e-48ca-9f93-428ca8ce27f0
A_label_count = [count(x -> (x==i), A1_Res) / length(A1_Res) * 100 for i in 1:4]

# ╔═╡ 13c4f451-ec0e-4756-9996-792df499e15b
B_label_count = [count(x -> (x==i), B1_Res) / length(B1_Res) * 100 for i in 1:4]

# ╔═╡ afa4744d-1fc7-4468-af0d-0733170d5ef6
C_label_count = [count(x -> (x==i), C1_Res) / length(C1_Res) * 100 for i in 1:4]

# ╔═╡ b65146e7-ea8b-4672-a5c4-ec3bab9dc229
D_label_count = [count(x -> (x==i), D1_Res) / length(D1_Res) * 100 for i in 1:4]

# ╔═╡ 2656dc33-d6c7-430a-b96b-c0505cd465cc


# ╔═╡ Cell order:
# ╠═9c6a9a42-ae6d-11ef-1d5f-311cb3655916
# ╠═18d70e9e-b154-4993-a3fb-baa1003f7eed
# ╠═a64d86ab-c9f5-44b8-aadf-f66b1b9f92b2
# ╠═cd5b3b47-96d5-4a56-9460-71893b5bd0bc
# ╟─58f982b7-b8d2-41ca-aae5-198bf81759fb
# ╠═c776f85f-8bef-4e86-a556-496ff4cc8d65
# ╠═1bc03235-f0a3-4517-9be6-6329d8934ff4
# ╠═0d5f401e-2c74-4c0c-a89e-3117710a5700
# ╠═0434ce5d-ed1b-4746-a5a6-503fb04064f6
# ╠═c68812a7-0b64-4ad0-87c1-069ceb0e2a6a
# ╠═fe5de852-e15b-418c-8052-c4a9a21b25a1
# ╠═7d50960d-082a-4c07-80ae-c7c09dfd32e2
# ╠═b1c55a46-b277-4835-aadc-772a2bb0a88f
# ╠═c28b4d5f-6bf1-4cbe-aa8c-dfb9e28a3caa
# ╠═fb1e68ce-2647-4c1e-93da-be2b77b0c9bf
# ╠═f5534483-1c56-41c5-a309-8708435b6389
# ╠═c670fac6-a8be-4900-a35a-32fe8574afaa
# ╟─b9e1254d-125a-4be8-a01c-1a5a78c76ba2
# ╠═8642c0a4-9ef0-499d-b68e-176af46f6e2a
# ╠═3befa9a5-b34b-4da3-98d4-0976e3f7ca63
# ╠═9093b52f-6653-4e5c-96e6-123a9902a5fb
# ╠═017f233c-25a7-4c4b-858f-577e3654304d
# ╠═cb7b25f2-273d-4a68-ab49-f79279462b05
# ╠═f4acff73-86ce-4561-96c4-bd71e3242582
# ╠═47ce0e35-573c-41e9-a7c9-226049379d0f
# ╠═eb04b83e-5e40-4531-8f43-b3e9945fb762
# ╠═92f8441d-1426-4d23-841d-cf1e665b4bd5
# ╠═06ed6eca-fbfc-4f66-bc0e-4de0eac3ef53
# ╠═2101a8e1-700d-4d81-a3f9-15e283fc4a15
# ╠═d68dd25a-79bd-46bc-90be-02e404d29265
# ╠═72478454-f83b-4e98-bc10-ff0444c1d770
# ╠═58464366-c9c2-4a5a-b3cf-bbcec13721b9
# ╠═8f51c4cb-8fc9-4f2b-a621-fcd7d89a7244
# ╠═f679d461-ca23-462b-ac74-4365d4f2d2b4
# ╠═b6874048-8668-4927-b6aa-1e16a1063c75
# ╠═8b68ef0a-97a3-4c28-b5bd-af3e2f294903
# ╠═5578ef41-36eb-48e4-8c5b-18f00589d185
# ╠═06d5b31c-03b5-4c4a-a9e8-dff6e1e470dc
# ╠═8d695307-0ddb-4f8d-b356-aeab8f00f90c
# ╠═d7d5382c-1b7f-4a35-84c3-77ebb1e962ed
# ╠═0828d6c0-5a27-4b28-966f-e03d06724c3c
# ╠═fbfd33e3-eaef-49f8-a07e-c1abba784e2b
# ╠═dc15cb8a-04e9-4ef8-809d-fdcadab3e842
# ╠═96686b4e-ce37-4407-9979-d3094d543366
# ╠═826161f6-d5a2-4f05-9bde-eff31eb0ab58
# ╠═a361d03f-6691-4a97-8368-82144cac020d
# ╠═b6660c89-a96c-4540-bc3a-63157eb7f7b2
# ╠═bcb709ca-c1b2-40f3-9c9f-52d1db9a0048
# ╠═317a2be3-68ba-4c1f-bb73-d1ffc83bc45f
# ╠═5391d4d9-feee-41a5-aa60-a43e4b3f494c
# ╠═51082690-d147-41f3-930a-1d4f0b60b7ba
# ╠═3253c492-fc96-4083-92b7-6b341526f331
# ╠═7a8e84ec-d33e-49f5-9a35-d6325b88aa2d
# ╠═c043ce09-a518-4341-9ec8-dcfea8b728ef
# ╠═e05244e2-662f-4149-b5c8-e0a8c24bcf35
# ╠═2dcfee65-45bc-45cf-8975-0c39c3a771f0
# ╠═5e3521cb-ae02-4f9b-a921-1bb11e42eb6c
# ╠═1e74ef70-2c84-4db7-851e-ac1829be1bfd
# ╠═151a7f51-a773-476f-a119-8b688c51f076
# ╠═ad437c06-a8f0-4304-b5ad-f67b522129af
# ╟─d95c604e-f7f1-4c57-a870-134fda8d907e
# ╠═7fb05ff7-c40e-431b-8064-ebc5d7bdbfe2
# ╠═2434fc83-648b-46df-a446-8c5e9409580b
# ╠═935ce798-3823-4644-8537-ec82e7f8fdc9
# ╠═a6a34fb4-c1e9-41a1-8a96-905913b1aac2
# ╠═68eed153-b75a-439c-ba55-37aa22d842fb
# ╠═61de8c7f-1521-4082-b925-e30ae4c23d54
# ╠═66b2de17-c727-44b2-9681-b5c8b6203bb9
# ╠═40230085-bd95-47fb-9e9e-807fd4eeba95
# ╠═885c18ad-17f8-4012-aaf8-8a7a78d1bc98
# ╠═1c3840a0-475e-48ca-9f93-428ca8ce27f0
# ╠═13c4f451-ec0e-4756-9996-792df499e15b
# ╠═afa4744d-1fc7-4468-af0d-0733170d5ef6
# ╠═b65146e7-ea8b-4672-a5c4-ec3bab9dc229
# ╠═2656dc33-d6c7-430a-b96b-c0505cd465cc
