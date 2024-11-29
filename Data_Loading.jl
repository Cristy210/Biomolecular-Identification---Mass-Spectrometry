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

# ╔═╡ 05602255-cb32-4da9-9198-1c75528cf61d
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ ef4d5137-8b69-4590-b005-a589fa29e5ef
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random

# ╔═╡ 253941dc-a2db-11ef-3f8d-7bff3a38c73c
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

# ╔═╡ 6a16b1ba-4922-4371-a1a6-400557cc40e7
md"""
### Activate Project Directory
"""

# ╔═╡ 65edf46e-a573-466d-90d9-d92929d66961
md"""
### Import Necessary packages
"""

# ╔═╡ 2c132f34-2398-4493-a24b-1eb020aef0bf
md"""
### Data Directory
"""

# ╔═╡ 62c5a0c1-9c16-459c-acd4-695c318ba29b
data_dir = joinpath(@__DIR__, "data")

# ╔═╡ 22974108-69ba-4aff-9168-a9b6c144d9bc
Data_Paths = Dict(
	"A" => Dict(
		"A-1" => joinpath(data_dir, "A", "A_1.npy"),
		"A-2" => joinpath(data_dir, "A", "A_2.npy"),
		"A-3" => joinpath(data_dir, "A", "A_3.npy"),
		"A-4" => joinpath(data_dir, "A", "A_4.npy"),
		"A-5" => joinpath(data_dir, "A", "A_5.npy")
	),
	"B" => Dict(
		"B-1" => joinpath(data_dir, "B", "B_1.npy"),
		"B-2" => joinpath(data_dir, "B", "B_2.npy")
	),
	"C" => Dict(
		"C-1" => joinpath(data_dir, "C", "C_1.npy"),
		"C-2" => joinpath(data_dir, "C", "C_2.npy"),
		"C-3" => joinpath(data_dir, "C", "C_3.npy")
	),
	"D" => Dict(
		"D-1" => joinpath(data_dir, "D", "D_1.npy"),
		"D-2" => joinpath(data_dir, "D", "D_2.npy")
	)
)

# ╔═╡ 48daa613-ff44-4f75-ad35-813a9175ec48
md"""
#### Agent Selection
"""

# ╔═╡ b92e040c-670f-4a16-b96a-306e91f3e6a8
@bind Agent Select(["A", "B", "C", "D"])

# ╔═╡ 54f208da-0aac-42b1-a8f0-f3554d851652
Selected_Paths = Data_Paths[Agent]

# ╔═╡ 3e98d3e3-f9a7-4e86-93ff-6caca31fc0bd
@bind file_path Select(collect(keys(Selected_Paths)))

# ╔═╡ fc12a93a-5614-48bb-9c15-d4c8401dd8a6
md"""
## Data corresponding to the Agent
"""

# ╔═╡ 02631383-bf6a-4497-bed9-dddd9d1f1926
D = npzread(Selected_Paths[file_path])

# ╔═╡ f5c5c65c-16ad-4789-b074-ede9a4d2b3e7
B = permutedims(D)

# ╔═╡ 40902ee0-af38-420b-ac12-0b9d03cc486c
md"""
### Plotting the heatmap
"""

# ╔═╡ 3bc221ae-042f-472c-8bd3-a6b48274c6a5
with_theme() do
	fig = Figure(; size=(700, 600))
	ax = Axis(fig[1, 1], yreversed=true)
	hm = heatmap!(ax, permutedims(D), colormap=:viridis, colorrange=(-5, 5))
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 5c50fdf3-ab4b-4c64-80a2-fe81a7247615
md"""
### KSS Implementation
"""

# ╔═╡ 6048a7ea-41d1-4a64-aed8-940dfb5d1d3e
function polar(X)
	U, _, V = svd(X)
	U*V'
end

# ╔═╡ 53842642-8f41-431d-88b2-511be889cb92
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
				U[k] = polar(randn(D, d[k]))
			else
				U[k] = svd(view(X, :, ilist); full=true).U[:,1:d[k]]
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

# ╔═╡ 2210d3ea-4a73-48f3-ac6a-00178cf93eb0
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ a1367151-c6ae-4031-9b9f-c811fc521a7c
CACHEDIR = splitext(relpath(@__FILE__))[1]

# ╔═╡ 37e72dc4-853c-4239-b4ec-0868a1f45c9b
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

# ╔═╡ 7515a10d-58db-41a0-87f6-cbd1a25f5d7d
fill(2, 2)

# ╔═╡ d5edb3e2-ad7d-4499-bb49-2afa6b8426f9
fill(1, 2)

# ╔═╡ 869f0d2f-6afb-4ce2-997d-814e88908f5c
# D_1 = D[index_1, :]

# ╔═╡ 276d3181-625a-49ab-8053-573f89f80a30
# D_2 = D[index_2, :]

# ╔═╡ 25539b53-155c-4944-95f0-1b4e1522f8ad
with_theme() do
	fig = Figure(; size=(700, 500))
	ax = Axis(fig[1, 1], aspect=nothing)
	lines!(ax, vec(mean(D, dims=1)))
	# lines!(ax, D[1, :])
	# scatter!(ax, D[1, :])
	# Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 6cdf9fc2-48b3-4790-ad8b-d71b52b4cd55
with_theme() do
	fig = Figure(; size=(800, 700))
	ax = Axis(fig[1, 1])
	ax1 = Axis(fig[1, 2])
	lines!(ax, vec(mean(D_1, dims=1)))
	lines!(ax1, vec(mean(D_2, dims=1)))
	fig
end

# ╔═╡ c0849f4d-4288-4f02-a104-e5d0eea2f01d
md"""
### Rough Work
"""

# ╔═╡ 0b26340c-6857-43e5-b644-884ffee4e340
Noise_path = joinpath(@__DIR__, "data", "Noise", "Noise_1.npy")

# ╔═╡ c600b4f4-4c74-4dc0-975a-1cbfb200d9f6
Noise = npzread(Noise_path)

# ╔═╡ 8309751a-fbe3-4f9e-89da-bd0031430c7b
N_clipped = Noise[1:273, :]

# ╔═╡ 3ad96e81-9fbb-41ec-8714-9b75678879a7
B1_N_Clipped = vcat(D, N_clipped)

# ╔═╡ 05e4b893-cc8e-49eb-b2e0-2df804d1f06e
permutedims(B1_N_Clipped)

# ╔═╡ a874456f-58a9-4d2e-9864-d80df5312d53
results = batch_KSS(permutedims(B1_N_Clipped), fill(2, 2); niters=200, nruns=100)

# ╔═╡ 66bcee83-71e6-4e24-bd43-21fa0a2d56f0
unique(results[1][2])

# ╔═╡ 9dbb856f-6746-4425-9bcf-8f6fb2704ea8
min_cost_idx = argmax(results[i][3] for i in 1:100)

# ╔═╡ 1db8094e-d671-4471-8fac-68fa464a8221
D_assign = results[min_cost_idx][2]

# ╔═╡ 7ff1a038-24ca-4136-a7fd-c9d33f826089
index_1 = findall(==(1), D_assign)

# ╔═╡ 45639ee9-45af-48e9-8be3-223eec894f3f
index_2 = findall(==(2), D_assign)

# ╔═╡ 1114c67b-5513-40f6-b6c3-ccc23fe3d8b0
d_N_clipped_1 = B1_N_Clipped[index_1, :]

# ╔═╡ 25d4ec36-e620-463c-857e-bb39a8927687
d_N_clipped_2 = B1_N_Clipped[index_2, :]

# ╔═╡ f66be983-d45f-4cf0-aeae-f475ede8f36e
with_theme() do
	fig = Figure(; size=(900, 700))
	ax = Axis(fig[1, 1])
	ax1 = Axis(fig[1, 2])
	ax2 = Axis(fig[1, 3])
	lines!(ax, vec(mean(d_N_clipped_1, dims=1)))
	lines!(ax1, vec(mean(N_clipped, dims=1)))
	lines!(ax2, vec(mean(d_N_clipped_2, dims=1)))
	# lines!(ax1, vec(mean(d_N_clipped_2, dims=1)))
	fig
end

# ╔═╡ fc6df003-01ee-4b38-84ba-1e1895988940


# ╔═╡ Cell order:
# ╠═253941dc-a2db-11ef-3f8d-7bff3a38c73c
# ╟─6a16b1ba-4922-4371-a1a6-400557cc40e7
# ╠═05602255-cb32-4da9-9198-1c75528cf61d
# ╟─65edf46e-a573-466d-90d9-d92929d66961
# ╠═ef4d5137-8b69-4590-b005-a589fa29e5ef
# ╟─2c132f34-2398-4493-a24b-1eb020aef0bf
# ╠═62c5a0c1-9c16-459c-acd4-695c318ba29b
# ╠═22974108-69ba-4aff-9168-a9b6c144d9bc
# ╟─48daa613-ff44-4f75-ad35-813a9175ec48
# ╠═b92e040c-670f-4a16-b96a-306e91f3e6a8
# ╠═54f208da-0aac-42b1-a8f0-f3554d851652
# ╠═3e98d3e3-f9a7-4e86-93ff-6caca31fc0bd
# ╟─fc12a93a-5614-48bb-9c15-d4c8401dd8a6
# ╠═02631383-bf6a-4497-bed9-dddd9d1f1926
# ╠═f5c5c65c-16ad-4789-b074-ede9a4d2b3e7
# ╟─40902ee0-af38-420b-ac12-0b9d03cc486c
# ╠═3bc221ae-042f-472c-8bd3-a6b48274c6a5
# ╟─5c50fdf3-ab4b-4c64-80a2-fe81a7247615
# ╠═6048a7ea-41d1-4a64-aed8-940dfb5d1d3e
# ╠═53842642-8f41-431d-88b2-511be889cb92
# ╠═37e72dc4-853c-4239-b4ec-0868a1f45c9b
# ╠═2210d3ea-4a73-48f3-ac6a-00178cf93eb0
# ╠═a1367151-c6ae-4031-9b9f-c811fc521a7c
# ╠═05e4b893-cc8e-49eb-b2e0-2df804d1f06e
# ╠═7515a10d-58db-41a0-87f6-cbd1a25f5d7d
# ╠═a874456f-58a9-4d2e-9864-d80df5312d53
# ╠═66bcee83-71e6-4e24-bd43-21fa0a2d56f0
# ╠═9dbb856f-6746-4425-9bcf-8f6fb2704ea8
# ╠═1db8094e-d671-4471-8fac-68fa464a8221
# ╠═7ff1a038-24ca-4136-a7fd-c9d33f826089
# ╠═45639ee9-45af-48e9-8be3-223eec894f3f
# ╠═d5edb3e2-ad7d-4499-bb49-2afa6b8426f9
# ╠═869f0d2f-6afb-4ce2-997d-814e88908f5c
# ╠═276d3181-625a-49ab-8053-573f89f80a30
# ╠═25539b53-155c-4944-95f0-1b4e1522f8ad
# ╠═6cdf9fc2-48b3-4790-ad8b-d71b52b4cd55
# ╟─c0849f4d-4288-4f02-a104-e5d0eea2f01d
# ╠═0b26340c-6857-43e5-b644-884ffee4e340
# ╠═c600b4f4-4c74-4dc0-975a-1cbfb200d9f6
# ╠═8309751a-fbe3-4f9e-89da-bd0031430c7b
# ╠═3ad96e81-9fbb-41ec-8714-9b75678879a7
# ╠═1114c67b-5513-40f6-b6c3-ccc23fe3d8b0
# ╠═25d4ec36-e620-463c-857e-bb39a8927687
# ╠═f66be983-d45f-4cf0-aeae-f475ede8f36e
# ╠═fc6df003-01ee-4b38-84ba-1e1895988940
