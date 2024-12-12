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

# ╔═╡ 3ed3e6a0-bf84-4ce1-b5e7-7a092c9bc485
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 156f528c-8253-4ff3-880c-725ebd45f4f5
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

# ╔═╡ 11d7e2e7-e7cb-4236-baee-c66a8655eb74
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

# ╔═╡ d4178dfd-1f6b-4f56-b82a-79139eacd853
md"""
### Activate Project Directory
"""

# ╔═╡ 300fb9c2-d3da-4e7b-8f65-bf97c0c975da


# ╔═╡ 6af5021a-6dd2-4295-885c-62e88c562232
dir = joinpath(@__DIR__, "data", "new_data")

# ╔═╡ b270e537-1a9a-495e-93f6-56a93e61560a
md"""
### Signals with different time steps
"""

# ╔═╡ 44455aeb-ad57-4b12-b21e-b552e39088c5
@bind Features Select(["size_3737", "size_7474", "size_14948", "size_74740"])

# ╔═╡ aa0aab75-97e8-4911-a5b6-f49f7492bc5f
data_dir = joinpath(dir, "$Features")

# ╔═╡ 17052856-c3c2-493b-a263-d91d9b6ecc0a
file_names = ["A.npy", "B.npy", "C.npy", "D.npy"]

# ╔═╡ b61fe3ea-af55-4ba7-8f29-0b545203f884
file_paths = [joinpath(data_dir, file_name) for file_name in file_names]

# ╔═╡ 27311cea-60c0-4d92-bd48-8ae8cd80b97b
Data = [permutedims(npzread(path)) for path in file_paths]

# ╔═╡ cd0a013d-4f80-4fd4-81f6-ae7e16eee82d
md"""
#### Data Matrix for all the .npy files from the agents A,B,C,D, and Noise
"""

# ╔═╡ 3684150b-92d5-4116-ba85-467923db0277
D_org = hcat(Data...)

# ╔═╡ 6c289b70-e518-4561-9536-8115425e598a
D_abs = abs.(D_org.* (D_org .> 0))

# ╔═╡ 242c152e-96f0-48b0-ab0a-f56ef70b7691
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
	hm1=heatmap!(ax11, Data[1], colormap = :viridis, colorrange = (vmin, vmax))
	hm2=heatmap!(ax21, Data[2], colormap = :viridis, colorrange = (vmin, vmax))
	hm3=heatmap!(ax31, Data[3], colormap = :viridis, colorrange = (vmin, vmax))
	hm4=heatmap!(ax41, Data[4], colormap = :viridis, colorrange = (vmin, vmax))

	lines!(ax13,mean(Data[1],dims=2)[:,1])
	lines!(ax23,mean(Data[2],dims=2)[:,1])
	lines!(ax33,mean(Data[3],dims=2)[:,1])
	lines!(ax43,mean(Data[4],dims=2)[:,1])

	Colorbar(grid1[2, 1], hm1, label = "", vertical = false)
	Colorbar(grid2[2, 1], hm2, label = "", vertical = false)
	Colorbar(grid3[2, 1], hm3, label = "", vertical = false)
	Colorbar(grid4[2, 1], hm4, label = "", vertical = false)

	fig
end

# ╔═╡ Cell order:
# ╟─11d7e2e7-e7cb-4236-baee-c66a8655eb74
# ╟─d4178dfd-1f6b-4f56-b82a-79139eacd853
# ╠═300fb9c2-d3da-4e7b-8f65-bf97c0c975da
# ╠═3ed3e6a0-bf84-4ce1-b5e7-7a092c9bc485
# ╠═156f528c-8253-4ff3-880c-725ebd45f4f5
# ╠═6af5021a-6dd2-4295-885c-62e88c562232
# ╠═b270e537-1a9a-495e-93f6-56a93e61560a
# ╠═44455aeb-ad57-4b12-b21e-b552e39088c5
# ╠═aa0aab75-97e8-4911-a5b6-f49f7492bc5f
# ╠═17052856-c3c2-493b-a263-d91d9b6ecc0a
# ╠═b61fe3ea-af55-4ba7-8f29-0b545203f884
# ╠═27311cea-60c0-4d92-bd48-8ae8cd80b97b
# ╟─cd0a013d-4f80-4fd4-81f6-ae7e16eee82d
# ╠═3684150b-92d5-4116-ba85-467923db0277
# ╠═6c289b70-e518-4561-9536-8115425e598a
# ╠═242c152e-96f0-48b0-ab0a-f56ef70b7691
