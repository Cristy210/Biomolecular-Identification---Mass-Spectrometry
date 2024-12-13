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

# ╔═╡ 9267ff10-b7d9-11ef-0d41-09b0dfe4e26e
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ dbe8eea1-ac17-406d-82f2-5ff9bc4d4884
begin

    Pkg.add("GLMakie")
end

# ╔═╡ d96fc7cf-ee9c-49bd-9e9e-bc95bf8a30f7
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

# ╔═╡ d1d21814-12ce-4298-9bf2-a9a561096291
data_dir = joinpath(@__DIR__,"data", "new_data")

# ╔═╡ 98ab1a3e-516f-405f-b36f-1ae3ab07af15
A_Path = joinpath(data_dir, "A.npy")

# ╔═╡ 0eff0088-095a-49db-82bf-3daf5e5fe0db
B_Path = joinpath(data_dir, "B.npy")

# ╔═╡ e845a064-9b6d-4c1c-9ebf-2f4dac11a8e9
C_Path = joinpath(data_dir, "C.npy")

# ╔═╡ c18193ce-2f91-4c35-883b-2c4e81adaef7
D_Path = joinpath(data_dir, "D.npy")

# ╔═╡ ec000b2b-297e-4d34-b61d-a0ef9d3adb58
Noise_path = joinpath(data_dir, "Noise.npy")

# ╔═╡ 76228ae7-5b9f-4ad5-b782-15b5dc54ca07
A = permutedims(npzread(A_Path))

# ╔═╡ c8204765-0947-42a9-b23c-f9c3c969677f
B = permutedims(npzread(B_Path))	

# ╔═╡ 38beb1a3-e2c8-481d-a5eb-398808e330e2
C = permutedims(npzread(C_Path))

# ╔═╡ 5b9dcfac-c609-4d9f-b99c-669ad126f23a
D = permutedims(npzread(D_Path))

# ╔═╡ 9b8e2a3a-1df8-4bbb-a1df-c79f9a677c07
N = permutedims(npzread(Noise_path))

# ╔═╡ 5f0d6b10-4159-498b-924e-435c767c082e
begin
	
	
	# Initialize a figure
	fig = Figure(resolution = (900, 800))
	
	# Create a 1x3 grid layout
	ax1 = Axis(fig[1, 1], title = "A")
	ax2 = Axis(fig[1, 2], title = "C")
	ax3 = Axis(fig[1, 3], title = "Background")

	ax4 = Axis(fig[3, 1], title = "A")
	ax5 = Axis(fig[3, 2], title = "C")
	ax6 = Axis(fig[3, 3], title = "Background")
	vmin,vmax=0,3
	# Plot heatmaps in each axis
	hm1=heatmap!(ax1, A, colormap = :viridis, colorrange = (vmin, vmax))
	hm2=heatmap!(ax2, C, colormap = :viridis, colorrange = (vmin, vmax))
	hm3=heatmap!(ax3, N, colormap = :viridis, colorrange = (vmin, vmax))
	lines!(ax4,mean(A,dims=2)[:,1])
	lines!(ax5,mean(C,dims=2)[:,1])
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

# ╔═╡ 69b012e6-e4a3-467a-91be-852f79d72888
begin
	# Compute the singular values
	singular_values1 = svd(A').S
	singular_values2 = svd(C').S
	singular_values3 = svd(N').S
	
	# Compute the log of the singular values
	log_singular_values1 = log10.(singular_values1)
	log_singular_values2 = log10.(singular_values2)
	log_singular_values3 = log10.(singular_values3)
	
	# Initialize a figure with size
	fig2 = Figure(size = (1000,400))
	
	# Create axes for each subplot
	ax7 = Axis(fig2[1, 1], title = "Log Singular Values (Protein)", xlabel = "Index", ylabel = "Log10(Singular Value)")
	ax8 = Axis(fig2[1, 2], title = "Log Singular Values (Bacteria)", xlabel = "Index", ylabel = "Log10(Singular Value)")
	ax9 = Axis(fig2[1, 3], title = "Log Singular Values (Noise)", xlabel = "Index", ylabel = "Log10(Singular Value)")
	
	# Plot log singular values on each axis
	scatter!(ax7, 1:length(log_singular_values1), log_singular_values1, color = :blue)
	scatter!(ax8, 1:length(log_singular_values2), log_singular_values2, color = :red)
	scatter!(ax9, 1:length(log_singular_values3), log_singular_values3, color = :green)

	linkyaxes!(ax7,ax8,ax9)

	# Display the figure
	fig2
	
end

# ╔═╡ 24189239-67e3-482c-83f2-5fb5edb1e561
function low_rank_approx(data,k)
	U, S, V = svd(data)

	U_k = U[:, 1:k]        # First k columns of U
	S_k = Diagonal(S[1:k]) # First k singular values (diagonal matrix)
	V_k = V[:, 1:k]        # First k columns of V
	
	# Reconstruct the k-rank approximation
	data_k = U_k * S_k * V_k'
	return data_k'
end


# ╔═╡ e787dd88-bed7-40e4-adf9-827420d8f4b0
rank=5

# ╔═╡ 5f0b3ffc-7f9d-4bd7-b0be-2bbbd247464c
begin
	
	
	# Initialize a figure
	fig3 = Figure(resolution = (900, 800))
	low_rank_A = low_rank_approx(A',rank)
	println(size(low_rank_A))
	low_rank_C = low_rank_approx(C',rank)
	low_rank_N = low_rank_approx(N',rank)
	# Create a 1x3 grid layout
	ax10 = Axis(fig3[1, 1], title = "A")
	ax11 = Axis(fig3[1, 2], title = "C")
	ax12 = Axis(fig3[1, 3], title = "Background")

	ax13 = Axis(fig3[3, 1], title = "A")
	ax14 = Axis(fig3[3, 2], title = "C")
	ax15 = Axis(fig3[3, 3], title = "Background")
	# Plot heatmaps in each axis
	hm4=heatmap!(ax10, low_rank_A, colormap = :viridis, colorrange = (vmin, vmax))
	hm5=heatmap!(ax11, low_rank_C, colormap = :viridis, colorrange = (vmin, vmax))
	hm6=heatmap!(ax12, low_rank_N, colormap = :viridis, colorrange = (vmin, vmax))
	lines!(ax13,mean(low_rank_A,dims=2)[:,1])
	lines!(ax14,mean(low_rank_C,dims=2)[:,1])
	lines!(ax15,mean(low_rank_N,dims=2)[:,1])

	# Add individual color bars for each heatmap
	Colorbar(fig3[2, 1], hm4, label = "", vertical = false)
	Colorbar(fig3[2, 2], hm5, label = "", vertical = false)
	Colorbar(fig3[2, 3], hm6, label = "", vertical = false)
	
	# Adjust layout
	#fig.layoutgap[] = 10  # Optional: Adjust gaps between plots for clarity
	
	# Display the figure
	fig3
end

# ╔═╡ 50d5234c-b6d7-49fe-af2d-b5b62f44c7c3
@bind idx PlutoUI.Slider(1:500; show_value=true)

# ╔═╡ 5390b452-996c-4828-b84b-3eaf1d152380
begin
	
	using GLMakie  # Import this to use sliders and other interactive components

	# Create a figure and axis layout
	fig4 = Figure(resolution = (1200, 500))
	ax20 = Axis(fig4[1, 1], title = "Plot 1")
	ax21 = Axis(fig4[1, 2], title = "Plot 2")
	ax22 = Axis(fig4[1, 3], title = "Plot 3")
	
	# Initialize plots (dummy data to start)
	line1 = lines!(ax20, A'[1, :])
	line2 = lines!(ax21, C'[1, :])
	line3 = lines!(ax22, N'[1, :])
	
	# Add a slider to control the index
	

	# Update function for the slider
	# idx=3
	line1[1] = A'[idx, :]
	line2[1] = C'[idx, :]
	line3[1] = N'[idx, :]
	
	
	# Display the figure
	fig4
	
end

# ╔═╡ Cell order:
# ╠═9267ff10-b7d9-11ef-0d41-09b0dfe4e26e
# ╠═d96fc7cf-ee9c-49bd-9e9e-bc95bf8a30f7
# ╠═d1d21814-12ce-4298-9bf2-a9a561096291
# ╠═98ab1a3e-516f-405f-b36f-1ae3ab07af15
# ╠═0eff0088-095a-49db-82bf-3daf5e5fe0db
# ╠═e845a064-9b6d-4c1c-9ebf-2f4dac11a8e9
# ╠═c18193ce-2f91-4c35-883b-2c4e81adaef7
# ╠═ec000b2b-297e-4d34-b61d-a0ef9d3adb58
# ╠═76228ae7-5b9f-4ad5-b782-15b5dc54ca07
# ╠═c8204765-0947-42a9-b23c-f9c3c969677f
# ╠═38beb1a3-e2c8-481d-a5eb-398808e330e2
# ╠═5b9dcfac-c609-4d9f-b99c-669ad126f23a
# ╠═9b8e2a3a-1df8-4bbb-a1df-c79f9a677c07
# ╟─5f0d6b10-4159-498b-924e-435c767c082e
# ╟─69b012e6-e4a3-467a-91be-852f79d72888
# ╠═24189239-67e3-482c-83f2-5fb5edb1e561
# ╠═e787dd88-bed7-40e4-adf9-827420d8f4b0
# ╟─5f0b3ffc-7f9d-4bd7-b0be-2bbbd247464c
# ╠═dbe8eea1-ac17-406d-82f2-5ff9bc4d4884
# ╠═50d5234c-b6d7-49fe-af2d-b5b62f44c7c3
# ╠═5390b452-996c-4828-b84b-3eaf1d152380
