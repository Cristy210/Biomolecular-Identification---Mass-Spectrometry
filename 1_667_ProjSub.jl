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

# ╔═╡ 3011a102-ce11-4df1-8a1c-2624c3920430
using Base.Threads

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

# ╔═╡ 2b3a9820-15bd-4215-b723-8184e59e3bf4
md"""
# Ensemble K-Subpsaces
#### ELEG667 Matrix and Tensor Methods
##### - Kyle Regan
##### - Christian Newman-Sanders
##### - Cristy Mathey
"""

# ╔═╡ dbe2c43a-7f5d-443b-9253-dc1c630cdb15
md"
###### About the project
- In this project we investigate an unsupervised approach to classifying biological signals from a mass spectrometry dataset. 

- We focus on comparing three clustering algorithms which include Spectral Clustering (Thresholding Clustering), K-Subspaces(KSS), and Ensemble KSS. 

- Our hypothesis was that Ensemble KSS will yield the best results.

- We used three data preproccesing methods: Original Data, Element-wise Absolute Value, and Element-wise Squaring
###### Data is property of Zeteo Tech Inc and results from this project should not be published. 

"

# ╔═╡ 8c5e5e6c-56f0-4734-aeb4-fe066d35a849
md"
**Activating the directory and installing necessary packages**
"

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

# ╔═╡ 16e69f05-fa6f-49af-ab18-950dc3763061
md"
**Choose the preprocessing method in the drop-down menu below**
"

# ╔═╡ 3d02e7c2-809b-4fea-b6dd-2b57e80bd9fd
@bind Preprocessing Select(["Original", "Absolute", "Squaring"])

# ╔═╡ 0192f614-0739-4a64-94eb-b168b9b27023
Data = [permutedims(npzread(path)) for path in file_paths]

# ╔═╡ 6acf8b8d-77dc-4da9-9fa1-32d31f2d17be
md"""
**Data Matrix for all the .npy files from the agents A,B,C,D**
"""

# ╔═╡ a33479ce-23be-450f-86e3-943a895116fe
D_Preprocessing = Dict("Original" => hcat(Data...), "Absolute" =>abs.(hcat(Data...)), "Squaring" => hcat(Data...).^2)

# ╔═╡ 783f183b-f14a-44d5-a22f-dd53b0667f2e
D = D_Preprocessing[Preprocessing]

# ╔═╡ d549356f-fcac-49e0-8f60-852d2b081fba
md"""
#### Data Visualizations
"""

# ╔═╡ c3038511-149c-4235-b95a-d157ecfb4394
with_theme() do
	fig = Figure(; size=(1300, 700))

	#Setting up the Gridlayouts one for each Agent
	grid1 = GridLayout(fig[1, 1]; nrow=3, ncol=1)
    grid2 = GridLayout(fig[1, 2]; nrow=3, ncol=1)
	grid3 = GridLayout(fig[1, 3]; nrow=3, ncol=1)
	grid4 = GridLayout(fig[1, 4]; nrow=3, ncol=1)

	if Preprocessing == "Original"
		vmin,vmax=0,3
	
	elseif Preprocessing == "Absolute"
		vmin,vmax=0,10
	
	else
		vmin,vmax=0,100
		
	end
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
	linkyaxes!(ax1,ax2,ax3,ax4)

	# Display the figure
	fig
end

# ╔═╡ f83c400c-7c6a-40ec-9784-a690506367de
md"""
### Spectral Clustering
"""

# ╔═╡ 7efcff6a-630f-49aa-bfb9-64c242a5a526
begin 
	col_norms = [norm(D[:, i]) for i in 1:size(D, 2)]
	Norm_vec = [D[:, i] ./ col_norms[i] for i in 1:size(D, 2)]
	Norm_mat = hcat(Norm_vec...);
	n_clusters = 4;
	A = transpose(Norm_mat) * Norm_mat
end

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

# ╔═╡ 45cda150-03ec-4a25-9135-c037eae1e817
md"
**Results for spectral clustering**
"

# ╔═╡ 1192b8a0-7d4f-45a2-a506-70846ae0d206
SC_Results = spec_clusterings[1].assignments

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

# ╔═╡ 275b6ebb-63bf-4067-9aca-90da9f7abfc7
md"
**Results from K-Subspaces (KSS)**
"

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
Co_Clustering = 
begin 
	cache(joinpath(CACHEDIR,"co_clustering_$Preprocessing.bson")) do
	Co_Clu = begin
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
	end
end

# ╔═╡ 8e653298-c048-4f90-8954-58e380f6fc2d
V_Ensemble = embedding(Co_Clustering, n_clusters)

# ╔═╡ 07999787-dfcb-41b6-9125-201d5b307923
EKSS_clusterings = batchkmeans(permutedims(V_Ensemble), n_clusters; maxiter=1000)

# ╔═╡ bdc49ad4-511c-4090-8bd8-288a6afd8b4f
md"
**Results from Ensemble K-Subspaces (EKSS)**
"


# ╔═╡ 6bf31f57-d627-42cc-af6a-d4088e87edfe
EKSS_Results = EKSS_clusterings[1].assignments

# ╔═╡ d8a0e4a4-0af6-4842-9a99-14b5df4b36ea
md"
### Results
"

# ╔═╡ 8cb033bd-16f2-4da4-90c9-fddc7d4733b1
function get_Confusion_matrix(results)
	confusion_matrix = zeros(4,4)
	groupings = reshape(results,500,4)'
	for j in 1:4
		for i in 1:size(groupings)[2]
			confusion_matrix[j,groupings[j,:][i]]+=1
		end
	end
	return confusion_matrix / size(groupings)[2] *100
end

# ╔═╡ b7d8c9e5-434a-4773-80ae-f056abe1ecf0
function plot_confusion_matrix(SC_Results,KSS_Results,EKSS_Results)
    SC_confusion_matrix = get_Confusion_matrix(SC_Results)
	KSS_confusion_matrix = get_Confusion_matrix(KSS_Results)
    EKSS_confusion_matrix = get_Confusion_matrix(EKSS_Results)

	# Initialize a figure with size
	fig = Figure(size = (800,400))
	Label(fig[0, 1:3], "Data Preprocessing - $Preprocessing", fontsize = 20, halign = :center,valign = :center)

	# Create axes for each subplot
	ax1 = Axis(fig[1, 1],title = "Spectral Clustering",
    xlabel = "Agent",ylabel = "Predicted Cluster",
	xticks = (1:4, ["A", "B", "C", "D"]), yticks = (1:4, ["1", "2", "3", "4"]) )
	hm1=heatmap!(ax1, SC_confusion_matrix, colormap = :viridis)
	Colorbar(fig[2, 1], hm1,  vertical = false)

	for j in 1:4, i in 1:4
		value = round((SC_confusion_matrix)[i, j], digits=2)
		text!(ax1, i-0.01, j - 0.10, text = "$value", color=:black, align = (:center, :center), fontsize=13)
	end
	
	ax2 = Axis(fig[1, 2],title = "K-Subspaces (KSS)",
    xlabel = "Agent",ylabel = "Predicted Cluster",
	xticks = (1:4, ["A", "B", "C", "D"]), yticks = (1:4, ["1", "2", "3", "4"]) )
	hm2=heatmap!(ax2,KSS_confusion_matrix, colormap = :viridis)
	Colorbar(fig[2, 2], hm2,  vertical = false)

	for j in 1:4, i in 1:4
		value = round((KSS_confusion_matrix)[i, j], digits=2)
		text!(ax2, i-0.01, j - 0.10, text = "$value", color=:black, align = (:center, :center), fontsize=13)
	end
		
	ax3 = Axis(fig[1, 3],title = "Ensemble KSS (EKSS)",
    xlabel = "Agent",ylabel = "Predicted Cluster",
	xticks = (1:4, ["A", "B", "C", "D"]), yticks = (1:4, ["1", "2", "3", "4"]) )
	hm3=heatmap!(ax3, EKSS_confusion_matrix, colormap = :viridis)
	Colorbar(fig[2, 3], hm3,  vertical = false)

	for j in 1:4, i in 1:4
		value = round((EKSS_confusion_matrix)[i, j], digits=2)
		text!(ax3, i-0.01, j - 0.10, text = "$value", color=:black, align = (:center, :center), fontsize=13)
	end
	
	fig  # Display the figure
end

# ╔═╡ afede35a-eeca-4273-895b-b80e01e076b0
plot_confusion_matrix(SC_Results,KSS_Results,EKSS_Results)

# ╔═╡ ecd896e8-e2a7-4d0c-8ec3-aa3d0d318781
function plot_coclustering_matrix(co_clustering_matrix)
    

	# Initialize a figure with size
	fig = Figure(size = (400,400))
	
	# Create axes for each subplot
	ax1 = Axis(fig[1, 1],title = "EKSS Co-Clustering Matrix",
    xlabel = "Agents",ylabel = "Agents" ,xticks = (250:500:1750, ["A", "B", "C", "D"])
	,yticks = (250:500:1750, ["A", "B", "C", "D"])  )
	hm1=heatmap!(ax1, co_clustering_matrix, colormap = :viridis)
	Colorbar(fig[1,2], hm1,  vertical = true)
	
	fig  # Display the figure
end

# ╔═╡ f24de622-78a9-492f-b355-8c91a5d83b42
plot_coclustering_matrix(Co_Clustering)

# ╔═╡ 848efb01-827d-4ae6-b80b-4101546e0853
md"
#### Threading for Acceleration

- To accelerate the KSS algorithm for KSS and EKSS we used the Base.Threads library
- We used 16 threads with 16 cpu cores

	Run Times for 10 runs of EKSS :
		(Without Threads)  4457 seconds
		(With Threads)      962 seconds

	Below are the functions with Threads for acceleration
"

# ╔═╡ 3ca236ff-bd38-4bd5-b820-467fee4fb35a
function accel_KSS(X, d; niters=100, Uinit=polar.(randn.(size(X, 1), collect(d))))
    K = length(d)
    D, N = size(X)
    # Initialize
    U = deepcopy(Uinit)
    c = [argmax(norm(U[k]' * view(X, :, i)) for k in 1:K) for i in 1:N]
    c_prev = copy(c)
    # Iterations
    @progress for t in 1:niters
        # Update subspaces
        @threads for k in 1:K
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
        @threads for i in 1:N
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

# ╔═╡ 1b07c0b1-d128-4add-a8d4-5a22218014f8
function accel_batch_KSS(X, d; niters=100, nruns=10)
    D, N = size(X)
    runs = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Int}, Float64}}(undef, nruns)
    @threads for idx in 1:nruns
        log_message(string(idx))
        U, c = cachet(joinpath(CACHEDIR, "Squaring", "run-$idx.bson")) do
            Random.seed!(idx)
            KSS(X, d; niters=niters)
        end
        total_cost = 0
        @threads for i in 1:N
            cost = norm(U[c[i]]' * view(X, :, i))
            total_cost += cost
        end
        runs[idx] = (U, c, total_cost)
    end
     return runs
end

# ╔═╡ Cell order:
# ╟─4b89c903-e5d9-4066-b5aa-e6ba8712e056
# ╟─2b3a9820-15bd-4215-b723-8184e59e3bf4
# ╟─dbe2c43a-7f5d-443b-9253-dc1c630cdb15
# ╟─8c5e5e6c-56f0-4734-aeb4-fe066d35a849
# ╟─57783977-c0f5-43b0-b43c-01d51e5a37cf
# ╠═1b187d2e-958f-42a5-a266-6f0dd1847a6b
# ╠═323391e7-2911-455e-813c-4f1b09aa02f6
# ╠═2fde9133-c6c0-4b13-bb76-b7dcf3f91c59
# ╟─16e69f05-fa6f-49af-ab18-950dc3763061
# ╠═3d02e7c2-809b-4fea-b6dd-2b57e80bd9fd
# ╠═0192f614-0739-4a64-94eb-b168b9b27023
# ╟─6acf8b8d-77dc-4da9-9fa1-32d31f2d17be
# ╠═a33479ce-23be-450f-86e3-943a895116fe
# ╠═783f183b-f14a-44d5-a22f-dd53b0667f2e
# ╟─d549356f-fcac-49e0-8f60-852d2b081fba
# ╟─c3038511-149c-4235-b95a-d157ecfb4394
# ╟─65320198-78e1-442d-9ca3-5029393e20c2
# ╟─f83c400c-7c6a-40ec-9784-a690506367de
# ╠═7efcff6a-630f-49aa-bfb9-64c242a5a526
# ╟─e0bec9ff-0442-4134-bf4e-636ec9bf6eeb
# ╠═3ffbeab0-c815-43f1-97d4-441969bea6b1
# ╟─cf041b97-409f-4c30-96bb-40ed0f1ec441
# ╠═d1e78fd8-3403-40b2-9acf-43204caf5bf1
# ╟─45cda150-03ec-4a25-9135-c037eae1e817
# ╠═1192b8a0-7d4f-45a2-a506-70846ae0d206
# ╟─e17c9163-7b1d-4672-9846-b526c9f24343
# ╠═0dfa6d32-6689-4c05-8c25-9a2f6c932f63
# ╠═5843384d-ee11-4a04-9c52-a33efc527783
# ╠═a8c2cb19-febb-40ea-9766-d190c4ccda74
# ╠═077b8ae4-79ed-4806-8a78-63c58f07c882
# ╠═16a77379-f74d-4e7f-b175-8e9ed2a02965
# ╠═7a2b7ea0-3ae6-4c3d-b1ba-e4c1bd3828b3
# ╠═5c31fed2-e41a-4444-b886-22da7fecd7fc
# ╠═689d4ad3-31a2-425c-b6fe-69983797fac7
# ╟─275b6ebb-63bf-4067-9aca-90da9f7abfc7
# ╠═5f6480e8-36a5-4a6b-9aa3-4a5cddf0b6e8
# ╟─30918205-73f4-44a2-8f7d-6457378c2996
# ╠═c2b1858d-c0a8-4ed7-8149-1e0634c1901e
# ╟─73df4b78-a703-44cd-875c-536538225070
# ╠═50367ac4-71d5-405a-b2c9-ebd9717e920f
# ╠═29af46ef-91fc-4fb9-8fa4-0cc02b311b30
# ╠═196a6ad0-13a3-4678-8f5e-9212d8be46a3
# ╠═8e653298-c048-4f90-8954-58e380f6fc2d
# ╠═07999787-dfcb-41b6-9125-201d5b307923
# ╟─bdc49ad4-511c-4090-8bd8-288a6afd8b4f
# ╠═6bf31f57-d627-42cc-af6a-d4088e87edfe
# ╟─d8a0e4a4-0af6-4842-9a99-14b5df4b36ea
# ╟─8cb033bd-16f2-4da4-90c9-fddc7d4733b1
# ╟─b7d8c9e5-434a-4773-80ae-f056abe1ecf0
# ╠═afede35a-eeca-4273-895b-b80e01e076b0
# ╟─ecd896e8-e2a7-4d0c-8ec3-aa3d0d318781
# ╠═f24de622-78a9-492f-b355-8c91a5d83b42
# ╟─848efb01-827d-4ae6-b80b-4101546e0853
# ╠═3011a102-ce11-4df1-8a1c-2624c3920430
# ╠═3ca236ff-bd38-4bd5-b820-467fee4fb35a
# ╠═1b07c0b1-d128-4add-a8d4-5a22218014f8
