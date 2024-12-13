### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ c5b0ad20-b81c-11ef-0cf3-e3e4066ee84a
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ f4d22f6e-5c6c-4d72-bf0c-ba6457c5c9ad
using LinearAlgebra, NPZ, PlutoUI, CairoMakie, Statistics, ProgressLogging, CacheVariables, Dates, Random, ArnoldiMethod, Logging, Clustering

# ╔═╡ faccf8aa-4ef7-4b18-acab-db6afb7358d4
md"""
# Ensemble K-Subpsaces
#### ELEG667 Matrix and Tensor Methods
##### - Kyle Regan
##### - Christian Newman-Sanders
##### - Cristy Mathey
"""

# ╔═╡ b8fd5c7f-b526-41d2-99c3-3c8ead68801b
md"
###### About the project
- In this project we investigate an unsupervised approach to classifying biological signals from a mass spectrometry dataset. 

- We focus on comparing three clustering algorithms which include Spectral Clustering (Thresholding Clustering), K-Subspaces(KSS), and Ensemble KSS. 

- Our hypothesis was that Ensemble KSS will yield the best results.

###### Data is property of Zeteo Tech Inc and results from this project should not be published. 

"

# ╔═╡ 36804857-d9d4-4aa5-b626-9410f8a4a8df
md"
**Activating the directory and installing necessary packages**
"

# ╔═╡ 064b00d4-402f-4cfe-9faa-b37878478846
md"
##### Functions
"

# ╔═╡ 05e27d7f-8c2c-4006-912c-1a6f3f33d7c3
md"
**Load data from the data directory**
"

# ╔═╡ 97a0cf74-9978-4af7-b535-91cc052ef9e2
data_directory = joinpath(@__DIR__, "data", "Datasets")

# ╔═╡ f4ae61c4-165e-4d77-8d5b-87d8ab63fe90
dataset_types = readdir(data_directory)

# ╔═╡ 63c81574-47d1-4760-9fb4-19957d1578fc
data_dir = joinpath(data_directory,dataset_types[3])

# ╔═╡ 84a7526a-24c7-4301-90d4-d410830aa472
file_names = ["A.npy", "B.npy", "C.npy", "D.npy"]#, "Noise.npy"]

# ╔═╡ a8aca3c8-90ac-48a2-85af-60e3d8e894c6
file_paths = [joinpath(data_dir, file_name) for file_name in file_names]

# ╔═╡ f5be85b7-e7df-4e18-97c9-c658507b0f9b
Data = [permutedims(npzread(path)) for path in file_paths]

# ╔═╡ 494b6f3d-74d2-48bd-8e1e-4de59856607a
D_orig = hcat(Data...)

# ╔═╡ 05b0acaf-c923-4f33-8a19-aab8e5d590a2
md"
**D is the data matrix containing 2000 signals each of length 2491. There are 4 classes each with 500 signals which make up the 2000 signals**
"

# ╔═╡ dbec1b17-55f3-4e51-a549-375a1dc5c7d7
#D= D_orig.* (D_orig .> 0) #relu

D = abs.(D_orig) #abs

D = D_orig.^2 #squaring

# ╔═╡ cfa165b8-b93d-4a4c-9bdd-ba72a7555f37
# CACHEDIR = splitext(relpath(@__FILE__))[1]

# ╔═╡ 36acedfc-6387-40f6-bf8c-8e7747c1a642
col_norms = [norm(D[:, i]) for i in 1:size(D, 2)]

# ╔═╡ bc93313e-4c03-42ed-a5b2-9dcbbd81676a
Norm_vec = [D[:, i] ./ col_norms[i] for i in 1:size(D, 2)]

# ╔═╡ d902515a-3e7f-407c-8fe0-80348ba59c75
Norm_mat = hcat(Norm_vec...)

# ╔═╡ ffb3ca55-5430-4371-8c1c-e3c19b71e4fb
A = transpose(Norm_mat) * Norm_mat

# ╔═╡ c6f9b85b-9aff-457b-a87d-1f5cbee148a8
S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

# ╔═╡ 02623902-8fc8-45df-921b-fe18733be09b
diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ 043d1a06-0b35-424a-9dc3-2064ebea6d92
D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ 1b16c05e-a6ab-4e10-b90c-85cb105552b9
L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ fd9e8f34-8751-4ea1-82cb-a83ab1d5b578
n_clusters = 4

# ╔═╡ 7c9eada0-8d4d-4463-9464-8ce221131a7d
decomp, history = partialschur(L_sym; nev=n_clusters, which=:SR)

# ╔═╡ 6499d67e-5c74-4723-b5a5-688352aa8051
V = mapslices(normalize, decomp.Q; dims=2)

# ╔═╡ 24596262-752b-455e-a212-25ac3a491d4f
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

# ╔═╡ d62de40f-c270-42b1-9c3a-1ef0fd295a93
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 74dfaa85-5725-4d02-972c-ef110d84784e
SC_Results = spec_clusterings[1].assignments

# ╔═╡ 97bc4055-b337-4124-bf37-c90a0d0eca0d
matrix_for_spec = copy(D)

# ╔═╡ ffbd5e3e-1cdb-4577-9677-e886a0a6e8e0


# ╔═╡ aeaf904c-f9ef-42ac-b3fc-99dae706a969
groupings = reshape(SC_Results,500,4)'

# ╔═╡ dab0e60d-9775-46c7-94cd-3fed8e3015c1
begin
	#using StatsBase
	
	# Example label matrix (4 classes, 500 samples per class)
	true_labels = repeat(1:4, inner=500)  # True labels: Class 1, Class 2, ...
	# Reshape true labels into a 4x500 matrix
	true_label_matrix = reshape(true_labels, (500, 4))'
	predicted_label_matrix = groupings
	# Initialize confusion matrix
	# Initialize confusion matrix
	confusion_matrix = zeros(Int, 4, 4)
	
	# Populate confusion matrix
	for i in 1:4  # True classes
	    for j in 1:4  # Predicted classes
	        confusion_matrix[i, j] = count(x -> x == j, vec(predicted_label_matrix[i, :]))
	    end
	end
	
	# Display the confusion matrix
	println("Confusion Matrix:")
	println(confusion_matrix)
	
	# Plot the heatmap using CairoMakie
	begin
	
	# Initialize a figure with size
	fig = Figure(size = (900,400))
	
	# Create axes for each subplot
	ax1 = Axis(fig[1, 1],title = "Confusion Matrix",
    xlabel = "Predicted Class",
    ylabel = "True Class",xticks = (1:4, ["1", "2", "3", "4"]),
    yticks = (1:4, ["1", "2", "3", "4"])
)
		
	hm1=heatmap!(ax1, confusion_matrix/500, colormap = :viridis)
		#colorrange = (vmin, vmax
	Colorbar(fig[2, 1], hm1,  vertical = false)
	
end
	
	fig  # Display the figure
end

# ╔═╡ 605e11d6-6956-482e-9b6a-7dca5299c89b
true_label_matrix

# ╔═╡ 4cbeb7bb-16de-4158-b3c7-a929268c010a
groupings

# ╔═╡ 58e98e19-4f46-45a8-a384-5fcafca46a84
confusion_matrix/500

# ╔═╡ 40c6b4b6-c622-45b0-8013-226e91a38e9e
temp = zeros(4,4)

# ╔═╡ fc683845-3b71-49c8-80e2-e85d21ad8790
for j in 1:4
	for i in 1:size(groupings)[2]
		#println("$j $i --> $(groupings[j,:][i])")
		#println("$j , $(groupings[j,:][i]) -->$(temp[j,groupings[j,:][i]] )")
		#println("$(temp[j,groupings[j,:][i]])")
		temp[j,groupings[j,:][i]]+=1
	end
end

# ╔═╡ 5710411a-b119-4801-bbfb-f947720010f7
temp/500

# ╔═╡ 6c8e34cf-5338-4db8-a815-6ddfa8f89562
temp[1,:][groupings[1,:][1]]+=1

# ╔═╡ 65df47bf-4948-43bd-97c8-41ed9ff7cce8
temp

# ╔═╡ 665c0703-ca15-4db2-a68b-69f1e91c8664
function

# ╔═╡ 9452cf1f-c9c4-48b6-b8a7-118d505970c1
function plot_confusion_matrix(labels)#SC_labels,KSS_labels,EKSS_labels)
	confusion_matrix = zeros(4,4)
	groupings = reshape(labels,500,4)'
	for j in 1:4
		for i in 1:size(groupings)[2]
			confusion_matrix[j,groupings[j,:][i]]+=1
		end
	end
	
	
	# Initialize a figure with size
	fig = Figure(size = (800,400))
	
	# Create axes for each subplot
	ax1 = Axis(fig[1, 1],title = "Spectral Clustering",
    xlabel = "Agent",ylabel = "Predicted Cluster",
	xticks = (1:4, ["A", "B", "C", "D"]), yticks = (1:4, ["1", "2", "3", "4"]) )
	hm1=heatmap!(ax1, confusion_matrix/size(groupings)[2]*100, colormap = :viridis)
	Colorbar(fig[2, 1], hm1,  vertical = false)

	for j in 1:4, i in 1:4
		value = round((confusion_matrix/size(groupings)[2])[i, j], digits=2)
		text!(ax1, i-0.01, j - 0.10, text = "$value", color=:black, align = (:center, :center), fontsize=13)
	end
	
	ax2 = Axis(fig[1, 2],title = "K-Subspaces (KSS)",
    xlabel = "Agent",ylabel = "Predicted Cluster",
	xticks = (1:4, ["A", "B", "C", "D"]), yticks = (1:4, ["1", "2", "3", "4"]) )
	hm2=heatmap!(ax2, confusion_matrix/size(groupings)[2], colormap = :viridis)
	Colorbar(fig[2, 2], hm2,  vertical = false)

	for j in 1:4, i in 1:4
		value = round((confusion_matrix/size(groupings)[2])[i, j], digits=2)
		text!(ax2, i-0.01, j - 0.10, text = "$value", color=:black, align = (:center, :center), fontsize=13)
	end
		
	ax3 = Axis(fig[1, 3],title = "Ensemble KSS (EKSS)",
    xlabel = "Agent",ylabel = "Predicted Cluster",
	xticks = (1:4, ["A", "B", "C", "D"]), yticks = (1:4, ["1", "2", "3", "4"]) )
	hm3=heatmap!(ax3, confusion_matrix/size(groupings)[2], colormap = :viridis)
	Colorbar(fig[2, 3], hm3,  vertical = false)

	for j in 1:4, i in 1:4
		value = round((confusion_matrix/size(groupings)[2])[i, j], digits=2)
		text!(ax3, i-0.01, j - 0.10, text = "$value", color=:black, align = (:center, :center), fontsize=13)
	end
	
	fig  # Display the figure
end

# ╔═╡ d75ce31e-7694-4fd5-be4b-7adc011e2ebb
plot_confusion_matrix(SC_Results)

# ╔═╡ Cell order:
# ╟─faccf8aa-4ef7-4b18-acab-db6afb7358d4
# ╟─b8fd5c7f-b526-41d2-99c3-3c8ead68801b
# ╟─36804857-d9d4-4aa5-b626-9410f8a4a8df
# ╟─064b00d4-402f-4cfe-9faa-b37878478846
# ╠═c5b0ad20-b81c-11ef-0cf3-e3e4066ee84a
# ╠═f4d22f6e-5c6c-4d72-bf0c-ba6457c5c9ad
# ╟─05e27d7f-8c2c-4006-912c-1a6f3f33d7c3
# ╟─97a0cf74-9978-4af7-b535-91cc052ef9e2
# ╠═f4ae61c4-165e-4d77-8d5b-87d8ab63fe90
# ╠═63c81574-47d1-4760-9fb4-19957d1578fc
# ╠═84a7526a-24c7-4301-90d4-d410830aa472
# ╠═a8aca3c8-90ac-48a2-85af-60e3d8e894c6
# ╠═f5be85b7-e7df-4e18-97c9-c658507b0f9b
# ╠═494b6f3d-74d2-48bd-8e1e-4de59856607a
# ╟─05b0acaf-c923-4f33-8a19-aab8e5d590a2
# ╠═dbec1b17-55f3-4e51-a549-375a1dc5c7d7
# ╠═cfa165b8-b93d-4a4c-9bdd-ba72a7555f37
# ╠═36acedfc-6387-40f6-bf8c-8e7747c1a642
# ╠═bc93313e-4c03-42ed-a5b2-9dcbbd81676a
# ╠═d902515a-3e7f-407c-8fe0-80348ba59c75
# ╠═ffb3ca55-5430-4371-8c1c-e3c19b71e4fb
# ╠═c6f9b85b-9aff-457b-a87d-1f5cbee148a8
# ╠═02623902-8fc8-45df-921b-fe18733be09b
# ╠═043d1a06-0b35-424a-9dc3-2064ebea6d92
# ╠═1b16c05e-a6ab-4e10-b90c-85cb105552b9
# ╠═fd9e8f34-8751-4ea1-82cb-a83ab1d5b578
# ╠═7c9eada0-8d4d-4463-9464-8ce221131a7d
# ╠═6499d67e-5c74-4723-b5a5-688352aa8051
# ╠═24596262-752b-455e-a212-25ac3a491d4f
# ╠═d62de40f-c270-42b1-9c3a-1ef0fd295a93
# ╠═74dfaa85-5725-4d02-972c-ef110d84784e
# ╠═97bc4055-b337-4124-bf37-c90a0d0eca0d
# ╠═ffbd5e3e-1cdb-4577-9677-e886a0a6e8e0
# ╠═aeaf904c-f9ef-42ac-b3fc-99dae706a969
# ╠═dab0e60d-9775-46c7-94cd-3fed8e3015c1
# ╠═605e11d6-6956-482e-9b6a-7dca5299c89b
# ╠═4cbeb7bb-16de-4158-b3c7-a929268c010a
# ╠═58e98e19-4f46-45a8-a384-5fcafca46a84
# ╠═40c6b4b6-c622-45b0-8013-226e91a38e9e
# ╠═fc683845-3b71-49c8-80e2-e85d21ad8790
# ╠═5710411a-b119-4801-bbfb-f947720010f7
# ╠═6c8e34cf-5338-4db8-a815-6ddfa8f89562
# ╠═65df47bf-4948-43bd-97c8-41ed9ff7cce8
# ╠═665c0703-ca15-4db2-a68b-69f1e91c8664
# ╠═9452cf1f-c9c4-48b6-b8a7-118d505970c1
# ╠═d75ce31e-7694-4fd5-be4b-7adc011e2ebb
