# Packages used 
using Statistics, LinearAlgebra, Distributions
using Makie, CairoMakie, Splines2, ColorSchemes
using CSV, DataFrames, StatsBase
using NLSolversBase
using GLMakie
using Random
using LaTeXStrings
using Distributions
using Optim
using JLD2

# include functions from helper files
include("meta_analysis.jl")
include("gmodeling.jl")


# Define color scheme
cols = vcat(ColorSchemes.tol_vibrant...)


# Set simulation specifics
# Number of studies 
k = [10, 30, 100]
# standard deviation of prior distribution
taus = [sqrt(.01), sqrt(.05), sqrt(.1)]
# prior distributions
dist = [Normal(.5, taus[1]), Normal(.5, taus[2]), Normal(.5, taus[3])]
# Degree of natural spline basis
degree = [5] 
# Number of iterations
iter = 100
# Different penalty factors 
c0 = [0.05, 0.2, 0.6]


# Allocate three-dimensional empty array that stores named tuples 
# (i.e., type of output returned by `sim_gmeta`). 
# Stores results of sim_gmeta for every iteration x N x dist combination. 

const rnglist = [MersenneTwister() for i in 1:Threads.nthreads()]
res = Array{NamedTuple, 4}(undef, length(k), length(dist), length(degree), iter)
for j in eachindex(k)
  for m in eachindex(dist)
    for l in eachindex(degree)
      Threads.@threads for i in 1:iter
        # Get assurance that julia is doing something
        print(string(j, ",", m, ",", l, ",", i, "\r")) 
        res[j,m,l,i] = 
          GModeling.sim_gmeta(dist[m], k[j]; degree = degree[l], 
            # c0 = c0[m], 
            iteration = i, rng = rnglist[Threads.threadid()],
            grid = collect(LinRange(-1,1.5,100)))
      end
    end
  end
end


# Save to HDF5 Format
@save "../data/res.jld2" res

# load results
@load "../data/res.jld2"



# Obtain estimates for prior mean and variance
prior_moments = Array{Vector{Float64}, 4}(undef, length(k), length(dist), length(degree), iter)
for j in eachindex(k)
  for m in eachindex(dist)
    for l in eachindex(degree)
      for i in 1:iter 
        prior_prob = res[j,m,l,i].g_corrected
        support = res[j,m,l,i].grid 
        mu_hat = support'*prior_prob
        tau2_hat = ((support .- mu_hat) .^2)'*prior_prob
        prior_moments[j,m,l,i] = [mu_hat,tau2_hat,k[j],scale(dist[m])^2,degree[l],i]
      end
    end
  end
end

# Store results in data frame (allows for easier processing)
prior_moments_df = [
  begin
DataFrame(
  hcat(prior_moments[j,m,l,:]...)', 
       [:mu_hat, :tau2_hat, :N, :tau2, :df, :iteration]
) 
end for j in eachindex(k), m in eachindex(dist), l in eachindex(degree)
]


vcat([mean(Matrix(prior_moments_df[j,m,l]); dims = 1) for j in eachindex(k), m in eachindex(dist), l in eachindex(degree)]...)


#= Figure 1: Estimated Priors =#
# Grid of 'true' prior distributions to be plotted 
true_prior = reshape(repeat(dist, inner = length(k)), length(k), length(dist))

# Activate plotting device (want to create pdf) 
CairoMakie.activate!() 
# Set theme 
set_theme!(theme_light(); 
  resolution = (600, 600), fontsize = 10,
  fonts = (; regular = "Latin Modern Roman", bold = "Latin Modern Roman"))

# Initiate Figure
f1 = Figure() 

# Define axes in f1 (3 times 3 grid)
ax = [Axis(f1[j,m:(m+2)]; 
      xlabel = L"\theta", 
      ylabel = "", 
      yticks = (collect(2:2:6), string.(collect(2:2:6))), 
      yticksvisible = false) 
      for j in eachindex(k), m in collect(1:3:(3*length(dist)))]

# Create plot iteratively 
for j in eachindex(k)
  for m in eachindex(dist)
    for l in eachindex(degree)
      xlims!(ax[j,m], -0.5, 1.5)
      ax[j,m].xticks = [-0.5, 0, 0.5, 1, 1.5]
      # Extract densities from res object and store all iterations in array 
      g_density = hcat([(res[j,m,l,i].g_density) for i in 1:iter]...)

      # Create array of plotting data
      plot_data = hcat(mean(g_density; dims = 2), 
                  [std(g_density[r,:]) for r in axes(g_density,1)])
      qs = [quantile(g_density[r,:], [.025, .975]) for r in axes(g_density,1)]

      # Plot error bars (+/- 1 sd)
      errorbars!(ax[j,m], 
                 res[j,m,l,1].grid, 
                 plot_data[:,1], 
                 plot_data[:,2], 
                 plot_data[:,2], 
                 color = cols[3],
                 linewidth = .5)

      # Add true prior distribution
      lines!(ax[j,m], true_prior[j,m], color = cols[1], linewidth = 2,
             linestyle = :dash) 

      # Add means of estimates
      lines!(ax[j,m], res[j,m,l,1].grid, plot_data[:,1], 
             color = cols[2], linewidth = 1.5)
    end
  end
end

# Add labels on top and set decorations
[ax[1,m].title = string("Normal(", location(dist[m]), ", ", 
                        round(scale(dist[m])^2; digits = 2), ")") 
                        for m in eachindex(dist)]
[hidexdecorations!(ax[j,m], grid = false) 
                  for j in 1:(length(k)-1), m in eachindex(dist)]
[hideydecorations!(ax[j,m], grid = false) 
                  for j in eachindex(k), m in 2:length(dist)]
[xlims!(ax[j,m], [-1,1.5]) for j in eachindex(k), m in eachindex(dist)]
[ylims!(ax[j,m], [0,4.5]) for j in eachindex(k), m in eachindex(dist)]
ax1 = [Axis(f1[j,10], yticksvisible = false, 
      xticksvisible = false, 
      ygridvisible = false, xgridvisible = false, 
      bottomspinevisible = false, topspinevisible = false,
      leftspinevisible = false, rightspinevisible = false,
      limits = (-1,1,-1,1)) for j in eachindex(k)]
hidedecorations!.(ax1)
[text!(ax1[j], [-1], [0], text = latexstring.("\\textit{k} = ", k[j]); 
       justification = :left, rotation = -pi/2) for j in eachindex(k)]
ax2 = [Axis(f1[(1:length(k)),0], 
      yticksvisible = false, 
      xticksvisible = false, 
      ygridvisible = false, xgridvisible = false, 
      bottomspinevisible = false, topspinevisible = false,
      leftspinevisible = false, rightspinevisible = false,
      limits = (-1,1,-1,1)) for j in eachindex(k)]
hidedecorations!.(ax2)
text!(ax2[2], [1], [0], text = "Density"; rotation = pi/2)
elem_degree = [LineElement(color = cols[l], linewidth = 1) 
               for l in eachindex(degree)]

# save plot as figure
Makie.save("../figs/figure1.pdf", f1, px_per_unit = 5)



#= Figure 2: Posterior Quantiles =#
# Allocate four-dimensional array for posterior quantiles for each 
# theta.

gmodel_qs = Array{Matrix{Float64}, 4}(undef, length(k), 
                                         length(dist), length(degree), iter)

for j in eachindex(k)
  for m in eachindex(dist)
    for l in eachindex(degree)
      for i in 1:iter 
        qs = zeros(length(res[j,m,l,i].y), 9)
        for p in axes(qs,1)
          tmp = GModeling.post_draw(res[j,m,l,i].grid, 
                                     exp.(res[j,m,l,i].posterior[:,p]))
          q = quantile(tmp, [.025, .5, .975])
          qs[p,:] = vcat(q, hcat(res[j,m,l,i][[:y, :v]]...)[p,:], 
                         res[j,m,l,i].k, res[j,m,l,i].tau, 
                         res[j,m,l,i].degree, i...)
        end
        print(string(j,",",m,",",l,",",i, "\r"))
        gmodel_qs[j,m,l,i] = qs
      end
    end
  end
end


# Convert to data frame, makes sorting by y easier
gmodel_qs = [DataFrame(gmodel_qs[j,m,l,i], 
                [:lb, :md, :ub, :y, :v, :N, :tau, :degree, :sim]) 
                for j in eachindex(k), 
                m in eachindex(dist), 
                l in eachindex(degree), i in 1:iter]

# Sort by the observations y
sort!.(gmodel_qs, [:y])


# Function for computing mean and variance of theoretical posterior distribution 
# under the assumption that mu and tau are known.
function moments(y, v, mu, tau)
  if abs(tau) > 10^(-16)
    V = sum((1 ./v .+ 1 ./tau^2)) .^(-1)
    m = V * sum(y ./v + mu ./tau^2)
  else 
    V = sum((1 ./v)) .^(-1)
    m = V* sum(y./v)
  end
   return [m V]
end

# Which taus and mus were used?
taus = scale.(dist)
mus = location.(dist)

# Average quantiles at each inverse-variance weighted y-value
gmodel_means = [
  begin
    v = hcat([1 ./ gmodel_qs[j,m,l,i].v for i in 1:iter]...)
    y = hcat([gmodel_qs[j,m,l,i].y for i in 1:iter]...) 
    ybar = vcat([moments(y[i,:],v[i,:],0,0)[:,1] for i in 1:k[j]]...)
    md = mean(hcat([gmodel_qs[j,m,l,i][!,:md] 
              for i in 1:iter]...); dims = 2)
    lb = mean(hcat([gmodel_qs[j,m,l,i][!,:lb] 
              for i in 1:iter]...); dims = 2)
    ub = mean(hcat([gmodel_qs[j,m,l,i][!,:ub] 
              for i in 1:iter]...); dims = 2)
    DataFrame(hcat(lb, md, ub, ybar, gmodel_qs[j,m,l,1].degree), 
              [:lb, :md, :ub, :ybar, :degree])
  end
  for j in eachindex(k), m in eachindex(dist), l in eachindex(degree)
];

# function for computing true posterior quantiles 
function posterior(y, v, mu, tau)
    M = vcat(moments.(y, v, mu, tau)...)
    return hcat([vcat(quantile(Normal(M[r,1], 
                sqrt(M[r,2])), [.025, .5, .975]), 
                y[r], v[r]...) for r in axes(M,1)]...)'
end



# Compute true posterior quantiles 
posterior_qs = [posterior(res[j,m,l,i].y, res[j,m,l,i].v, 
                  mus[m], taus[m]) 
                  for  j in eachindex(k), 
                  m in eachindex(dist), 
                  l in eachindex(degree), i in 1:iter]
posterior_qs = [DataFrame(d, vcat(:lb, :md, :ub, :y, :v)) for d in posterior_qs]
sort!.(posterior_qs, [:y])


posterior_means = [
  begin
    v = hcat([1 ./ posterior_qs[j,m,l,i].v for i in 1:iter]...)
    y = hcat([posterior_qs[j,m,l,i].y for i in 1:iter]...) 
    ybar = vcat([moments(y[i,:],v[i,:],0,0)[:,1] for i in 1:k[j]]...)
    md = mean(hcat([posterior_qs[j,m,l,i][!,:md] 
              for i in 1:iter]...); dims = 2)
    lb = mean(hcat([posterior_qs[j,m,l,i][!,:lb] 
              for i in 1:iter]...); dims = 2)
    ub = mean(hcat([posterior_qs[j,m,l,i][!,:ub] 
              for i in 1:iter]...); dims = 2)
    DataFrame(hcat(lb, md, ub, ybar, repeat(vcat(res[j,m,l,1].degree), k[j])), 
              [:lb, :md, :ub, :ybar, :degree])

  end
  for j in eachindex(k), m in eachindex(dist), l in eachindex(degree)
];
  




# f-modeling quantiles 
fmodel_qs = [
  begin
    print(string(j,",",m,",",i, "\r"))
    temp = MetaAnalysis.reml(res[j,m,l,i].y, res[j,m,l,i].v)
    qs = DataFrame(posterior(res[j,m,l,i].y, res[j,m,l,i].v, 
                   temp[1], sqrt(temp[2])),
                   [:lb, :md, :ub, :y, :v])
  end 
  for j in eachindex(k), m in eachindex(dist), l in eachindex(degree), 
      i in 1:iter
]

sort!.(fmodel_qs, [:y])


fmodel_means = [
  begin
    v = hcat([1 ./ fmodel_qs[j,m,l,i].v for i in 1:iter]...)
    y = hcat([fmodel_qs[j,m,l,i].y for i in 1:iter]...) 
    ybar = vcat([moments(y[i,:],v[i,:],0,0)[:,1] for i in 1:k[j]]...)
    md = mean(hcat([fmodel_qs[j,m,l,i][!,:md] 
              for i in 1:iter]...); dims = 2)
    lb = mean(hcat([fmodel_qs[j,m,l,i][!,:lb] 
              for i in 1:iter]...); dims = 2)
    ub = mean(hcat([fmodel_qs[j,m,l,i][!,:ub] 
              for i in 1:iter]...); dims = 2)
    DataFrame(hcat(lb, md, ub, ybar, repeat(vcat(res[j,m,l,1].degree), k[j])), 
              [:lb, :md, :ub, :ybar, :degree])
  end for j in eachindex(k), m in eachindex(dist), l in eachindex(degree)
]




# Plot 
CairoMakie.activate!() 
set_theme!(theme_light(); resolution = (600, 600), linewidth = 2.5, fontsize = 12,
  fonts = (; regular = "Latin Modern Roman", bold = "Latin Modern Roman"))

f2 = Figure() # Initiate Figure
# Define axes in f1 
ax = [Axis(f2[j,m:(m+2)]; xlabel = L"\hat{\theta}", ylabel = "") 
      for j in eachindex(k), m in collect(1:3:(3*length(dist)))]
ax[2,1]. ylabel = "0.025, 0.5 and 0.975 Posterior Quantiles"
[ax[1,m].title = string("Normal(", location(dist[m]), ", ", 
                        round(scale(dist[m])^2; digits = 2), ")") 
                        for m in eachindex(dist)]

for j in eachindex(k)
  for m in eachindex(dist)
   for l in eachindex(degree)
     ylims!(ax[j,m], [-0.5,1.5])
     xlims!(ax[j,m], [-1,2])

     # True posterior
     lines!(ax[j,m], posterior_means[j,m,l].ybar,posterior_means[j,m,l].md, color = cols[1],  linestyle = :dashdot)
     lines!(ax[j,m], posterior_means[j,m,l].ybar,posterior_means[j,m,l].lb, color = cols[1],  linestyle = :dashdot)
     lines!(ax[j,m], posterior_means[j,m,l].ybar,posterior_means[j,m,l].ub, color = cols[1],  linestyle = :dashdot)

     # f-modeling posterior
     lines!(ax[j,m], fmodel_means[j,m,l].ybar,fmodel_means[j,m,l].md, color = cols[3], linestyle = :dash)
     lines!(ax[j,m], fmodel_means[j,m,l].ybar,fmodel_means[j,m,l].lb, color = cols[3], linestyle = :dash)
     lines!(ax[j,m], fmodel_means[j,m,l].ybar,fmodel_means[j,m,l].ub, color = cols[3], linestyle = :dash)

     # g-modeling posterior
     lines!(ax[j,m], gmodel_means[j,m,l].ybar,gmodel_means[j,m,l].md, color = cols[2])
     lines!(ax[j,m], gmodel_means[j,m,l].ybar,gmodel_means[j,m,l].lb, color = cols[2])
     lines!(ax[j,m], gmodel_means[j,m,l].ybar,gmodel_means[j,m,l].ub, color = cols[2])
    end
  end
end

# hideydecorations!.(ax); # remove y ticks, etc. (fine-tuning)
[hidexdecorations!(ax[j,m], grid = false) for j in 1:(length(k)-1), m in eachindex(dist)]
[hideydecorations!(ax[j,m], grid = false) for j in eachindex(k), m in 2:length(dist)]
#[xlims!(ax[j,k], [-1,1.5]) for j in eachindex(k), m in eachindex(dist)];
#[ylims!(ax[j,k], [0,4.5]) for j in eachindex(k), m in eachindex(dist)];
ax1 = [Axis(f2[j,10], yticksvisible = false, 
      xticksvisible = false, 
      ygridvisible = false, xgridvisible = false, 
      bottomspinevisible = false, topspinevisible = false,
      leftspinevisible = false, rightspinevisible = false,
      limits = (-1,1,-1,1)) for j in eachindex(k)]
hidedecorations!.(ax1)
[text!(ax1[j], [-1], [0], text = latexstring.("\\textit{k} = ", k[j]); 
  justification = :left, rotation = -pi/2) for j in eachindex(k)]

elem_method = [LineElement(color = cols[1], linewidth = 2, linestyle = :dashdot), 
               LineElement(color = cols[3], linewidth = 2, linestyle = :dash),
               LineElement(color = cols[2], linewidth = 2)]
legend = Legend(f2[4,5], [elem_method], [["True Posterior", "F-Modeling", "G-Modeling"]], [""])
legend.orientation = :horizontal

save("../figs/figure2.pdf", f2, px_per_unit = 5)


# Assess performance 
performance = Array{Vector, 4}(undef, length(k), length(dist), 
                               length(degree), iter)
for j in eachindex(k)       
  for m in eachindex(dist)
   for l in eachindex(degree)
    for i in 1:iter   
      g_corrected = res[j,m,l,i].g_corrected
      grid = res[j,m,l,i].grid 
      g_exp = grid'*g_corrected 
      g_tau2 = ((grid .- g_exp) .^2)'*g_corrected
      g_mu = MetaAnalysis.mu_hat(g_tau2, res[j,m,l,i].y, res[j,m,l,i].v)
      ml = MetaAnalysis.mle(vcat(res[j,m,l,i][:y]...), vcat(res[j,m,l,i][:v]...))
      se_ml = GModeling.se(res[j,m,l,i].v, ml[2])
      reml = MetaAnalysis.reml(vcat(res[j,m,l,i][:y]...), vcat(res[j,m,l,i][:v]...))  
      se_reml = GModeling.se(res[j,m,l,i].v, reml[2])
      se_g = GModeling.se(res[j,m,l,i].v, g_tau2)

      performance[j,m,l,i] = [
        g_mu, g_tau2, g_tau2 < 10^(-5) , ml[1], ml[2], ml[2] <10^(-5), reml[1], reml[2], reml[2] < 10^(-5), 
          se_ml, se_reml, se_g, k[j], scale(dist[m])^2, degree[l]
      ]
    end
    end
  end
end

performance = [hcat([performance[j,m,l,i] for i in 1:iter]...)' for j in eachindex(k), m in eachindex(dist), l in eachindex(degree)];
perf_df = [DataFrame(performance[j,m,l], 
[:eb_mu, :eb_tau2, :eb_zero, :ml_mu, :ml_tau2, :ml_zero, :reml_mu, :reml_tau2, :reml_zero, 
    :ml_se, :reml_se, :eb_se, :N, :tau2_true, :degree]) for j in eachindex(k), m in eachindex(dist), l in eachindex(degree)];


boundary_df = DataFrame(zeros(0,5), [:tau2, :eb, :ml, :reml, :k])
rmse_df = DataFrame(zeros(0,5), [:tau2, :eb, :ml, :reml, :k]) 
bias_tau2_df = DataFrame(zeros(0,5), [:tau2, :eb, :ml, :reml, :k])  
coverage_df = DataFrame(zeros(0,5), [:tau2, :eb, :ml, :reml, :k])


for j in eachindex(k)
  for m in eachindex(dist)
    for l in eachindex(degree)
      # Proportion of Boundary estimates 
      push!(boundary_df, hcat(mean(Matrix(perf_df[j,m,l][!, [:tau2_true, :eb_zero, :ml_zero, :reml_zero]]); dims = 1), k[j]))

      # RMSE
      rmse = sqrt.(mean((Matrix(perf_df[j,m,l][!, [:eb_tau2, :ml_tau2, :reml_tau2]] .- 
        perf_df[j,m,l].tau2_true) .^2); dims = 1))
      push!(rmse_df, hcat(perf_df[j,m,l].tau2_true[1], rmse, k[j]...))

      # Bias 
      bias_tau2 = mean(Matrix(perf_df[j,m,l][!, [:eb_tau2, :ml_tau2, :reml_tau2]] .- perf_df[j,m,l].tau2_true); dims = 1)
      push!(bias_tau2_df, hcat(perf_df[j,m,l].tau2_true[1], bias_tau2, k[j]...))

      # Coverage 
      ml_coverage = mean(perf_df[j,m,l].ml_mu .- 1.96 .*perf_df[j,m,l].ml_se .<= 
                     0.5 .<= 
                     perf_df[j,m,l].ml_mu .+ 1.96 .*perf_df[j,m,l].ml_se)

      reml_coverage = mean(perf_df[j,m,l].reml_mu .- 1.96 .*perf_df[j,m,l].reml_se .<= 
                     0.5 .<= 
                     perf_df[j,m,l].reml_mu .+ 1.96 .*perf_df[j,m,l].reml_se)

      eb_coverage = mean(perf_df[j,m,l].eb_mu .- 1.96 .*perf_df[j,m,l].eb_se .<= 
                     0.5 .<= 
                     perf_df[j,m,l].eb_mu .+ 1.96 .*perf_df[j,m,l].eb_se)

      push!(coverage_df, hcat(perf_df[j,m,l].tau2_true[1], eb_coverage, ml_coverage, reml_coverage, k[j]...))
    end
  end
end


set_theme!(theme_light(); 
           resolution = (600, 600), 
           linewidth = 1, 
           fontsize = 10,
           fonts = (; regular = "Latin Modern Roman", bold = "Latin Modern Roman"))
CairoMakie.activate!() 
f_perf = Figure()
measure = ["Proportion of \n Zero-Variance Estimates", "RMSE", "Bias", "Coverage \n Probability"]
ax = [Axis(f_perf[p,j]; ylabel = "", xticks = [0,0.01,0.05,0.1]) for p in 1:4, j in eachindex(k)]
[ax[1,j].title = latexstring.("\\textit{k} = ", k[j]) for j in eachindex(k)]
[ax[p, 1].ylabel = measure[p] for p in 1:4]
hideydecorations!.(ax[1:4, 2:3]; grid = false, label = false)
hidexdecorations!.(ax[1:3, 1:3]; grid = false)
[ax[4,j].xticks = [0, 0.01, 0.05, 0.1] for p in 1:4, j in eachindex(k)]
[ax[4,j].xlabel = L"\tau^2" for p in 1:4, j in eachindex(k)]

for j in eachindex(k)
  for m in eachindex(dist)
    for l in eachindex(degree)
      # plot boundary estimates 
      boundary_temp = boundary_df[boundary_df.k .== k[j], :]
      ylims!(ax[1,j], [-0.05, 0.8])
      scatterlines!(ax[1, j], boundary_temp.tau2, boundary_temp.eb, color = cols[2], markersize = 10)
      scatterlines!(ax[1, j], boundary_temp.tau2, boundary_temp.ml, color = cols[1], marker = :x, markersize = 10)
      scatterlines!(ax[1, j], boundary_temp.tau2, boundary_temp.reml, color = cols[3], marker = :rect, markersize = 10)

      # plot rmse 
      rmse_temp = rmse_df[rmse_df.k .== k[j], :]
      ylims!(ax[2,j], [0, 0.12])
      scatterlines!(ax[2, j], rmse_temp.tau2, rmse_temp.eb, color = cols[2], markersize = 10)
      scatterlines!(ax[2, j], rmse_temp.tau2, rmse_temp.ml, color = cols[1], marker = :x, markersize = 10)
      scatterlines!(ax[2, j], rmse_temp.tau2, rmse_temp.reml, color = cols[3], marker = :rect, markersize = 10)

      # plot bias for tau^2
      ylims!(ax[3,j], [-0.025, 0.04])
      ax[3,j].yticks = [-0.02, 0., 0.02, 0.04]
      bias_tau2_temp = bias_tau2_df[bias_tau2_df.k .== k[j], :]
      hlines!(ax[3,j], [0], linestyle = :dash, color = :black, linewidth = 1)
      scatterlines!(ax[3, j], bias_tau2_temp.tau2, bias_tau2_temp.eb, color = cols[2], markersize = 10)
      scatterlines!(ax[3, j], bias_tau2_temp.tau2, bias_tau2_temp.ml, color = cols[1], marker = :x, markersize = 10)
      scatterlines!(ax[3, j], bias_tau2_temp.tau2, bias_tau2_temp.reml, color = cols[3], marker = :rect, markersize = 10)

      # plot coverage of confidence intervals for mu
      coverage_temp = coverage_df[coverage_df.k .== k[j], :]
      ylims!(ax[4,j], [0.83,1])
      ax[4,j].yticks = [0.85, 0.9, 0.95, 1]
      hlines!(ax[4,j], [.95], linestyle = :dash, color = :black, linewidth = 1)
      scatterlines!(ax[4, j], coverage_temp.tau2, coverage_temp.eb, color = cols[2], markersize = 10)
      scatterlines!(ax[4, j], coverage_temp.tau2, coverage_temp.ml, color = cols[1], marker = :x, markersize = 10)
      scatterlines!(ax[4, j], coverage_temp.tau2, coverage_temp.reml, color = cols[3], marker = :rect, markersize = 10)
    end
  end
end

elem_method = [[MarkerElement(color = cols[2], marker = :circ),
                  LineElement(color = cols[2], linewidth = 2)], 
[MarkerElement(color=cols[1], marker = :x), LineElement(color = cols[1], linewidth = 2)],
[MarkerElement(color=cols[3], marker = :rect), LineElement(color = cols[3], linewidth = 2)]]
legend = Legend(f_perf[5,2], [elem_method], [["G-Modeling", "ML", "REML"]], [""])
legend.orientation = :horizontal

save("../figs/figure3.pdf", f_perf, px_per_unit = 5)


# Additional plot showing example of boundary estimate.

# boundary_example = GModeling.sim_data(5, Normal(0,0.2))
# @save "./data/boundary_example.jdl2" boundary_example

@load "../data/boundary_example.jld2"

tau2 = LinRange(-0.2,0.5,200)
pll = [
  try 
    -MetaAnalysis.pll!([0], nothing, tau2[t], boundary_example.y, boundary_example.v)
    catch e
   if isa(e, DomainError)
    missing
   end
end for t in eachindex(tau2)] 


CairoMakie.activate!() 

# Set theme 
set_theme!(theme_light(); 
  resolution = (600, 300), fontsize = 12,
  fonts = (; regular = "Latin Modern Roman", bold = "Latin Modern Roman"))

# Initiate Figure
f_boundary = Figure()

# Define axes in f1 (3 times 3 grid)
ax = Axis(f_boundary[1,1]; 
      xlabel = L"\tau^2", 
      ylabel = "Profile Log-Likelihood", 
      yticksvisible = false)
ind = .!ismissing.(pll)
ax.xticks = [-0.05, 0., 0.1, 0.2, 0.3, 0.4]
lines!(ax, vcat(tau2[ind]...), vcat(pll[ind]...), color = cols[2], linewidth = 2)
vlines!(ax, [0], linestyle = :dash, color = :black, linewidth = 2)
hideydecorations!(ax, label = false)
save("../figs/figure_boundary.pdf", f_boundary, px_per_unit = 5)





