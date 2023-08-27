module GModeling

# Packages used inside Module
using Statistics, LinearAlgebra, Distributions
using Splines2
using StatsBase
using Optim
using NLSolversBase
using Optim
using Random
using DataFrames

"""
  log_g(alpha, grid; degree = 5)

Compute log-density of exponential-family prior distribution. 

Returns vector of log-densities/log-probabilities for 
each value in the discretized support of the prior distribution (called `grid`) 
based on exponential-family distribution with structure matrix Q and natural parameter vector 
alpha. The rows of the structure matrix Q contain natural spline bases 
for each value of the discretized support (called grid) of the prior distribution 
The number of inner knots is equal to `degree` (degree = 5 per default). 
The knots are uniformly-spaced between the min(grid) and max(grid).
Returns log-densities and structure matrix Q.

# Arguments

- `alpha::Vector`: Natural parameter of exponential family distribution (`degree`+1 vector).
- `grid::Vector`:  Discrete support of prior distribution. 
- `degree`: Degree of natural cubic spline
"""
function log_g(alpha::Vector, grid::Vector; degree = 5)
    Q = ns(vcat(grid...); df = degree)
    Q = hcat(ones(axes(Q,1)), Q)        # Add column of 1's
    return (Q*alpha .- log(sum(exp.(Q*alpha))), Q)
end

"""
Compute statistics used for optimizing log-likelihood below.
"""
function ll_components(alpha::Vector, grid::Vector, y::Vector, v::Vector;
                        degree = 5, c0 = 0.05)
    log_dnorm = [logpdf(Normal(t,sqrt(v[i])), y[i]) for i in eachindex(y), t in grid]
    log_dprior, Q = log_g(alpha, grid; degree = degree)
    dmarginal = exp.(log_dnorm) * exp.(log_dprior)
    pred_factor = exp.(log_dnorm) ./ dmarginal
    W = exp.(log_dprior) .* (pred_factor' .- 1)
    score = Q'*W
    l2_norm = sqrt(sum(alpha .^2))
    penalty = c0 .* alpha ./ l2_norm

    return (log_dnorm = log_dnorm, log_dprior = log_dprior, Q = Q, dmarginal = dmarginal, 
            pred_factor = pred_factor, W = W, score = score, l2_norm = l2_norm, 
            penalty = penalty)
end


"""
Evaluate penalized log-likelihood function at specified inputs and compute log-likelihood and gradient. 

# Arguments 
- `F` and `G` are the log-likelihood and the score vector (the gradient of the log-likelihood) respectively. 
- `alpha::Vector`: Natural parameter of exponential family distribution.
- `grid::Vector`: Discrete support of prior distribution.
- `y::Vector`: Vector of estimates of quantity of interest (aka "effect sizes")
- `v::Vector`: Vector of variances of estimates of quantity of interest
- `degree`: Degree of natural cubic spline 
- `c0`: Penalty factor of penalized log-likelihood
"""
function fg!(F, G, alpha::Vector, grid::Vector, y::Vector, v::Vector; 
             degree = 5, c0 = 0.05)
    ll_comp = ll_components(alpha, grid, y, v; degree = degree, c0 = c0)

    if !(G === nothing)
        G .= - sum(ll_comp.score; dims = 2) + ll_comp.penalty
    end
    if !(F === nothing)
        return - sum(log.(ll_comp.dmarginal)) + c0 .* sqrt(sum(alpha .^2))
    end
end

"""
Delta-method estimate of bias and variance-covariance matrix of 
prior distribution 
"""
function bias(alpha, grid, y, v; degree = 5, c0 = 0.05)
    ll_comp = ll_components(alpha, grid, y, v; degree = degree, c0 = c0)
    Wp = vcat(sum(ll_comp.W; dims = 2)...)
    aa = sqrt(sum(alpha .^2))
    sdot = c0 .* alpha ./aa
    sdot2 = c0/aa*(I-alpha*alpha' ./aa^2)
    Ws = zeros(length(Wp), length(Wp))
        for i in axes(ll_comp.W, 2)
            Ws = Ws + ll_comp.W[:,i]*ll_comp.W[:,i]'
        end
    info = ll_comp.Q' * (Ws + Wp*exp.(ll_comp.log_dprior)' + exp.(ll_comp.log_dprior)*Wp' - diagm(Wp)) * ll_comp.Q
    alpha_bias = -inv(info + sdot2) * sdot
    alpha_var = inv(info + sdot2) * info * inv(info + sdot2)
    D = diagm(exp.(ll_comp.log_dprior)) - exp.(ll_comp.log_dprior) * exp.(ll_comp.log_dprior)'
    return (g_bias = D*ll_comp.Q*alpha_bias, 
            g_variance = D * ll_comp.Q * alpha_var * ll_comp.Q' * D)
end



"""
Draw 2000 samples from `support` using `weights`.
"""
function post_draw(support, w)
    w = Weights(w)
    u = sample(support, w, 5000)
    return u
end


"""
Simulate data for meta-analysis. 
# Arguments 
- `k`: Number of studies
- `distribution`: A distribution.  
- `rng`: Random number generator (rng) to be used. Defaults to GLOBAL_RNG.
"""
function sim_data(k, distribution; rng = Random.GLOBAL_RNG)
    v = rand(rng,truncated(0.25 * Chisq(1), .009, .6), k)
    theta = rand(rng,distribution, length(v))
    return (y = [rand(rng, Normal(theta[i], sqrt(v[i]))) 
                 for i in eachindex(theta)], v = v)
end

struct GMeta
    y::Vector 
    v::Vector 
    g_est::Vector 
    grid::Vector 
    g_bias::Vector 
    g_variance::Matrix
    g_mse::Float64
    g_corrected::Vector 
    g_density::Vector 
    posterior::Matrix
    alpha::Vector 
    structure::Matrix
end


"""
Function for estimating prior and posteriors over values in grid 
given observations y and variances v. c0 is penalty factor 
and degree denotes the number of knots the natural spline 
matrix is based on.
"""
function fit_gmeta(y::Vector, v::Vector, grid::Vector; c0 = .05, degree = 5)
    opt = Optim.optimize(
              Optim.only_fg!(
                  (F, G, x) -> 
                  fg!(F, G, x, grid, y, v; c0 = c0, degree = degree)
              ), 
          ones(degree+1),  # Starting value is vector of 1's
          LBFGS())         # Optimization algorithm

    # solution of optimization problem
    alpha_est = Optim.minimizer(opt)

    # estimate of prior distribution
    g_est, Q = log_g(alpha_est, grid; degree = degree)
    g_est = exp.(g_est)

    # estimate of bias, variance and mse of g_est 
    g_bias, g_variance = bias(alpha_est, grid, y, v; degree = degree, c0 = c0) 
    g_mse = sum(diag(g_variance)) + sum(g_bias .^2)

    # bias-corrected estimate of g_est is g_est - g_bias. However, this 
    # may be negative and hence does not make sense for a probability 
    # measure. Ad-hoc solution is to switch sign of negative values as these 
    # are usually small. 
    g_corrected = max.(g_est - g_bias, eps()) 

    # normalizing constant for calculating posterior distributions 
    norm_const = [sum(pdf.(Normal.(grid, sqrt(v[i])), y[i]) .* g_corrected) 
                  for i in eachindex(y)]

    # log-posterior distribution for each y
    posterior = hcat([(logpdf.(Normal.(grid,sqrt(v[i])), y[i]) 
                 .+ log.(g_corrected)) .- log(norm_const[i]) 
                 for i in eachindex(norm_const)]...)

    # output named tuple 
    return GModeling.GMeta(y, v, g_est, grid, 
            g_bias, g_variance, g_mse, g_corrected, 
            g_corrected ./ (grid[2]-grid[1]),
            posterior, alpha_est, Q)
end



# Function for simulating data and computing convolution estimates. 
function sim_gmeta(distribution, # Distribution theta's are drawn from
                   k;            # Number of theta's to draw  
                   c0 = .05,     # penalty factor 
                   degree = 5,   # Degree of natural spline basis
                   iteration = 1,# used to generate unique id
                   # provides ability to set random-number generator. 
                   rng = Random.GLOBAL_RNG, 
                   # provide grid of theta-values
                   grid = LinRange(-1,1.5,100))

  # simulate observations 
  d = sim_data(k, distribution; rng = rng)
  est = fit_gmeta(d.y, d.v, grid; c0 = c0, degree = degree) 

  # output named tuple  
  return(
    y = est.y, v = est.v,
    g_est = est.g_est, g_bias = est.g_bias, 
    g_corrected = est.g_corrected, 
    g_density = est.g_density, 
    posterior = est.posterior, grid = vcat(grid...), 
    mu = location(distribution), tau = scale(distribution), 
    k = k, degree = degree, iteration = iteration
  )
end


# standard error of weighted mean
function se(v,tau2)
  1 ./ sqrt(sum( 1 ./ (v .+ tau2)))
end

# Define constructor for output of gmeta below 
struct GMetaShow 
    est::Float64 
    tau2_hat::Float64 
    ci::Vector 
    pi::Vector
    ci_level::Float64
    crI
    fit::GMeta
end

Base.show(io::IO, z::GMetaShow) = println(
  io, "Estimate: ", round(z.est; digits = 4), "\n", 
      "τ² (τ): ", round(z.tau2_hat; digits = 4), " (",round(sqrt(z.tau2_hat); digits = 4), ")", "\n", 
      Int(z.ci_level * 100), "% CI: ", round.(z.ci; digits = 4), "\n",  
)

function gmeta(y::Vector, v::Vector, grid::Vector; c0 = .05, degree = 5, ci_level = .95, labels = nothing)
    if isnothing(labels) 
        labels = string.("Study", eachindex(y))  
    end
    fit = fit_gmeta(y,v,grid; c0 = c0, degree = degree)
    moments = vcat(fit.grid', (fit.grid .^2)') * fit.g_corrected 
    est = moments[1]
    tau2_hat = moments[2] - moments[1]^2
    ci = est .+ [-1, 1] .* quantile(Normal(), 1 - (1 - ci_level) / 2) .* se(v, tau2_hat)
    prior_sample = post_draw(fit.grid, fit.g_corrected)
    pred_sample = vcat([rand.(Normal.(prior_sample[i], sqrt.(v))) for i in eachindex(prior_sample)]...)
    pi = quantile(pred_sample, [(1-ci_level)/2, 1-(1-ci_level)/2])
    posterior_sample = [post_draw(fit.grid, exp.(fit.posterior[:,j])) for j in axes(fit.posterior,2)]
    crI = hcat([quantile(posterior_sample[j], [(1-ci_level)/2, .5, 1-(1-ci_level)/2]) for j in eachindex(posterior_sample)]...)
    crI_df = DataFrame(hcat(labels, crI'), [:study, :lb, :md, :ub])
    return GMetaShow(est, tau2_hat, ci, pi, ci_level, crI_df, fit)
end


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


# function for computing true posterior quantiles 
function posterior(y, v, mu, tau)
    M = vcat(moments.(y, v, mu, tau)...)
    return hcat([vcat(quantile(Normal(M[r,1], 
                sqrt(M[r,2])), [.025, .5, .975]), 
                y[r], v[r]...) for r in axes(M,1)]...)'
end


end # GModeling

