# "module MetaAnalysis" defines a name space where all the 
# functions used to calculate ML and REML Meta-Analyses
# "live".
module MetaAnalysis 
using Base: Forward
using Statistics
using Optim


"""
Takes vector of estimates y and corresponding variances and returns a 
method of moments estimator for the between-study variance tau^2.
UMM is used to set a sensible starting value for the optimization 
algorithms below.
"""
function umm(y::Vector, v::Vector)
  max(10^(-5), 1 ./(length(y)-1)*sum((y .-mean(y)) .^2)-1/length(y)*sum(v))
end

# Evaluates the profile log-likelihood of y ~ N(mu, tau2+v) at 
# tau2 (mu is profiled out).
function pll!(F, G, tau2, y::Vector, v::Vector)
  w = ifelse.(isinf.(1 ./v), 0, 1 ./(v .+tau2))
  b = sum(w .*y) ./sum(w)

  if !(G === nothing)
    mu_prime = sum(w)^(-2) * sum(w .^2) * sum(y .* w)  
      - sum(w)^(-1)*sum(y .* w .^2) 
    G .= 1/2 * sum(w) + 1/2 * sum(w .^2 .* (-2 * (y .- b) ./ w 
      .* mu_prime - (y .- b) .^2))
  end
  
  if !(F === nothing)
    return 1/2*(sum(log.(v .+tau2)) .+ sum((y .- b) .^2 .*w))
  end
end

# Evaluates the restricted profile log-likelihood at tau2
function rpll(tau2, y::Vector, v::Vector)
  w = ifelse.(isinf.(1 ./v), 0, 1 ./(v .+tau2))
  b = sum(w .*y) ./sum(w)
  log_l = -1/2*(sum(log.(v .+tau2)) .+ log(sum(w)) .+sum((y .- b) .^2 .*w))
  -log_l
end

# Evaluates the log-likeliood at tau2 and mu, returns score vector.
function loglik!(F, G, param, y::Vector, v::Vector)
  if !(G === nothing)
    G[1] = - sum((y .- param[1]) ./ (v .+ param[2]))
    G[2] = 1/2*(sum(1 ./ (v .+ param[2])) .- 
         sum(((y .- param[1]) ./ (v .+ param[2])) .^2))
  end

  if !(F === nothing)
    return 1/2*(sum(log.(v .+param[2])) .+ 
           sum((y .- param[1]) .^2 ./(v .+param[2])))
  end
end


# Evaluates the restricted log-likeliood at tau2 and mu, returns score vector.
function rloglik!(F, G, param, y::Vector, v::Vector)
  if !(G === nothing)
    G[1] = - sum((y .- param[1]) ./ (v .+ param[2]))
    G[2] = 1/2*(sum(1 ./ (v .+ param[2])) - 
           sum(((y .- param[1]) ./ (v .+ param[2])) .^2) + 
           sum((v .+ param[2]) .^(-2))/sum((v .+ param[2]) .^(-1)))
  end

  if !(F === nothing)
    return 1/2*(sum(log.(v .+param[2])) .+ log(sum(v .+ param[2])) .+ 
           sum((y .- param[1]) .^2 ./(v .+param[2])))
  end
end


# Compute estimate of mu using for tau2, y and v.
function mu_hat(tau2, y::Vector, v::Vector)
  w = 1 ./ (v .+ tau2)
  sum(w .* y) ./ sum(w)
end

# Compute MLE for mu and tau2
function mle(y,v)
  # Starting values 
  x0 = umm(y,v)
  opt = Optim.optimize(
    Optim.only_fg!((F,G,t) -> pll!(F, G, t, y, v)), 
    [0], [Inf], [x0], Fminbox(LBFGS()))
  tau2_hat = Optim.minimizer(opt)[1]
  est = [mu_hat(tau2_hat, y, v), tau2_hat]
  return est
end

# Compue REML for mu and tau2
function reml(y,v)
  # Starting values 
  x0 = umm(y,v)
  opt = Optim.optimize(t -> rpll(t, y, v), [0], [Inf], [x0], 
    Fminbox(LBFGS()); autodiff = :forward)
  tau2_hat = Optim.minimizer(opt)[1]
  est = [mu_hat(tau2_hat, y, v), tau2_hat]
  return est
end

end #MetaAnalysis
