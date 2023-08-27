# Repository for "Empirical Bayes Meta-Analysis Through G-Modeling"

This repository contains Julia code and figures accompanying my bachelors 
thesis in statistics (see [this pdf](manuscript/_manuscript/index.pdf) for a recent draft). 

The primary objective is to employ the *g*-modeling framework to empirical Bayes
as developed by e.g., Efron (2016), to combine effect size estimates into a simple meta-analysis in Julia.  

The abstract reads:
 
> The usual approach to empirical Bayes modeling, referred to as $f$-modeling, builds on the assumption of a 
  prior distribution with known shape up to a parameter $\gamma$. Inference is carried out by setting 
  $\gamma$ equal to an estimate obtained from the marginal distribution of observations $y_1,\ldots,y_n$. A less restrictive modeling approach, referred to as $g$-modeling, 
  relaxes this assumption by requiring solely that the prior distribution belongs to the exponential family of distributions
  indexed by a parameter $\alpha$. So far, applications of $g$-modeling comprise parallel estimation problems 
  in relatively large data sets. Meta-Analyses also give rise to parallel estimation problems. Moreover, $f$-modeling approaches to 
  meta-analysis are well established. Here, a $g$-modeling based approach for conducting a 
  meta-analysis is developed and its performance is compared to both maximum- and restricted 
  maximum-likelihood based random-effects meta-analysis in simulations. Results indicate that the $g$-modeling meta-analysis compares reasonably well. 
  Zero-variance estimates of between-study effects are completely avoided by construction, and
  coverage probabilities of Wald-type confidence intervals for the average meta-analytic effect were consistently 
  better than those obtained in the maximum- and restricted maximum-likelihood meta-analytic frameworks.


# References 
Efron, Bradley. 2016. “Empirical Bayes Deconvolution Estimates.” Biometrika 103 (1): 1–20. [https://doi.org/10.1093/biomet/asv068](https://doi.org/10.1093/biomet/asv068).
