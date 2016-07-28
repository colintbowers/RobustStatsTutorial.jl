module RobustStatsTutorial
#--------------------------------------------
# SOURCE CODE FOR ROBUST STATISTICS TUTORIAL
#--------------------------------------------

#Draft order
#Plot of data from t-distribution with DoF = 2. Set specific seed. Show how much better we can do by eliminating a couple of observations.
#Simulate empirical densities of sample means, sample medians, and sample trimmed means, for t-distribution with DoF = 2
#Discuss fat-tails of t-distribution with DoF = 2. Explain why kurtosis, although the common characterisation, is not an ideal measure of how fat the tails are. Possibly plot sample kurtosis as a function of sample size.
#Introduce alternative robust measure of fat-tails.
#Provide measures of fat-tails for t-distribution with DoF = 2.
#Provide measures of fat-tails for unconditional risky asset returns.
#Show how fat-tails in unconditional risky asset returns can be duplicated using a conditionally Normal model, i.e. heteroskedasticity in Normal sequence
#Something to do with estimating first moment of risky asset returns (?)
#Move onto OLS model for risky asset returns


*A demonstration that Normality is an inappropriate assumption for the *unconditional* distribution of stock returns
*A demonstration of how a simple estimator everyone is familiar with, like the sample mean, is seriously sub-optimal when data is leptokurtic, and a brief introduction to robust statistics as a better method of estimation
    *Move on to showing how OLS (depending on audience, might quickly sidetrack into how least squares is an equivalent assumption to Normality via the equivalence of OLS and maximum likelihood with Normality assumption) is suboptimal when residuals are leptokurtic. Can use simulated data first, and then move onto using real-world stock return data.
   *Demonstrate other methods of estimating the parameters of a linear model that outperform OLS when the residuals are leptokurtic
    *Depending on timing, I could now move on to discussing conditional Normality of stock returns, i.e. how daily returns can be modelled as Normal if you can model how variance changes from day to day (potentially mention the equivalence between a heteroskedastic sequence of Normal random variables and an unconditional leptokurtic distribution). I can show a neat trick here how if you estimate daily variance using one of the neat tricks from high frequency estimation (I don't need to go into too much detail on how this is done if it will bore the audience), then you can standardise daily returns by these daily variance estimates and the resulting sequence is astonishingly close to iid standard Normal, ie N(0, 1).
   *This can lead into the point that another modelling trick if you want to use OLS is to appropriately standardise your data first, although there are some real statistical traps here, particularly if you believe volatility has predictive power for returns.





end # module
