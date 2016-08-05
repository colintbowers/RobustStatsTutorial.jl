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
#Compare least squares estimated forecast model for risky asset returns with robust loss estimation method
#Also compare to case where returns are standardised for volatility first
#Might need to compare to other robust methods of estimation too, ie ridge regression, lasso (? not robust...)


#Load relevant modules
using Base.Dates, Compose, Gadfly, StatsBase, KernelDensity, Distributions

#Fixed output directory based on Linux OS. Will need to be adjusted for Windows or Mac users.
const outputDir = "/home/"*ENV["USER"]*"/robust_stats_tutorial_output/"::ASCIIString

#Fixed theme override for all plots
const defaultThemeOverride = Theme(key_title_font_size=14pt, key_label_font_size=14pt, minor_label_font_size=14pt, major_label_font_size=17pt, key_title_font_size=17pt)::Theme


#Create output directory
!isdir(outputDir) && mkdir(outputDir)

#Include other module files
include("common.jl")

#--------------------------------------------
# EXAMPLE WITH t-DISTRIBUTION
#--------------------------------------------
function t_dist_simple_example()
    numObs = 20 #Number of observations to use in this subsection
    tDist = TDist(2) #Initiate t-distribution with 2 degrees of freedom
    srand(78) #Specify a range that results in a good dataset for demonstrating the point (yes I mined this from the first 100 integers)
    tData = rand(tDist, numObs) #Simulate iid from t-distribution
    tDataSort = sort(tData) #Sort the data
    tDataMean = mean(tData) #Get sample mean
    p = 0.2 #Set a trimming proportion
    tDataTrimMean = tmean(tDataSort, p, sorted=true) #Get trimmed mean
    (numLower, numUpper) = tmean_num_cut(tDataSort, p/2, p/2) #Get number of observations cut below and above
    yLower = mean(tDataSort[numLower:numLower+1]) #Get the midpoint of the last lower cut observation and the first lower kept observation
    yUpper = mean(tDataSort[end-numUpper:end-numUpper+1]) #Get the midpoint of the last upper kept observation and the first upper cut observation
    dataPlot1 = plot(x=collect(1:numObs), y=tData, yintercept=[yLower, yUpper], Geom.point, Geom.hline, defaultThemeOverride)
    draw_local(dataPlot1, "t_Dist_Data_With_Trim_Cutoff", dirPath=outputDir, fileType=:svg)
end






end # module
