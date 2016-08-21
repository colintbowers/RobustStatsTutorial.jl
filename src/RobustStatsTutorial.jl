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
using Base.Dates, Compose, Gadfly, StatsBase, KernelDensity, Distributions, DependentBootstrap
#WARNING: DependentBootstrap is not currently an official package (although I am the author). To install, use:
#pkg.clone("https://github.com/colintbowers/DependentBootstrap.jl.git")

#Fixed output directory based on Linux OS. Will need to be adjusted for Windows or Mac users.
const outputDir = "/home/"*ENV["USER"]*"/robust_stats_tutorial_output/"::ASCIIString
const dataDir = "/home/"*ENV["USER"]*"/.julia/v0.4/RobustStatsTutorial/data/"::ASCIIString

#Fixed theme override for all plots and colour order for multivariate plots
const defaultThemeOverride = Theme(key_title_font_size=14pt, key_label_font_size=14pt, minor_label_font_size=14pt, major_label_font_size=17pt, key_title_font_size=17pt)::Theme
const colourVec = ["blue", "green", "red", "black", "purple", "dark blue", "darkgreen", "gray", "brown", "cyan",
				   "violetred", "blue2", "orange", "green2", "darkred", "gray20", "chocolate", "brown2", "darkorange", "gray40",
				   "cadetblue", "violet", "green4", "brown4", "blue4"]::Vector{ASCIIString} #A vector of 25 colours to use in plots

#Security list used throughout the tutorial
const secList = ["AMP", "ANZ", "BHP", "CBA", "CCL", "JBH", "LLC", "NAB", "RIO", "SUN", "TLS", "TOL", "WBC", "WES", "WOW"]::Vector{ASCIIString}

#Create output directory
!isdir(outputDir) && mkdir(outputDir)

#Include other module files
include("common.jl")

#--------------------------------------------
# SIMPLE EXAMPLE WITH t-DISTRIBUTION
#--------------------------------------------
#Plot of data from t-distribution with DoF = 2. Set specific seed. Show how much better we can do by eliminating a couple of observations.
function t_dist_simple_example()
    numObs = 20 #Number of observations to use in this subsection
    tDist = TDist(2) #Initiate t-distribution with 2 degrees of freedom
    srand(78) #Specify a range that results in a good dataset for demonstrating the point (yes I mined this from the first 100 integers)
    tData = rand(tDist, numObs) #Simulate iid from t-distribution
    tDataSort = sort(tData)
    tDataMean = mean(tData)
    p = 0.2 #Set a trimming proportion
    tDataTrimMean = tmean(tDataSort, p, sorted=true) #Get trimmed mean
    (numLower, numUpper) = tmean_num_cut(tDataSort, p/2, p/2) #Get number of observations cut below and above
    yLower = mean(tDataSort[numLower:numLower+1]) #Get the midpoint of the last lower cut observation and the first lower kept observation
    yUpper = mean(tDataSort[end-numUpper:end-numUpper+1]) #Get the midpoint of the last upper kept observation and the first upper cut observation
    println("Plotting data")
    dataPlot1 = plot(x=collect(1:numObs), y=tData, Geom.point, defaultThemeOverride)
    dataPlot2 = plot(x=collect(1:numObs), y=tData, yintercept=[yLower, yUpper], Geom.point, Geom.hline, defaultThemeOverride)
    draw_local(dataPlot1, "t_Dist_Data", dirPath=outputDir, fileType=:svg)
    draw_local(dataPlot2, "t_Dist_Data_With_Trim_Cutoff", dirPath=outputDir, fileType=:svg)
    println("Routine complete")
end

#--------------------------------------------
# ESTIMATOR DENSITIES WITH t-DISTRIBUTION
#--------------------------------------------
#Simulate empirical densities of sample means, sample medians, and sample trimmed means, for t-distribution with DoF = 2
function t_dist_estimator_densities( ; numIter::Int=5000, numObs::Int=50, tMeanProp1::Float64=0.1, tMeanProp2::Float64=0.5)
    tDist = TDist(2) #Initiate t-distribution with 2 degrees of freedom
    estVecVec = Vector{Float64}[ Array(Float64, numIter) for k = 1:4 ] #Pre-allocate structure to hold sample estimators
    println("Simulating estimators")
    for n = 1:numIter
        #Simulate t-distribution data, and then get mean, two trimmed means, and the median for each dataset
        tData = rand(tDist, numObs)
        tDataSort = sort(tData)
        estVecVec[1][n] = mean(tData)
        estVecVec[2][n] = tmean(tDataSort, tMeanProp1, sorted=true)
        estVecVec[3][n] = tmean(tDataSort, tMeanProp2, sorted=true)
        estVecVec[4][n] = median_sorted(tDataSort)
    end
    println("Building kernel densities")
    kDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(estVecVec[k], boundary=(-1.0, 1.0)) for k = 1:4 ] #Get kernel density for each estimator
    layerVec = Vector{Gadfly.Layer}[ layer(x=collect(kDVec[k].x), y=kDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:4 ] #Construct plot layers using kernel densities
    println("Plotting kernel densities")
    kernelPlot1 = plot(layerVec..., Guide.xlabel("Estimator value"), Guide.ylabel("Density"), Guide.manual_color_key(default_legend(["mean", string(tMeanProp1)*" trimmed mean", string(tMeanProp2)*" trimmed mean", "median"])...), defaultThemeOverride)
    draw_local(kernelPlot1, "t_Dist_Estimator_Densities", dirPath=outputDir, fileType=:svg)
    println("Routine complete")
end

#--------------------------------------------
# MEASURING TAIL FATNESS
#--------------------------------------------
#Discuss fat-tails of t-distribution with DoF = 2. Explain why kurtosis, although the common characterisation, is not an ideal measure of how fat the tails are. Possibly plot sample kurtosis as a function of sample size.
#Introduce alternative robust measure of fat-tails and show it is stable relative to sample kurtosis
#Specifically, show that sample kurtosis diverges as sample size increases if population kurtosis is undefined
function tail_fatness_and_kurtosis( ; numIter::Int=100, numObsVec::Vector{Int}=collect(20:20:200))
    #Simulate the kurtosis and robust kurtosis as a function of sample size for t-distribution with 2 degrees of freedom
	tDist = TDist(2) #Initiate t-distribution with 2 degrees of freedom
    simKurtosisMat = Array(Float64, numIter, length(numObsVec))
	simRobustKurtosisMat = Array(Float64, numIter, length(numObsVec))
    for k = 1:length(numObsVec)
		for m = 1:numIter
			tData = rand(tDist, numObsVec[k])
        	simKurtosisMat[m, k] = kurtosis(tData)
			simRobustKurtosisMat[m, k] = hogg_robust_kurt!(tData, sorted=false, numerTail=0.05, denomTail=0.5)
		end
    end
	simKurtosis = vec(mean(simKurtosisMat, 1))
	simRobustKurtosis = vec(mean(simRobustKurtosisMat, 1))
    estVec = Vector{Float64}[simKurtosis, simRobustKurtosis]
    layerVec = Vector{Gadfly.Layer}[ layer(x=numObsVec, y=estVec[k], Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ] #Construct plot layers for the two estimators
    estPlot1 = plot(layerVec..., Guide.xlabel("Number of observations"), Guide.ylabel("Simulated average estimator value"), Guide.manual_color_key(default_legend(["Sample kurtosis", "Robust measure"])...), defaultThemeOverride)
    println("Plotting estimators...")
    draw_local(estPlot1, "t_Dist_Kurtosis_Versus_Robust_Kurtosis", dirPath=outputDir, fileType=:svg)
	println("Routine complete")
end


#--------------------------------------------
# TAIL FATNESS OF FINANCIAL DATA VERSUS NORMAL
#--------------------------------------------
#Compare robust measure of fat-tails for lots of different stocks to value under Normal distribution and value under t-distribution with 2 DoF
#Discuss how we can generate unconditional fat-tails using a conditional Normal model with time-varying variance. Plot robust measure of fat-tails for returns standardised by realised variance type estimator. Show how it is close to Normal.
#Conclude by comparing mean of financial returns to trimmed mean and median.
function tail_fatness_financial_data( ; scaleMethod::Symbol=:historicvariance)
	#Get robust measure of return data and compare it to Normal and t-Dist with 2 DoF
	secRet = read_local(secList, :return)
    hoggEst = Float64[ hogg_robust_kurt(secRet[j]) for j = 1:length(secList) ]
    tDist = TDist(2)
    tDistHoggEst = mean(Float64[ hogg_robust_kurt(rand(tDist, 100)) for k = 1:1000 ])
    normalHoggEst = mean(Float64[ hogg_robust_kurt(randn(100)) for k = 1:1000 ])
	println("Drawing plot 1")
	estPlot1 = plot(x=secList, y=hoggEst, yintercept=[normalHoggEst, tDistHoggEst], Geom.point, Geom.hline, defaultThemeOverride)
    draw_local(estPlot1, "Robust_Kurtosis_of_Daily_Financial_Returns", dirPath=outputDir, fileType=:svg)
	#Standardise returns by a variance estimate and then repeat above exercise
	if scaleMethod == :historicvariance
		secVar = Vector{Float64}[ historical_variance(secRet[k]) for k = 1:length(secRet) ]
		secRetStd = Vector{Float64}[ secRet[j][end-length(secVar[j])+1:end] ./ sqrt(secVar[j]) for j = 1:length(secList) ]
	else
		error("Invalid scaleMethod")
	end
	hoggEstStd = Float64[ hogg_robust_kurt(secRetStd[j]) for j = 1:length(secList) ]
	println("Drawing plot 2")
	estPlot2 = plot(x=secList, y=hoggEstStd, yintercept=[normalHoggEst, tDistHoggEst], Geom.point, Geom.hline, defaultThemeOverride)
    draw_local(estPlot2, "Robust_Kurtosis_of_Standardised_Daily_Financial_Returns", dirPath=outputDir, fileType=:svg)
	#Compare mean and trimmed mean on resampled financial returns (use NAB since it had the unconditionally fattest tails based on above analysis)
	numResample = 1000
	iNAB = find(secList .== "NAB")
	length(iNAB) != 1 && error("Unable to find NAB data")
	rNABCent = secRet[iNAB[1]] - mean(secRet[iNAB[1]]) #Centred so re-sampled data has true mean of zero
	rBootNAB = dbootstrapdata(rNABCent, blockLength=4.0, numResample=numResample)
	bootEst = Vector{Float64}[Float64[ 10000*mean(rBootNAB[:, m]) for m = 1:numResample ], Float64[ 10000*tmean(rBootNAB[:, m], 0.4) for m = 1:numResample ]]
	kDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(bootEst[k]) for k = 1:2 ]
	layerVec = Vector{Gadfly.Layer}[ layer(x=collect(kDVec[k].x), y=kDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	kernelPlot = plot(layerVec..., Guide.xlabel("Estimator value (basis points)"), Guide.ylabel("Density"), Guide.title("Location estimator densities for NAB return data"), Guide.manual_color_key(default_legend(["Mean", "Trimmed mean (0.4)"])...), defaultThemeOverride)
	draw_local(kernelPlot, "NAB_Bootstrapped_Location_Estimator_Density", dirPath=outputDir, fileType=:svg)
	println("Routine complete")
end




#-------------------------------------------
#
#-------------------------------------------




end # module
