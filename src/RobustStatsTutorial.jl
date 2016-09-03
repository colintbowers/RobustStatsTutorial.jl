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
using Base.Dates, Compose, Gadfly, StatsBase, KernelDensity, Distributions, DependentBootstrap, NLopt
#WARNING: DependentBootstrap is not currently an official package (although I am the author). To install, use:
#pkg.clone("https://github.com/colintbowers/DependentBootstrap.jl.git")

#Fixed output directory based on Linux OS. Will need to be adjusted for Windows or Mac users.
const outputDir = "/home/"*ENV["USER"]*"/robust_stats_tutorial_output/"::ASCIIString
const dataDir = "/home/"*ENV["USER"]*"/.julia/v"*string(VERSION.major)*"."*string(VERSION.minor)*"/RobustStatsTutorial/data/"::ASCIIString

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
function tail_fatness_financial_data( ; scaleMethod::Symbol=:historicvariance, blockLength::Float64=4.0)
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
	#Compare mean and trimmed mean on resampled financial returns (use NAB (fattest tails) and CCL (thinnest tails))
	println("Drawing mean comparison plots (full period)")
	pNABFull = kernel_plot_robust_mean_versus_trimmed_mean(secRet, "NAB", blockLength=blockLength)
	draw_local(pNABFull, "NAB_Bootstrapped_Location_Estimator_Density_(full_period)", dirPath=outputDir, fileType=:svg)
	pCCLFull = kernel_plot_robust_mean_versus_trimmed_mean(secRet, "CCL", blockLength=blockLength)
	draw_local(pCCLFull, "CCL_Bootstrapped_Location_Estimator_Density_(full_period)", dirPath=outputDir, fileType=:svg)
	#Compare mean and trimmed mean on short horizon financial returns for large robust kurtosis versus small robust kurtosis
	println("Estimating rolling robust kurtosis")
	rNAB = secRet[find(secList .== "NAB")[1]]
	rollingHoggNAB = Float64[ hogg_robust_kurt(rNAB[j:j+99]) for j = 1:length(rNAB)-99 ]
	iSmlHogg = indmin(rollingHoggNAB)
	iLrgHogg = indmax(rollingHoggNAB)
	rNABSml = rNAB[iSmlHogg:iSmlHogg+99]
	rNABLrg = rNAB[iLrgHogg:iLrgHogg+99]
	println("Drawing mean comparison plots (short horizon)")
	pNABSml = kernel_plot_robust_mean_versus_trimmed_mean(rNABSml, secStr="NAB Short Horizon (Robust kurt = " * string(rollingHoggNAB[iSmlHogg]) * ")", blockLength=blockLength)
	pNABLrg = kernel_plot_robust_mean_versus_trimmed_mean(rNABLrg, secStr="NAB Short Horizon (Robust kurt = " * string(rollingHoggNAB[iSmlHogg]) * ")", blockLength=blockLength)
	draw_local(pNABSml, "NAB_Bootstrapped_Location_Estimator_Density_(short_horizon_thin_tail)", dirPath=outputDir, fileType=:svg)
	draw_local(pNABLrg, "NAB_Bootstrapped_Location_Estimator_Density_(short_horizon_fat_tail)", dirPath=outputDir, fileType=:svg)
	println("Routine complete")
end
function kernel_plot_robust_mean_versus_trimmed_mean(secRet::Vector{Vector{Float64}}, secStr::String ; numResample::Int=1000, blockLength::Float64=4.0)
	iSec = find(secList .== secStr)
	length(iSec) != 1 && error("Unable to find " * secStr * " data")
	r = secRet[iSec[1]]
	return(kernel_plot_robust_mean_versus_trimmed_mean(r, secStr=secStr, numResample=numResample, blockLength=blockLength))
end
function kernel_plot_robust_mean_versus_trimmed_mean(r::Vector{Float64} ; numResample::Int=1000, blockLength::Float64=4.0, secStr::String="")
	rCent = r - mean(r) #Centred so re-sampled data has true mean of zero
	rBoot = dbootstrapdata(rCent, blockLength=blockLength, numResample=numResample)
	bootEst = Vector{Float64}[Float64[ 10000*mean(rBoot[:, m]) for m = 1:numResample ], Float64[ 10000*tmean(rBoot[:, m], 0.4) for m = 1:numResample ]]
	kDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(bootEst[k]) for k = 1:2 ]
	layerVec = Vector{Gadfly.Layer}[ layer(x=collect(kDVec[k].x), y=kDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	secStr == "" ? (localTitle = "Location estimator densities for return data") : (localTitle = "Location estimator densities for " * secStr * " return data")
	kernelPlot = plot(layerVec..., Guide.xlabel("Estimator value (basis points)"), Guide.ylabel("Density"), Guide.title(localTitle), Guide.manual_color_key(default_legend(["Mean", "Trimmed mean (0.4)"])...), defaultThemeOverride)
end


#-------------------------------------------
# ROBUST PREDICTION OF FINANCIAL RETURNS
#-------------------------------------------
function return_prediction_financial_data( ; blockLength::Float64=4.0, numResample::Int=1000, varProxyMethod::Symbol=:historicvariance, numObs::Int=2000, rSquared::Float64=0.05, a::Float64=0.0, lag::Int=17, tCost::Float64=0.0, trimSimTail::Float64=0.0, numAsset::Int=20, tradingSimRSquared::Float64=0.01, includeConstantInSim::Bool=true)
	#Get variance proxy
	println("Reading data")
	if varProxyMethod == :realisedvariance
		v = read_local(["NAB"], :realisedvariance)[1]
	elseif varProxyMethod == :historicvariance
		rNAB = read_local(["NAB"], :return)[1]
		v = historical_variance(rNAB, lag)
	else
		error("Invalid varProxyMethod")
	end
	#Simuate returns, signal, error term, constant and coefficient
	println("Simulating and bootstrapping")
	(r, s, e, a, b) = simulate_signal_and_returns(v, rSquared=rSquared, a=a)
	# #Resample r, x, and e
	# inds = dbootstrapindex(length(r), blockLength, numResample=numResample)
	# rBoot = r[inds]
	# sBoot = s[inds]
	# eBoot = e[inds]
	# hoggVec = Float64[ hogg_robust_kurt(rBoot[:, m]) for m = 1:numResample ]
	# qProb = Float64[0.05, 0.25, 0.5, 0.75, 0.95]
	# hoggVecQ = quantile(hoggVec, qProb)
	# #Run regressions on the resampled data using multiple methods and get plot of estimated coefficients
	# println("Estimating coefficients on bootstrapped data")
	# coefLS = Array(Vector{Float64}, numResample)
	# coefLAD = Array(Vector{Float64}, numResample)
	# rSqLS = Array(Float64, numResample)
	# rSqLAD = Array(Float64, numResample)
	# rHatLS = Array(Float64, length(r), numResample)
	# rHatLAD = Array(Float64, length(r), numResample)
	# for m = 1:numResample
	# 	y = rBoot[:, m]
	# 	x = [ones(Float64, length(y)) sBoot[:, m]]
	# 	coefLS[m] = x \ y
	# 	(coefLAD[m], _, _) = least_absolute_deviation(y, x)
	# 	rHatLS[:, m] = x * coefLS[m]
	# 	rHatLAD[:, m] = x * coefLAD[m]
	# 	eLS = y - rHatLS[:, m]
	# 	eLAD = y - rHatLAD[:, m]
	# 	tSS = sumabs2(y)
	# 	rSqLS[m] = 1 - sumabs2(eLS) / tSS
	# 	rSqLAD[m] = 1 - sumabs2(eLAD) / tSS
	# end
	# #Get kernel density plot of estimated coefficients (and include true value)
	# println("Building coefficient kernel densities")
	# aArr = Vector{Float64}[Float64[ coefLS[m][1] for m = 1:numResample ], Float64[ coefLAD[m][1] for m = 1:numResample ]]
	# bArr = Vector{Float64}[Float64[ coefLS[m][2] for m = 1:numResample ], Float64[ coefLAD[m][2] for m = 1:numResample ]]
	# aKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(aArr[k]) for k = 1:2 ]
	# bKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(bArr[k]) for k = 1:2 ]
	# aLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(aKDVec[k].x), y=aKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	# bLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(bKDVec[k].x), y=bKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	# aKernelPlot = plot(aLayerVec..., Guide.xlabel("Regression constant"), Guide.ylabel("Density"), Guide.title("Density of regression constant (true value = " * string(a) * ")"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	# bKernelPlot = plot(bLayerVec..., Guide.xlabel("Regression coefficient"), Guide.ylabel("Density"), Guide.title("Density of regression coefficient (true value = " * string(b) * ")"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	# println("Plotting coefficient kernel densities")
	# draw_local(aKernelPlot, "Regression_constant_density", dirPath=outputDir, fileType=:svg)
	# draw_local(bKernelPlot, "Regression_coefficient_density", dirPath=outputDir, fileType=:svg)
	# #Get kernel density plot of r-squares
	# println("Building rSquared kernel density")
	# rSqArr = Vector{Float64}[rSqLS, rSqLAD]
	# rSqKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(rSqArr[k]) for k = 1:2 ]
	# rSqLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(rSqKDVec[k].x), y=rSqKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	# rSqKernelPlot = plot(rSqLayerVec..., Guide.xlabel("R-squared"), Guide.ylabel("Density"), Guide.title("Density of regression R-squared (true value = " * string(rSquared) * ")"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	# println("Plotting rSquared kernel density")
	# draw_local(rSqKernelPlot, "Regression_rSquared_density", dirPath=outputDir, fileType=:svg)
	# println("Hogg robust estimator quantiles (" * string(qProb) * ") = " * string(hoggVecQ))
	#Perform simulations using the estimated models and plot density of terminal values
	println("Performing simple trading simulations")
	terminalValLS = Array(Float64, numResample)
	terminalValLAD = Array(Float64, numResample)
	avgAbsErrorLS = Array(Float64, numResample)
	avgAbsErrorLAD = Array(Float64, numResample)
	propCorSignPredictionLS = Array(Float64, numResample)
	propCorSignPredictionLAD = Array(Float64, numResample)
	coefMatLS = Array(Float64, numResample, numAsset)
	constantMatLS = Array(Float64, numResample, numAsset)
	coefMatLAD = Array(Float64, numResample, numAsset)
	constantMatLAD = Array(Float64, numResample, numAsset)
	bMat = Array(Float64, numResample, numAsset)
	rHatCorrLS = Array(Float64, numResample, numAsset)
	rHatCorrLAD = Array(Float64, numResample, numAsset)
	sCorr = Array(Float64, numResample, numAsset)
	for m = 1:numResample
		rMat = Array(Float64, length(r), numAsset)
		sMat = Array(Float64, length(r), numAsset)
		eMat = Array(Float64, length(r), numAsset)
		aVec = Array(Float64, numAsset)
		bVec = Array(Float64, numAsset)
		for j = 1:numAsset
			(rMat[:, j], sMat[:, j], eMat[:, j], aVec[j], bVec[j]) = simulate_signal_and_returns(v, rSquared=tradingSimRSquared, a=0.0)
		end
		bMat[m, :] = deepcopy(bVec)
		coefLSVec = Array(Vector{Float64}, numAsset)
		coefLADVec = Array(Vector{Float64}, numAsset)
		rHatLSMat = Array(Float64, length(r), numAsset)
		rHatLADMat = Array(Float64, length(r), numAsset)
		for j = 1:numAsset
			y = rMat[:, j]
			includeConstantInSim ? (x = [ones(Float64, length(y)) sMat[:, j]]) : (x = sMat[:, j]'')

			coefLSVec[j] = x \ y
			#coefLSVec[j] = inv(x' * x) * x' * y

			(coefLADVec[j], _, _) = least_absolute_deviation(y, x)
			any(isnan(coefLADVec[j])) && (coefLADVec[j] = coefLSVec[j])
			# println("-------")
			# println(coefLSVec[j])
			# println(coefLADVec[j])

			rHatLSMat[:, j] = x * coefLSVec[j]
			rHatLADMat[:, j] = x * coefLADVec[j]
			includeConstantInSim ? (constantMatLS[m, j] = coefLSVec[j][1]) : (coefMatLS[m, j] = NaN)
			includeConstantInSim ? (coefMatLS[m, j] = coefLSVec[j][2]) : (coefMatLS[m, j] = coefLSVec[j][1])
			includeConstantInSim ? (constantMatLAD[m, j] = coefLADVec[j][1]) : (coefMatLAD[m, j] = NaN)
			includeConstantInSim ? (coefMatLAD[m, j] = coefLADVec[j][2]) : (coefMatLAD[m, j] = coefLADVec[j][1])

			rHatCorrLS[m, j] = cor(rMat[:, j], coefLSVec[j][1] * sMat[:, j])
			rHatCorrLAD[m, j] = cor(rMat[:, j], coefLADVec[j][1] * sMat[:, j])
			sCorr[m, j] = cor(rMat[:, j], sMat[:, j])

			# rHatCorrLS[m, j] = cor(rMat[:, j], rHatLSMat[:, j])
			# rHatCorrLAD[m, j] = cor(rMat[:, j], rHatLADMat[:, j])
			# sCorr[m, j] = cor(rMat[:, j], sMat[:, j])

			# rHatCorrLS[m, j] = cor(y, rHatLSMat[:, j])
			# rHatCorrLAD[m, j] = cor(y, rHatLADMat[:, j])
			# sCorr[m, j] = cor(y, vec(x))


		end
		avgAbsErrorLS[m] = mean(abs(rHatLSMat - rMat))
		avgAbsErrorLAD[m] = mean(abs(rHatLADMat - rMat))
		propCorSignPredictionLS[m] = mean((rHatLSMat .> 0.0) .* (rMat .> 0.0))
		propCorSignPredictionLAD[m] = mean((rHatLADMat .> 0.0) .* (rMat .> 0.0))
		terminalValLS[m] = very_simple_trading_sim(rMat, rHatLSMat)[end]
		terminalValLAD[m] = very_simple_trading_sim(rMat, rHatLADMat)[end]
	end
	println("Proportion of samples when LAD has smaller absolute forecast error = " * string(sum(avgAbsErrorLAD .< avgAbsErrorLS) / numResample))
	println("Proportion of correct sign predictions (LS) = " * string(mean(propCorSignPredictionLS)))
	println("Proportion of correct sign predictions (LAD) = " * string(mean(propCorSignPredictionLAD)))
	println("Mean terminal value (LS) = " * string(mean(terminalValLS)))
	println("Mean terminal value (LAD) = " * string(mean(terminalValLAD)))
	println("Median terminal value (LS) = " * string(median(terminalValLS)))
	println("Median terminal value (LAD) = " * string(median(terminalValLAD)))
	println("Average correlation between signal and r = " * string(mean(sCorr, 1)))
	println("Average correlation between rHatLS and r = " * string(mean(rHatCorrLS, 1)))
	println("Average correlation between rHatLAD and r = " * string(mean(rHatCorrLAD, 1)))

	println("Average correlation (all assets) between signal and r = " * string(mean(sCorr)))
	println("Average correlation (all assets) between rHatLS and r = " * string(mean(rHatCorrLS)))
	println("Average correlation (all assets) between rHatLAD and r = " * string(mean(rHatCorrLAD)))


	writecsv("/home/colin/Temp/corMatLS.csv", rHatCorrLS)
	writecsv("/home/colin/Temp/corMatLAD.csv", rHatCorrLAD)
	writecsv("/home/colin/Temp/corSig.csv", sCorr)

	# writecsv("/home/colin/Temp/coefMatLS.csv", coefMatLS)
	# writecsv("/home/colin/Temp/coefMatLAD.csv", coefMatLAD)
	# writecsv("/home/colin/Temp/coefMatTrue.csv", bMat)
	# sort!(terminalValLS)
	# sort!(terminalValLAD)
	# iLower = Int(floor(trimSimTail * length(terminalValLS)))
	# iLower == 0 && (iLower = 1)
	# iUpper = length(terminalValLS) - iLower + 1
	# terminalValLS = terminalValLS[iLower:iUpper]
	# terminalValLAD = terminalValLAD[iLower:iUpper]
	# println("Plotting terminal value kernel densities")
	# tV = Vector{Float64}[terminalValLS, terminalValLAD]
	# tVKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(tV[k]) for k = 1:2 ]
	# tVLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(tVKDVec[k].x), y=tVKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	# tVKernelPlot = plot(tVLayerVec..., Guide.xlabel("Terminal value"), Guide.ylabel("Density"), Guide.title("Density of terminal values from simple simulations"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	# draw_local(tVKernelPlot, "Simulated_portfolio_terminal_values", dirPath=outputDir, fileType=:svg)
	println("Routine complete")
end
function simulate_signal_and_returns(v::Vector{Float64} ; rSquared::Float64=0.05, a::Float64=0.0)
	!(0.0 < rSquared < 1.0) && error("Invalid rSquared")
	e = sqrt(v) .* randn(length(v))
	x = randn(length(v))
	residSumSq = sumabs2(e)
	totalSumSq = residSumSq / (1 - rSquared)
	explainSumSq = totalSumSq - residSumSq
	b = sqrt(explainSumSq / sumabs2(x)) #This ensures sample rSquared will equal input rSquared (Note, sample rSquared will become population rSquared for re-sampled data)
	#b = sqrt(explainSumSq / length(v)) #This alternative would ensure population rSquared will equal input rSquared
	r = a + b*x + e
	return(r, x, e, a, b)
end
function least_absolute_deviation(y::Vector{Float64}, x::Matrix{Float64})
	#Build Opt object accepted by NLOpt
	fObj = ((param, grad) -> sumabs(y - x*param)) #Get local anonymous objective function
	initVal = zeros(Float64, size(x, 2))
	opt = Opt(:LN_COBYLA, length(initVal)) #Derivative-free convergence method (yes I'm being lazy here)
	xtol_rel!(opt, 1e-6)
	min_objective!(opt, fObj)
	#Perform optimisation
	(objFuncOpt, paramOpt, flag) = optimize(opt, initVal)
	checkNLoptFlag(flag)
	return(paramOpt, objFuncOpt, y - x*paramOpt) #Return optimal parameter, objective function at optimum, and estimated errors
end
function checkNLoptFlag(flag::Symbol)
	flag == :SUCCESS && return(true)
	flag == :XTOL_REACHED && return(true)
	if flag == :ROUNDOFF_LIMITED
		println("WARNING: Optimisation terminated due to floating point limit. Final iteration parameters used.")
		return(true)
	end
	flag == :MAXEVAL_REACHED && error("Maximum number of evaluations reached in NLopt")
	flag == :MAXTIME_REACHED && error("Maximum optimisation time reached in NLopt")
	flag == :FAILURE && error("Generic failure in NLopt")
	flag == :INVALID_ARGS && error("Invalid arguments to NLopt")
	flag == :OUT_OF_MEMORY && error("NLopt ran out of memory")
	flag == :FORCED_STOP && error("NLopt halted due to forced termination")
	error("Unexpected flag from NLopt. Flag = " * string(flag))
end
#Function for performing a very simple trading simulation. Either hold cash, or trade an asset if forecast return > 0. tCost is in basis points
function very_simple_trading_sim(r::Vector{Float64}, f::Vector{Float64} ; tCost::Float64=50.0)
	length(f) != length(r) && error("Length mismatch")
	N = length(f)
	portValue = Array(Float64, N)
	portValue[1] = 1000000.0
	for n = 2:N
		if f[n] > 0.0
			portValue[n] = portValue[n-1] * (r[n] + 1) - (1/10000) * tCost * portValue[n-1] #notional transaction cost
		else
			portValue[n] = portValue[n-1]
		end
	end
	return(portValue)
end
function very_simple_trading_sim(r::Matrix{Float64}, f::Matrix{Float64})
	size(r) != size(f) && error("Size mismatch")
	portValue = Array(Float64, size(f, 1))
	portValue[1] = 1000000.0
	for n = 2:size(r, 1)
		inds = (vec(f[n, :]) .> 0.0)
		if !any(inds)
			portValue[n] = portValue[n-1]
		else
			portValue[n] = port_value_update(portValue[n-1], vec(r[n, :])[inds])
		end
	end
	return(portValue)
end
port_value_update(p::Float64, r::Float64) = p * (r + 1)
port_value_update(p::Vector{Float64}, r::Vector{Float64}) = (length(p) == length(r)) ? sum(Float64[ port_value_update(p[j], r[j]) for j = 1:length(p) ]) : error("Length mismatch")
port_value_update(p::Float64, r::Vector{Float64}) = port_value_update((p/length(r))*ones(Float64, length(r)), r)
#-------------------------------------------
#
#-------------------------------------------




end # module
