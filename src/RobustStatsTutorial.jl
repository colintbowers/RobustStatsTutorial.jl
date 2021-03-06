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
const defaultThemeOverride = Theme(key_title_font_size=14pt, key_label_font_size=14pt, minor_label_font_size=14pt, major_label_font_size=17pt, key_title_font_size=17pt, default_color=color("blue"), default_point_size=4pt)::Theme
const colourVec = ["blue", "green", "red", "black", "purple", "dark blue", "darkgreen", "gray", "brown", "cyan",
				   "violetred", "blue2", "orange", "green2", "darkred", "gray20", "chocolate", "brown2", "darkorange", "gray40",
				   "cadetblue", "violet", "green4", "brown4", "blue4"]::Vector{ASCIIString} #A vector of 25 colours to use in plots

#Security list used throughout the tutorial
const secList = ["AMP", "ANZ", "BHP", "CBA", "CCL", "JBH", "LLC", "NAB", "RIO", "SUN", "TLS", "TOL", "WBC", "WES", "WOW"]::Vector{ASCIIString}

#Create output directory
!isdir(outputDir) && mkdir(outputDir)


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
	println("tDist data sample mean = " * string(tDataMean))
	println("tDist data trim mean = " * string(tDataTrimMean))
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
	estPlot1 = plot(x=secList, y=hoggEst, yintercept=[normalHoggEst, tDistHoggEst], Geom.point, Geom.hline, Guide.xlabel("Ticker code"), Guide.ylabel("Robust kurtosis"), defaultThemeOverride)
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
	estPlot2 = plot(x=secList, y=hoggEstStd, yintercept=[normalHoggEst, tDistHoggEst], Geom.point, Geom.hline, Guide.xlabel("Ticker code"), Guide.ylabel("Robust kurtosis"), defaultThemeOverride)
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
	pNABSml = kernel_plot_robust_mean_versus_trimmed_mean(rNABSml, secStr="NAB Short Horizon (Robust kurt = " * string(rollingHoggNAB[iSmlHogg])[1:3] * ")", blockLength=blockLength)
	pNABLrg = kernel_plot_robust_mean_versus_trimmed_mean(rNABLrg, secStr="NAB Short Horizon (Robust kurt = " * string(rollingHoggNAB[iLrgHogg])[1:3] * ")", blockLength=blockLength)
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
function kernel_plot_robust_mean_versus_trimmed_mean(r::Vector{Float64} ; numResample::Int=1000, blockLength::Float64=4.0, secStr::String="", bpMult::Int=10000)
	rCent = r - mean(r) #Centred so re-sampled data has true mean of zero
	rBoot = dbootstrapdata(rCent, blockLength=blockLength, numResample=numResample)
	bootEst = Vector{Float64}[Float64[ bpMult*mean(rBoot[:, m]) for m = 1:numResample ], Float64[ bpMult*tmean(rBoot[:, m], 0.4) for m = 1:numResample ]]
	kDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(bootEst[k]) for k = 1:2 ]
	layerVec = Vector{Gadfly.Layer}[ layer(x=collect(kDVec[k].x), y=kDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	secStr == "" ? (localTitle = "Location estimator densities") : (localTitle = "Location estimator densities for " * secStr * " return data")
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
	#Resample r, x, and e
	inds = dbootstrapindex(length(r), blockLength, numResample=numResample)
	rBoot = r[inds]
	sBoot = s[inds]
	eBoot = e[inds]
	hoggVec = Float64[ hogg_robust_kurt(rBoot[:, m]) for m = 1:numResample ]
	qProb = Float64[0.05, 0.25, 0.5, 0.75, 0.95]
	hoggVecQ = quantile(hoggVec, qProb)
	#Run regressions on the resampled data using multiple methods and get plot of estimated coefficients
	println("Estimating coefficients on bootstrapped data")
	coefLS = Array(Vector{Float64}, numResample)
	coefLAD = Array(Vector{Float64}, numResample)
	rSqLS = Array(Float64, numResample)
	rSqLAD = Array(Float64, numResample)
	rHatLS = Array(Float64, length(r), numResample)
	rHatLAD = Array(Float64, length(r), numResample)
	for m = 1:numResample
		y = rBoot[:, m]
		x = [ones(Float64, length(y)) sBoot[:, m]]
		coefLS[m] = x \ y
		(coefLAD[m], _, _) = least_absolute_deviation(y, x)
		rHatLS[:, m] = x * coefLS[m]
		rHatLAD[:, m] = x * coefLAD[m]
		eLS = y - rHatLS[:, m]
		eLAD = y - rHatLAD[:, m]
		tSS = sumabs2(y)
		rSqLS[m] = 1 - sumabs2(eLS) / tSS
		rSqLAD[m] = 1 - sumabs2(eLAD) / tSS
	end
	#Get kernel density plot of estimated coefficients (and include true value)
	println("Building coefficient kernel densities")
	aArr = Vector{Float64}[Float64[ coefLS[m][1] for m = 1:numResample ], Float64[ coefLAD[m][1] for m = 1:numResample ]]
	bArr = Vector{Float64}[Float64[ coefLS[m][2] for m = 1:numResample ], Float64[ coefLAD[m][2] for m = 1:numResample ]]
	aKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(aArr[k]) for k = 1:2 ]
	bKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(bArr[k]) for k = 1:2 ]
	aLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(aKDVec[k].x), y=aKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	bLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(bKDVec[k].x), y=bKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	aKernelPlot = plot(aLayerVec..., Guide.xlabel("Regression constant"), Guide.ylabel("Density"), Guide.title("Density of regression constant (true value = " * string(a) * ")"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	bKernelPlot = plot(bLayerVec..., Guide.xlabel("Regression coefficient"), Guide.ylabel("Density"), Guide.title("Density of regression coefficient (true value = " * string(b) * ")"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	println("Plotting coefficient kernel densities")
	draw_local(aKernelPlot, "Regression_constant_density", dirPath=outputDir, fileType=:svg)
	draw_local(bKernelPlot, "Regression_coefficient_density", dirPath=outputDir, fileType=:svg)
	#Get kernel density plot of r-squares
	println("Building rSquared kernel density")
	rSqArr = Vector{Float64}[rSqLS, rSqLAD]
	rSqKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(rSqArr[k]) for k = 1:2 ]
	rSqLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(rSqKDVec[k].x), y=rSqKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	rSqKernelPlot = plot(rSqLayerVec..., Guide.xlabel("R-squared"), Guide.ylabel("Density"), Guide.title("Density of regression R-squared (true value = " * string(rSquared) * ")"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	println("Plotting rSquared kernel density")
	draw_local(rSqKernelPlot, "Regression_rSquared_density", dirPath=outputDir, fileType=:svg)
	println("Hogg robust estimator quantiles (" * string(qProb) * ") = " * string(hoggVecQ))
	#Perform simulations using the estimated models and plot density of terminal values
	println("Performing simple trading simulations")
	println("Total number of observations = " * string(length(v)))
	iEstEnd = Int(round(length(v) / 2))
	terminalValLS = Array(Float64, numResample)
	terminalValLAD = Array(Float64, numResample)
	avgAbsErrorLS = Array(Float64, numResample)
	avgAbsErrorLAD = Array(Float64, numResample)
	for m = 1:numResample
		rMat = Array(Float64, length(r), numAsset)
		sMat = Array(Float64, length(r), numAsset)
		eMat = Array(Float64, length(r), numAsset)
		aVec = Array(Float64, numAsset)
		bVec = Array(Float64, numAsset)
		for j = 1:numAsset
			(rMat[:, j], sMat[:, j], eMat[:, j], aVec[j], bVec[j]) = simulate_signal_and_returns(v, rSquared=tradingSimRSquared, a=0.0)
		end
		coefLSVec = Array(Vector{Float64}, numAsset)
		coefLADVec = Array(Vector{Float64}, numAsset)
		rHatLSMat = Array(Float64, length(r), numAsset)
		rHatLADMat = Array(Float64, length(r), numAsset)
		for j = 1:numAsset
			y = rMat[:, j]
			includeConstantInSim ? (x = [ones(Float64, length(y)) sMat[:, j]]) : (x = sMat[:, j]'')
			yEst = y[1:iEstEnd]
			xEst = x[1:iEstEnd, :]
			ySim = y[iEstEnd+1:end]
			xSim = x[iEstEnd+1:end, :]
			coefLSVec[j] = xEst \ yEst
			(coefLADVec[j], _, _) = least_absolute_deviation(yEst, xEst)
			any(isnan(coefLADVec[j])) && (coefLADVec[j] = coefLSVec[j])
			rHatLSMat[:, j] = x * coefLSVec[j]
			rHatLADMat[:, j] = x * coefLADVec[j]
		end
		avgAbsErrorLS[m] = mean(abs(rHatLSMat[iEstEnd+1:end, :] - rMat[iEstEnd+1:end, :]))
		avgAbsErrorLAD[m] = mean(abs(rHatLADMat[iEstEnd+1:end, :] - rMat[iEstEnd+1:end, :]))
		terminalValLS[m] = very_simple_trading_sim(rMat[iEstEnd+1:end, :], rHatLSMat[iEstEnd+1:end, :])[end]
		terminalValLAD[m] = very_simple_trading_sim(rMat[iEstEnd+1:end, :], rHatLADMat[iEstEnd+1:end, :])[end]
	end
	println("Mean absolute forecast error (LS) = " * string(mean(avgAbsErrorLS)))
	println("Mean absolute forecast error (LAD) = " * string(mean(avgAbsErrorLAD)))
	println("Proportion of samples when LAD has smaller absolute forecast error = " * string(sum(avgAbsErrorLAD .< avgAbsErrorLS) / numResample))
	println("Mean terminal value (LS) = " * string(mean(terminalValLS)))
	println("Mean terminal value (LAD) = " * string(mean(terminalValLAD)))
	println("Median terminal value (LS) = " * string(median(terminalValLS)))
	println("Median terminal value (LAD) = " * string(median(terminalValLAD)))
	sort!(terminalValLS)
	sort!(terminalValLAD)
	iLower = Int(floor(trimSimTail * length(terminalValLS)))
	iLower == 0 && (iLower = 1)
	iUpper = length(terminalValLS) - iLower + 1
	terminalValLS = terminalValLS[iLower:iUpper]
	terminalValLAD = terminalValLAD[iLower:iUpper]
	println("Plotting terminal value kernel densities")
	tV = Vector{Float64}[terminalValLS, terminalValLAD]
	tVKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(tV[k]) for k = 1:2 ]
	tVLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(tVKDVec[k].x), y=tVKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	tVKernelPlot = plot(tVLayerVec..., Guide.xlabel("Terminal value"), Guide.ylabel("Density"), Guide.title("Density of terminal values from simple simulations"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	draw_local(tVKernelPlot, "Simulated_portfolio_terminal_values", dirPath=outputDir, fileType=:svg)
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
# ASYMMETRIC DISTRIBUTIONS IN FINANCIAL RETURNS
#-------------------------------------------
function asymmetric_distribution_effect( ; blockLength::Float64=5.0, numResample::Int=1000, varProxyMethod::Symbol=:historicvariance, numObs::Int=2000, rSquared::Float64=0.05, a::Float64=0.0, lag::Int=17, numAsset::Int=20, tradingSimRSquared::Float64=0.01, includeConstantInSim::Bool=true)
	#Get robust measure of return data and compare it to Normal and t-Dist with 2 DoF
	secRet = read_local(secList, :return)
    robustSkew = Float64[ robust_skew(secRet[j]) for j = 1:length(secList) ]
	logNormalRobustSkew = mean(Float64[ robust_skew(rand(LogNormal(0, 1), 100)) for k = 1:1000 ])
	println("Drawing plot 1")
	estPlot1 = plot(x=secList, y=robustSkew, yintercept=[0.0, logNormalRobustSkew], Geom.point, Geom.hline, Guide.xlabel("Ticker code"), Guide.ylabel("Robust skewness"), defaultThemeOverride)
    draw_local(estPlot1, "Robust_Skewness_of_Daily_Financial_Returns", dirPath=outputDir, fileType=:svg)
	#Compare mean and trimmed mean on resampled financial returns (use WOW (Largest right skew) and ANZ (Largest left skew))
	println("Drawing mean comparison plots (full period)")
	pWOWFull = kernel_plot_robust_mean_versus_trimmed_mean(secRet, "WOW", blockLength=blockLength)
	draw_local(pWOWFull, "WOW_Bootstrapped_Location_Estimator_Density_(full_period)", dirPath=outputDir, fileType=:svg)
	pANZFull = kernel_plot_robust_mean_versus_trimmed_mean(secRet, "ANZ", blockLength=blockLength)
	draw_local(pANZFull, "ANZ_Bootstrapped_Location_Estimator_Density_(full_period)", dirPath=outputDir, fileType=:svg)
	#Compare mean and trimmed mean on log-normal data
	kPlotLN = kernel_plot_robust_mean_versus_trimmed_mean(rand(LogNormal(0, 1), 100), bpMult=1, blockLength=1.0)
	draw_local(kPlotLN, "LogNormal_Bootstrapped_Location_Estimator_Density", dirPath=outputDir, fileType=:svg)
	println("Routine complete")
end





#--------------------------------------------
# ROBUST FUNCTIONS USED THROUGHOUT
#--------------------------------------------
#Number of observations to cut when trimming
tmean_num_cut{T}(x::AbstractVector{T}, p::Float64) = Int(floor(p*length(x)))
tmean_num_cut{T}(x::AbstractVector{T}, lp::Float64, up::Float64) = (tmean_num_cut(x, lp), tmean_num_cut(x, up))
#Trimmed mean
function tmean!{T}(x::AbstractVector{T}, lp::Float64=0.1, up::Float64=0.1 ; sorted::Bool=false)
    length(x) < 2 && error("Input must contain at least 2 observations")
    !(0.0 <= lp <= up) && error("Invalid lower portion")
	!(lp <= up <= 1.0) && error("Invalid upper portion")
    (numLower, numUpper) = tmean_num_cut(x, lp, up)
	!sorted && sort!(x)
	return(mean(sub(x, numLower+1:length(x)-numUpper)))
end
tmean!{T}(x::AbstractVector{T}, p::Float64=0.2 ; sorted::Bool=false) = tmean!(x, p/2, p/2, sorted=sorted)
tmean{T}(x::AbstractVector{T}, lp::Float64=0.1, up::Float64=0.1 ; sorted::Bool=false) = sorted ? tmean!(x, lp, up, sorted=true) : tmean!(deepcopy(x), lp, up, sorted=false)
tmean{T}(x::AbstractVector{T}, p::Float64=0.2 ; sorted::Bool=false) = tmean(x, p/2, p/2, sorted=sorted)
#Median when input is already sorted
function median_sorted{T}(x::AbstractVector{T})
    length(x) < 2 && error("Input must contain at least 2 observations")
    if iseven(length(x))
        i = Int(length(x)/2)
        return(mean(x[i:i+1]))
    else
        return(x[Int((length(x)+1)/2)])
    end
end
#Hogg's robust measure of tail-fatness.
function hogg_robust_kurt!{T}(x::Vector{T} ; sorted::Bool=false, numerTail::Float64=0.05, denomTail::Float64=0.5)
    length(x) < 10 && error("Input must contain at least 10 observations")
    !(0.0 < numerTail <= 0.2) && error("Non-sensible value for numerTail")
    !(0.3 < denomTail <= 0.5) && error("Non-sensible value for denomTail")
    numObsNumer = Int(ceil(numerTail * length(x)))
    numObsDenom = Int(ceil(denomTail * length(x)))
    !sorted && sort!(x)
    numerLeft = mean(sub(x, 1:numObsNumer))
    numerRight = mean(sub(x, length(x)-numObsNumer+1:length(x)))
    denomLeft = mean(sub(x, 1:numObsDenom))
    denomRight = mean(sub(x, length(x)-numObsDenom+1:length(x)))
    return((numerRight - numerLeft) / (denomRight - denomLeft))
end
hogg_robust_kurt{T}(x::Vector{T} ; sorted::Bool=false, numerTail::Float64=0.05, denomTail::Float64=0.5) = sorted ? hogg_robust_kurt!(x, sorted=true, numerTail=numerTail, denomTail=denomTail) : hogg_robust_kurt!(deepcopy(x), sorted=false, numerTail=numerTail, denomTail=denomTail)
#Robust skewness
function robust_skew{T<:Number}(x::Vector{T}; qProb::Float64=0.75)
	length(x) < 3 && error("Input must contain at least 10 observations")
	!(0.5 < qProb < 1.0) && error("Invalid quantile for robust skew procedure")
	qEst = quantile(x, Float64[1 - qProb, 0.5, qProb])
	return((qEst[3] + qEst[1] - 2*qEst[2]) / (qEst[3] - qEst[1]))
end


#--------------------------------------------
# ROLLING WINDOW HISTORICAL VARIANCE
#--------------------------------------------
historical_variance{T<:Number}(r::AbstractVector{T}, lagWindow::Int=100) = (0 < lagWindow < length(r)) ? Float64[ var(sub(r, n-lagWindow+1:n)) for n = lagWindow:length(r) ] : error("Invalid lagWindow")



#--------------------------------------------
# DATA READING FUNCTIONS USED THROUGHOUT
#--------------------------------------------
function read_local(secList::Vector{ASCIIString}, dataType::Symbol ; checkLength::Int=0)
    if dataType == :return
        x = Vector{Float64}[ vec(readcsv(dataDir*"ASX_"*secList[j]*"_Return-Trade_uMarketCloseAuction_Bow.csv", Float64)) for j = 1:length(secList) ]
    elseif dataType == :realisedvariance
        x = Vector{Float64}[ vec(readcsv(dataDir*"ASX_"*secList[j]*"_RealisedVariance-5Minute-BidAskMidpoint_uFirstToLast_Bow.csv", Float64)) for j = 1:length(secList) ]
    else
        error("Invalid data type")
    end
    if checkLength != 0
        any(Int[ length(x[j]) for j = 1:length(x) ] .!= checkLength) && error("Invalid data length")
    end
    return(x)
end
function read_local(dataType::Symbol)
    if dataType == :calendar
        x = vec(readcsv(dataDir*"Calendar.csv", DateTime))
    else
        error("Invalid data type")
    end
    return(x)
end


#--------------------------------------------
# PLOTTING FUNCTIONS USED THROUGHOUT
#--------------------------------------------
#Function for drawing plot to saved image file
function draw_local(p::Plot, fileName::ASCIIString ; dirPath::ASCIIString="", fileType::Symbol=:png, width::Measure=40cm, height::Measure=20cm)
	dirPath[end] != '/' && (dirPath = dirPath * "/")
    !isdir(dirPath) && error("Input directory does not exist")
    filePath = dirPath * fileName * "." * string(fileType)
    if fileType == :svg     ; draw(SVG(filePath, width, height), p)
	elseif fileType == :png ; draw(PNG(filePath, width, height), p)
	elseif fileType == :pdf ; draw(PDF(filePath, width, height), p)
	else                    ; error("Invalid fileType symbol")
	end
	return(filePath)
end
function adjust_default_theme_color(x::Theme, colourString::ASCIIString)
    xC = deepcopy(x)
    xC.default_color = parse(Compose.Colorant, colourString)
    return(xC)
end
function default_legend(legendLabel::Vector{ASCIIString})
	legendTitle = "Legend:"
    length(legendLabel) > length(colourVec) && error("Default legend function cannot handle more than " * string(length(colourVec)) * " series")
    return(legendTitle, legendLabel, deepcopy(colourVec[1:length(legendLabel)]))
end

#--------------------------------------------
# OTHER COMMON FUNCTIONS
#--------------------------------------------
arr_to_mat{T}(x::Vector{Vector{T}}) = T[ x[j][k] for j = 1:length(x), k = 1:length(x[1])]
mat_to_arr{T}(x::Matrix{T}) = Vector{T}[ x[:, j] for j = 1:size(x, 2) ]
julia_version_dir() = "v"*string(VERSION.major)*"."*string(VERSION.minor)






end # module
