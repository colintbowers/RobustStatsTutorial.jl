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
	kernelPlot = plot(layerVec..., Guide.xlabel("Estimator value (basis points)"), Guide.ylabel("Density"), Guide.title("Location estimator densities for CCL return data"), Guide.manual_color_key(default_legend(["Mean", "Trimmed mean (0.4)"])...), defaultThemeOverride)
end


#-------------------------------------------
# ROBUST PREDICTION OF FINANCIAL RETURNS
#-------------------------------------------
function return_prediction_financial_data( ; blockLength::Float64=4.0)
	#Read in variance proxy
	varNAB = read_local(["NAB"], :realisedvariance)[1]
	#Simuate returns, signal, error term, constant and coefficient
	(r, s, e, a, b) = simulate_signal_and_returns(varNAB, rSquared=0.05, a=0.0)
	#Resample r, x, and e
	numResample = 1000
	inds = dbootstrapindex(length(r), numResample=numResample, blockLength=blockLength)
	rBoot = r[inds]
	sBoot = x[inds]
	eBoot = e[inds]
	#Run regressions on the resampled data using multiple methods and get plot of estimated coefficients
	coefLS = Array(Vector{Float64}, numResample)
	coefLAD = Array(Vector{Float64}, numResample)
	rSqLS = Array(Float64, numResample)
	rSqLAD = Array(Float64, numResample)
	for m = 1:numResample
		y = rBoot[:, m]
		x = [ones(Float64, length(y)) sBoot[:, m]]
		coefLS[m] = x \ y
		(coefLAD[m], _, _) = least_absolute_deviation(y, x)
		eLS = y - x * coefLS[m]'
		eLAD = y - x * coefLAD[m]'
		tSS = sumabs2(y)
		rSqLS[m] = 1 - sumabs2(eLS) / tSS
		rSqLAD[m] = 1 - sumabs2(eLAD) / tSS
	end
	#Get kernel density plot of estimated coefficients (and include true value)
	aArr = Vector{Float64}[Float64[ coefLS[m][1] for m = 1:numResample ], Float64[ coefLAD[m][1] for m = 1:numResample ]]
	bArr = Vector{Float64}[Float64[ coefLS[m][2] for m = 1:numResample ], Float64[ coefLAD[m][2] for m = 1:numResample ]]
	aKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(aArr[k]) for k = 1:2 ]
	bKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(bArr[k]) for k = 1:2 ]
	aLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(aKDVec[k].x), y=aKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	bLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(bKDVec[k].x), y=bKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	aKernelPlot = plot(aLayerVec..., Guide.xlabel("Regression constant"), Guide.ylabel("Density"), Guide.title("Density of regression constant"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	bKernelPlot = plot(bLayerVec..., Guide.xlabel("Regression coefficient"), Guide.ylabel("Density"), Guide.title("Density of regression coefficient"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	draw_local(aKernelPlot, "Regression_constant_density", dirPath=outputDir, fileType=:svg)
	draw_local(bKernelPlot, "Regression_coefficient_density", dirPath=outputDir, fileType=:svg)
	#Get kernel density plot of r-squares
	rSqArr = Vector{Float64}[rSqLS, rSqLAD]
	rSqKDVec = KernelDensity.UnivariateKDE{FloatRange{Float64}}[ kde(rSqArr[k]) for k = 1:2 ]
	rSqLayerVec = Vector{Gadfly.Layer}[ layer(x=collect(rSqKDVec[k].x), y=rSqKDVec[k].density, Geom.line, adjust_default_theme_color(defaultThemeOverride, colourVec[k])) for k = 1:2 ]
	rSqKernelPlot = plot(rSqLayerVec..., Guide.xlabel("R-squared"), Guide.ylabel("Density"), Guide.title("Density of regression R-squared"), Guide.manual_color_key(default_legend(["Least squares", "Least absolute deviations"])...), defaultThemeOverride)
	draw_local(rSqKernelPlot, "Regression_rSquared_density", dirPath=outputDir, fileType=:svg)
end
function simulate_signal_and_returns(v::Vector{Float64} ; rSquared::Float64=0.05, a::Float64=0.0)
	!(0.0 < rSquared < 1.0) && error("Invalid rSquared")
	e = sqrt(v) * randn(length(v))
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
	fObj = ((param, grad) -> sumabs(y - x*param') #Get local anonymous objective function
	initVal = zeros(Float64, size(x, 2))
	opt = Opt(:LN_COBYLA, length(initVal)) #Derivative-free convergence method (yes I'm being lazy here)
	xtol_rel!(opt, 1e-6)
	min_objective!(opt, fObj)
	#Perform optimisation
	(objFuncOpt, paramOpt, flag) = optimize(opt, initialValue)
	verbose && println("    Output flag = " * string(flag))
	checkNLoptFlag(flag)
	return(paramOpt, objFuncOpt, y - x*paramOpt') #Return optimal parameter, objective function at optimum, and estimated errors
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



#-------------------------------------------
#
#-------------------------------------------




end # module
