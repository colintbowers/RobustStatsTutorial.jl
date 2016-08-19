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
