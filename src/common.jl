#--------------------------------------------
# ROBUST FUNCTIONS USED THROUGHOUT
#--------------------------------------------
#Number of observations to cut when trimming
tmean_num_cut{T}(x::AbstractVector{T}, p::Float64) = Int(floor(p*length(x)))
tmean_num_cut{T}(x::AbstractVector{T}, lp::Float64, up::Float64) = (tmean_num_cut(x, lp), tmean_num_cut(x, up))
#Trimmed mean
function tmean!{T}(x::AbstractVector{T}, lp::Float64=0.1, up::Float64=0.1 ; sorted::Bool=false)
    !(0.0 <= lp <= up) && error("Invalid lower portion")
	!(lp <= up <= 1.0) && error("Invalid upper portion")
    (numLower, numUpper) = tmean_num_cut(x, lp, up)
	!sorted && sort!(x)
	return(mean(sub(x, numLower+1:length(x)-numUpper)))
end
tmean!{T}(x::AbstractVector{T}, p::Float64=0.2 ; sorted::Bool=false) = tmean!(x, p/2, p/2, sorted=sorted)
tmean{T}(x::AbstractVector{T}, lp::Float64=0.1, up::Float64=0.1 ; sorted::Bool=false) = sorted ? tmean!(x, lp, up, sorted=true) : tmean!(deepcopy(x), lp, up, sorted=false)
tmean{T}(x::AbstractVector{T}, p::Float64=0.2 ; sorted::Bool=false) = tmean(x, p/2, p/2, sorted=sorted)


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
