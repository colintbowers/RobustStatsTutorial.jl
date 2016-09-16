# RobustStatsTutorial

This repository contains all the files and source code used in Colin T. Bowers presentation titled "Robust Statistics and Financial Data" at the JP Morgan Quantference (September 2016, Sydney). The source code is written in Julia and will require a local installation of Julia in order to run. It was written for v0.4, and may encounter problems in later versions of Julia. For example, the source code uses the type `ASCIIString` which is deprecated in v0.5+. The Julia language has similar syntax to Matlab (in some ways) so any users familiar with Matlab should be able to translate it without too much difficulty.

If you have a local installation of Julia, you can download the repository using `Pkg.clone("https://github.com/colintbowers/RobustStatsTutorial.jl.git")` at the Julia REPL.

If you do not have a Julia installation, but are interested in files other than the source code (i.e. csv data files, presentation images, or TeX source) then it is probably easiest to just use the green "Clone or download" button on the main repository webpage, and choose the "Download ZIP" option. Then just extract the files locally.

NOTE: I haven't bothered adding Travis CI to this project as it is just a single self contained source code file which I don't ever intend to revisit.
