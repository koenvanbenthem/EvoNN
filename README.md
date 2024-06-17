# How to install
This package contains the example code of integrating R with pre-trained neural network models for phylogenetic tree parameter estimation. Only R interface is provided in the code, although utilized both R and Python. 

**Step 1: Install R and Python**

The latest versions of R and Python should be installed correctly. It is recommended to not alter the default settings of the installers. 

[Click to open R official website](https://cran.r-project.org/)

[Click to open Python official website](https://www.python.org/downloads/)

The latest versions of Rtools and pip are also required. Rtools can be installed manually from the above website, pip is usually bundled with Python installation.

**Step 2: Install R packages**

(In R)
```r
install.packages("devtools")
remotes::install_github("EvoLandEco/treestats")
remotes::install_github("EvoLandEco/EvoNN")
```

# How to use

The phylogeny to be estimated must be in newick format. If not sure, read the phylogeny in R and write to a file with the `write.tree()` function from package `ape`. Make sure the phylogeny is fully bifurcated and ultrametric. A validity check will be performed, in case when the phylogeny is rejected, a reason will be reported.

(In R)
```r
library(EvoNN)

path <- "path_to_the_phylogeny"

# Estimating parameters under a birth-death scenario:
result1 <- parameter_estimation(file_path = path, scenario = "BD")

# Estimating parameters under a diversity-dependent diversification scenario:
result2 <- parameter_estimation(file_path = path, scenario = "DDD")
```

# Important note

The `parameter_estimation()` function will automatically set up a Python virtual environment named "EvoNN" in your home directory in its first run, this may take a while.

You can manually remove this virtual environment by running the following code in R:

```r
reticulate::virtualenv_remove("EvoNN")
```
