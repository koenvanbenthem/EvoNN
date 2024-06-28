<!-- badges: start -->
[![R-CMD-check](https://github.com/EvoLandEco/EvoNN/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/EvoLandEco/EvoNN/actions/workflows/R-CMD-check.yaml)
![GitHub Release](https://img.shields.io/github/v/release/EvoLandEco/EvoNN?include_prereleases)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

# EvoNN: Neural Network Parameter Estimation for Phylogenetic Trees

## How to install
This package illustrates how to integrate `R` with pre-trained neural network models for phylogenetic tree parameter estimation. Only `R` interface is provided in the package, although utilized both `R` and `Python`.

**Step 1: Install R and Python**

The latest versions of `R` and `Python` should be installed correctly. It is recommended to not alter the default settings of the installers. 

[Click to open R official website](https://cran.r-project.org/)

[Click to open Python official website](https://www.python.org/downloads/)

The latest versions of `Rtools` and `pip` are also required. `Rtools` can be installed manually from the above website, `pip` is usually bundled with Python installation.

**Step 2: Install R packages**
```r
install.packages("devtools")
remotes::install_github("EvoLandEco/treestats")
remotes::install_github("EvoLandEco/EvoNN")
```
---

## How to use
**Step 1: Prepare the phylogeny**

[Click Here](https://github.com/user-attachments/files/16026922/Columbiformes.zip) to download an example phylogeny by [Valente et al 2020 Nature](https://data.mendeley.com/datasets/p6hm5w8s3b/2 ), unzip to your favorite path.
```r
library(EvoNN)
# Set the path to the tree file
path <- "your_path/Columbiformes.tre"
# Read in tree file if it is in Newick format, e.g. the example phylogeny
test_tre <- ape::read.tree(path)
# Otherwise, use the following to read in Nexus format
# test_tre <- ape::read.nexus(path)
```
**Step 2: Estimate parameters**

This step may take a while to prepare a virtual environment the *first time* the function is run. *Wait for the completion message.*
```r
result <- nn_estimate(test_tre, scenario = "DDD")
```
**Step 3: Bootstrap the uncertainty**

The execution time depends on the estimated parameters, some parameter settings may take too long or fail consistently. It is recommended to set a timeout limit, bootstrap iterations exceed the limit will return NAs.
```r
bootstrap <- nn_bootstrap_uncertainty(result, n = 100, timeout = 30)
```
**Step 4: Plot the uncertainty**

The neural network estimates are indicated by the red dashed lines and the blue values. Their uncertainties are represented by the density curves of bootstrap results.
```r
nn_plot_bootstrap(bootstrap, result, scenario = "DDD")
```
![Plot for Bootstrap Results](https://github.com/EvoLandEco/EvoNN/assets/57348932/a98d521a-63a4-47d0-84c6-ccaaed74d6ea)


---

## Known issue

- Error messages relating to `numpy` version or installation could appear if you run the `parameter_estimation` function for the first time. The issue might be newly installed Python libraries not being loaded. There is a simple solution: fully close your IDE (e.g. R Studio, VS Code or DataSpell, the software you use for R coding), then open it again. Re-run the example code, it should work this time.

- Mac users might have to install `cmake` to build some dependencies from source code. Run `install brew install --cask cmake` in your terminal to install it.

---

## Important note

The `parameter_estimation()` function will automatically set up a Python virtual environment named "EvoNN" in your home directory in its first run, this may take a while.

You can manually remove this virtual environment by running the following code in R:

```r
reticulate::virtualenv_remove("EvoNN")
```
