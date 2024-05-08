# How to install
This package contains the example code of integrating R with pre-trained neural network models for phylogenetic tree parameter estimation. Only R interface is provided in the code, although utilized both R and Python. 

**Step 1: R and Python**

The latest versions of R and Python should be installed correctly. It is recommended to not alter the default settings of the installers. 

[Click to open R official website](https://cran.r-project.org/)

[Click to open Python official website](https://www.python.org/downloads/)

The latest versions of Rtools and pip are also required. Rtools can be installed manually from the above website, pip is usually bundled with Python installation.

**Step 2: Set up R packages**

(In R)
```r
install.packages("devtools")
remotes::install_github("thijsjanzen/treestats")
remotes::install_github("EvoLandEco/EvoNN")
```

**Step 3: Set up Python libraries**

Is is also recommended to create an independent Python virtual environment for sanity. The code implementation currently only supports virtual environment, you have to follow the steps anyway.

(In Terminal or PowerShell)
1. Create a python virtual enviroment, subsitute <...> with your own choice:
```
python -m venv <path_to_new_virtual_environment>
```
2. After creation, activate your virtual enviroment, subsitute <...> with your previously set path, see [here](https://docs.python.org/3/library/venv.html) for help if neither works for you. See [here](https://superuser.com/questions/106360/how-to-enable-execution-of-powershell-scripts) if you encounter error when executing the command in PowerShell.
```
# If in Windows PowerShell, use
<path_to_venv>\Scripts\Activate.ps1

# If in UNIX-like systems, use
source <path_to_venv>/bin/activate
```

3. Install required libraries in the virtual environment:
```
pip install torch torch_geometric pandas pyreadr
```

4. Deactivate the virtual environment:
```
deactivate
```

# How to use

Create a folder containing all the tree files to be estimated, the trees must be in newick format. Also make sure the trees are fully bifurcated and ultrametric. A validity check will be performed on all the trees, any tree being rejected will be reported with a reason.
Use the **path to nexus trees** and the **path to the previously created Python virtual environment** to call `parameter_estimation()`:

(In R)
```r
library(EvoNN)
library(ape)

path <- "path_to_nexus_trees"
venv <- "path_to_virtual_environment"

# Get some simulated trees
dir.create(path)
for (i in seq_len(20)) {
  phy <- rlineage(0.4, 0.2, 10)
  phy <- drop.fossil(phy)
  write.tree(phy, file.path(path, paste0("tree_", sample.int(1000,1))))
}

# Estimate parameters
result <- parameter_estimation(file_path = path, 
                     venv_path = venv,
                     scenario = "BD")
```
