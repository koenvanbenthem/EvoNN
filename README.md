# How to install
Before start, R and Python should be installed correctly. It is recommended to not alter the default settings of the installers. This package only provides R interface, although utilized both R and Python.

**Set up R packages**

(In R)
```r
install.packages("devtools")
remotes::install_github("thijsjanzen/treestats")
remotes::install_github("EvoLandEco/EvoNN")
```

**Set up Python libraries**

(In Terminal or PowerShell)
1. Create a python virtual enviroment, subsitute <...> with your own choice:
```
python -m venv <path_to_new_virtual_environment>
```
2. After creation, activate your virtual enviroment, subsitute <...> with your previously set path, see https://docs.python.org/3/library/venv.html for help if neither works for you:
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

Create a folder containing all the tree files to be estimated, the trees must be in nexus format. Also make sure the trees are fully bifurcated and ultrametric. A validity check will be performed on all the trees, any tree being rejected will be reported with a reason.
Use the **path to nexus trees** and the **path to the previously created Python virtual environment** to call `parameter_estimation()`:

(In R)
```r
library(EvoNN)
result <- parameter_estimation(file_path = "path_to_nexus_trees", 
                     venv_path = "path_to_venv",
                     scenario = "BD")
```
