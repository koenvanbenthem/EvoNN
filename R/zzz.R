bd_estimation <- NULL
ddd_estimation <- NULL

.onLoad <- function(libname, pkgname){
  # Check if the EvoNN virtual environment exists
  if (reticulate::virtualenv_exists("EvoNN")) {
    reticulate::use_virtualenv("EvoNN")
    message("Using existing Python virtual environment: EvoNN")
  } else {
    message("Preparing Python virtual environment, this may take a while the first time the function is run...")
    reticulate::virtualenv_create("EvoNN", packages = c("torch", "torch_geometric", "pandas", "numpy<2"))
    reticulate::use_virtualenv("EvoNN")
  }

  # Import Python dependencies
  reticulate::source_python(system.file(paste0("model/", "import.py"), package = "EvoNN"))

  # Import the EvoNN model
  reticulate::source_python(system.file(paste0("model/", "nn_model.py"), package = "EvoNN"))

  # Assign the estimation functions to the global environment
  bd_estimation <<- reticulate::py$bd_estimation
  ddd_estimation <<- reticulate::py$ddd_estimation
}