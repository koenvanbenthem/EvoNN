bd_estimation <- NULL
ddd_estimation <- NULL

.onLoad <- function(libname, pkgname){
  # Read package version list
  pkglist <- read.csv(system.file("pkglist.csv", package = "EvoNN"), row.names = 1)

  # Check if the EvoNN virtual environment exists
  env_exists <- reticulate::virtualenv_exists("EvoNN")
  if (env_exists) {
    # Check if the virtual environment has the required versions of the packages
    current_pkgs <- reticulate::py_list_packages("EvoNN")
    pkgs_matched <- TRUE
    reinstall_list <- c()
    for (pkg in pkglist$package) {
      if (pkg %in% current_pkgs$package) {
        if (current_pkgs[which(current_pkgs$package==pkg),]$version != pkglist[which(current_pkgs$package==pkg),]$version) {
          pkgs_matched <- FALSE
          reinstall_list <- c(reinstall_list, paste0(pkg, "==", pkglist[which(current_pkgs$package==pkg),]$version))
          break
        }
      } else {
        pkgs_matched <- FALSE
        reinstall_list <- c(reinstall_list, paste0(pkg, "==", pkglist[which(current_pkgs$package==pkg),]$version))
        break
      }
    }

    # Reinstall the packages if they do not match
    if (!pkgs_matched) {
      message("Updating Python virtual environment: EvoNN")
      reticulate::py_install(reinstall_list, envname = "EvoNN")
    } else {
      message("Using existing Python virtual environment: EvoNN")
    }
  } else {
    # Create the virtual environment if it does not exist
    message("Preparing Python virtual environment, this may take a while the first time the function is run...")
    install_list <- paste0(pkglist$package, "==", pkglist$version)
    reticulate::virtualenv_create("EvoNN", packages = install_list)
  }

  # Use the EvoNN virtual environment
  reticulate::use_virtualenv("EvoNN")

  # Import Python dependencies
  reticulate::source_python(system.file(paste0("model/", "import.py"), package = "EvoNN"))

  # Import the EvoNN model
  reticulate::source_python(system.file(paste0("model/", "nn_model.py"), package = "EvoNN"))

  # Assign the estimation functions to the global environment
  bd_estimation <<- reticulate::py$bd_estimation
  ddd_estimation <<- reticulate::py$ddd_estimation
}