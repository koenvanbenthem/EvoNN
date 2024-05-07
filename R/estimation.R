check_tree_validity <- function(file_path) {
  # Try reading the tree from file path using ape::read.tree
  tryCatch({
    tree <- NULL
    tryCatch({
      tree <- ape::read.nexus(file_path)
    }, error = function(e) {
      return("Error: Not a valid tree")
    }, warning = function(w) {
    })

    # Check format
    is_phylo <- FALSE
    if (inherits(tree, "phylo")) {
      is_phylo <- TRUE
    } else {
      return("Error: Tree is not in phylo format")
    }

    # Check topology
    is_binary <- ape::is.binary(tree)
    is_ultrametric <- ape::is.ultrametric(tree)

    if (!is_binary || !is_ultrametric) {
      return("Error: Tree is not binary or ultrametric")
    }

    # Check time calibration
    is_time_calibrated <- FALSE
    if (!is.null(tree$edge.length)) {
      is_time_calibrated <- TRUE
    }

    if (!is_time_calibrated) {
      return("Error: Tree is not time calibrated")
    }

    # Check tree size
    n_nodes <- 2 * tree$Nnode + 1
    if (n_nodes < 10 || n_nodes > 2000) {
      return("Error: Tree size is not within the range [10, 2000]")
    }

    return("SIG_SUCCESS")
  }, error = function(e) {
    return(e$message)
  }, warning = function(w) {})
}


write_tree_to_temp <- function(file_path) {
  require(ape)  # Ensure ape package is loaded

  # Attempt to read the Nexus tree file
  tree <- ape::read.nexus(file_path)
  message("Attempting to export the tree to temporary files")

  # Define the base temporary directory
  temp_dir <- tempdir()
  i <- 1
  path <- file.path(temp_dir, paste0("GNN", i, "/tree/"))

  # Find a unique directory by incrementing i if the directory already exists
  while (dir.exists(path)) {
    i <- i + 1
    path <- file.path(temp_dir, paste0("GNN", i, "/tree/"))
  }

  # Create the main directory and subdirectories
  path_EL <- file.path(path, "EL/")
  path_ST <- file.path(path, "ST/")
  path_BT <- file.path(path, "BT/")

  # Using a function to streamline directory creation
  create_dir <- function(dir_path) {
    if (!dir.exists(dir_path)) {
      dir.create(dir_path, recursive = TRUE)
    }
  }

  create_dir(path)
  create_dir(path_EL)
  create_dir(path_ST)
  create_dir(path_BT)

  # Define and save RDS files
  file_name <- file.path(path, paste0("tree_temp_1.rds"))
  saveRDS(tree_to_connectivity(tree, undirected = FALSE), file = file_name)

  file_name_el <- file.path(path_EL, "EL_temp_1.rds")
  saveRDS(tree_to_adj_mat(tree), file = file_name_el)

  file_name_st <- file.path(path_ST, "ST_temp_1.rds")
  saveRDS(tree_to_stats(tree), file = file_name_st)

  file_name_bt <- file.path(path_BT, "BT_temp_1.rds")
  saveRDS(tree_to_brts(tree), file = file_name_bt)

  # Return the index of the folder used
  return(i)
}



check_nn_model <- function(scenario = stop("Scenarios not specified")) {
  if (scenario == "DDD") {
    if (file.exists(system.file("model/DDD_FREE_TES_model_diffpool_2_gnn.pt", package = "eveGNN"))) {
      message("Found pre-trained GNN model for DDD")
    } else {
      stop("Missing pre-trained GNN model for DDD")
    }
    if (file.exists(system.file("model/DDD_FREE_TES_gnn_2_model_lstm.pt", package = "eveGNN"))) {
      message("Found pre-trained LSTM boosting model for DDD")
    } else {
      stop("Missing pre-trained LSTM boosting model for DDD")
    }
    if (file.exists(system.file("model/ddd_boosting_gnn_lstm.py", package = "eveGNN"))) {
      message("Found Python script for parameter estimation")
    } else {
      stop("Missing Python script for parameter estimation")
    }
  } else if (scenario == "BD") {

  } else {
    stop("Invalid scenario")
  }
}


parameter_estimation <- function(file_path = stop("Tree file path not provided"),
                                 venv_path = stop("Python virtual environment path not provided"),
                                 scenario = "DDD") {
  if (!(scenario %in% c("BD", "DDD"))) stop("Invalid scenario, should be either 'BD' or 'DDD'")
  message(paste0("Estimating under the ", scenario, " scenario"))
  file_list <- setdiff(list.files(file_path, full.names = TRUE), list.dirs(file_path, recursive = FALSE, full.names = TRUE))

  if (length(file_list) == 0) stop("No (visible) file found in the given path")

  signals <- vector("character", length(file_list))
  for (i in seq_along(file_list)) {
    signals[i] <- check_tree_validity(file_list[i])
  }

  success_indexes <- which(signals == "SIG_SUCCESS")
  failed_indexes <- which(signals != "SIG_SUCCESS")

  if (length(failed_indexes) > 0) {
    message("The following files are not valid and excluded from estimation:")
    for (i in failed_indexes) {
      print(paste0(file_list[i], " Reason: ", signals[i]))
    }
  }

  unique_i <- 0

  for (i in success_indexes) {
    unique_i <- write_tree_to_temp(file_list[i])
  }

  message("Tree exported")
  check_nn_model(scenario)

  message("Activating Python virtual environment")
  reticulate::use_virtualenv(venv_path)

  message("Estimating parameters")
  reticulate::py_run_string("import sys")
  # Simulating passing arguments
  system_path <- system.file("model", package = "eveGNN")
  temp_path <- gsub("\\\\", "/", tempdir())
  reticulate::py_run_string(paste0("sys.argv = ['", system_path, "', '", temp_path, "', ", unique_i, "]"))
  reticulate::py_run_file(system.file(paste0("model/", tolower(scenario), "_boosting_gnn_lstm.py"), package = "eveGNN"))

  out <- readRDS(file.path(tempdir(), paste0("empirical_gnn_2_lstm_result_", tolower(scenario), ".rds")))

  message("Result retrieved")

  return(out)
}

