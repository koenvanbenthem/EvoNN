check_tree_validity <- function(tree) {
  # Check format
  if (!inherits(tree, "phylo")) {
    stop("Error: Tree is not in phylo format")
  }

  # Check topology
  if (!ape::is.binary(tree) || !ape::is.ultrametric(tree)) {
    stop("Error: Tree is not binary or ultrametric")
  }

  # Check tree size
  n_nodes <- 2 * tree$Nnode + 1
  if (n_nodes < 10 || n_nodes > 2000) {
    stop("Error: Tree size is not within the range [6, 1000]")
  }

  return("SIG_SUCCESS")
}


compute_scale <- function(tree) {
  current_age <- treestats::crown_age(tree)
  scale <- current_age / 10

  return(scale)
}

#' Function to estimate phylogenetic parameters using EvoNN
#'
#' Estimates the speciation rate, extinction rate, and carrying capacity (DDD) of a
#' phylogenetic tree using the EvoNN pre-trained models
#'
#'
#' @param tree A phylogenetic tree in phylo format
#' @param scenario The diversification scenario: \cr \code{"BD"} : Birth-death
#' diversification \cr \code{"DDD"} : Diversity-dependent diversification
#' @return \item{ out }{ A list with the following elements: \cr
#' \code{ pred_lambda } : Speciation rate \cr \code{ pred_mu } : Extinction rate \cr
#' \code{ pred_cap } : Carrying capacity (Only for DDD) \cr
#' \code{ scale } : A factor to scale the tree to the desired age (10) \cr
#' \code{ num_nodes } : Number of nodes in the tree \cr}
#' @author Tianjian Qin
#' @export nn_estimate
nn_estimate <- function(tree, scenario = "DDD") {
  if (!(scenario %in% c("BD", "DDD"))) stop("Invalid scenario, should be either 'BD' or 'DDD'")

  check_tree_validity(tree)

  if (reticulate::virtualenv_exists("EvoNN")) {
    reticulate::use_virtualenv("EvoNN")
    message("Using existing Python virtual environment: EvoNN")
  } else {
    message("Preparing Python virtual environment, this may take a while the first time the function is run...")
    reticulate::virtualenv_create("EvoNN", packages = c("torch", "torch_geometric", "pandas", "numpy<2"))
    reticulate::use_virtualenv("EvoNN")
  }

  tree <- rescale_crown_age(tree, 10)
  scale <- compute_scale(tree)

  tree_nd <- tree_to_connectivity(tree, undirected = FALSE)
  tree_el <- tree_to_adj_mat(tree)
  tree_st <- tree_to_stats(tree)
  tree_bt <- tree_to_brts(tree)

  py_tree_nd <- reticulate::r_to_py(tree_nd)
  py_tree_el <- reticulate::r_to_py(tree_el)
  py_tree_st <- reticulate::r_to_py(tree_st)
  py_tree_bt <- reticulate::r_to_py(tree_bt)
  py_scale <- reticulate::r_to_py(scale)
  message("Tree transferred to Python")

  system_path <- system.file("model", package = "EvoNN")

  message("Estimating parameters")
  if (scenario == "BD") out <- bd_estimation(system_path, py_tree_nd, py_tree_el, py_tree_st, py_tree_bt, py_scale)
  if (scenario == "DDD") out <- ddd_estimation(system_path, py_tree_nd, py_tree_el, py_tree_st, py_tree_bt, py_scale)

  message("Estimation complete")
  return(out)
}
