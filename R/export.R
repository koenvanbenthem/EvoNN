#' @export tree_to_node_feature
tree_to_node_feature <- function(tree) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  ntips <- tree$Nnode + 1

  # Assign 0 to root node, 1 to internal nodes, and 2 to tips
  node_feature <- c(rep(2, ntips), 0, rep(1, tree$Nnode - 1))

  return(node_feature)
}


#' @export tree_to_edge_feature
tree_to_edge_feature <- function(tree, undirected = FALSE) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  if (undirected) {
    return(rbind(tree$edge.length, tree$edge.length))
  } else {
    return(tree$edge.length)
  }
}


#' @export tree_to_stats
tree_to_stats <- function(tree) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  # Check if the tree is ultrametric, if not return 0
  if (!ape::is.ultrametric(tree)) {
    return(0.00)
  }

  # TODO: Maintain a named list of the stats used, ensure they are consistent
  stats <- treestats::calc_all_stats(tree)

  return(unlist(stats))
}


# TODO: Also export LTT along with branching times (I doubt this will improve the performance)
#' @export tree_to_brts
tree_to_brts <- function(tree) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  # Check if the tree is ultrametric, if not return 0
  if (!ape::is.ultrametric(tree)) {
    return(0.00)
  }

  brts <- sort(treestats::branching_times(tree), decreasing = TRUE)

  return(brts)
}
