get_all_neighbors_distances <- function(tree) {
  # Initialize an empty list to store neighbors and edge lengths
  all_neighbors <- vector("list", ape::Nnode(tree) + ape::Ntip(tree))

  # Iterate through the edge matrix
  for (i in seq_len(nrow(tree$edge))) {
    from_node <- tree$edge[i, 1]
    to_node <- tree$edge[i, 2]
    edge_length <- tree$edge.length[i]

    # Append to_node and edge_length to the neighbor list of from_node
    if (is.null(all_neighbors[[from_node]])) {
      all_neighbors[[from_node]] <- setNames(vector("numeric", 0), character(0))
    }
    all_neighbors[[from_node]][as.character(to_node)] <- edge_length

    # Append from_node and edge_length to the neighbor list of to_node
    if (is.null(all_neighbors[[to_node]])) {
      all_neighbors[[to_node]] <- setNames(vector("numeric", 0), character(0))
    }
    all_neighbors[[to_node]][as.character(from_node)] <- edge_length
  }

  # Name the list elements for clarity
  names(all_neighbors) <- 1:(ape::Nnode(tree) + ape::Ntip(tree))

  return(all_neighbors)
}


tree_to_adj_mat <- function(tree, master = FALSE) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  neighbor_dists <- get_all_neighbors_distances(tree)

  padded_dists <- lapply(neighbor_dists, function(x) {
    if (length(x) == 1) {
      x <- c(x, 0, 0)  # Add two zeros after if length is 1
    }
    if (length(x) == 2) {
      x <- c(0, x)  # Add one zero before if length is 2
    }
    return(x)
  })

  neighbor_matrix <- do.call(rbind, padded_dists)
  colnames(neighbor_matrix) <- NULL

  if (master) {
    neighbor_matrix <- rbind(neighbor_matrix, rep(0, ncol(neighbor_matrix)))
  }

  return(neighbor_matrix)
}


tree_to_connectivity <- function(tree, undirected = FALSE, master = FALSE) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  out <- NULL

  if (undirected) {
    part_a <- tree$edge - 1
    part_b <- cbind(part_a[, 2], part_a[, 1])
    part_ab <- rbind(part_a, part_b)
    out <- part_ab
  } else {
    out <- tree$edge - 1
  }

  if (master && undirected) {
    # Nnode + 1 is the number of tips, Nnode + 2 is the index of the root node
    # Nnode + 3 is the starting index of the internal node
    # We need Nnode + 3 - 1 because of 0-based indexing in Python, thus Nnode + 2
    start_id <- tree$Nnode + 2

    # Similarly, Nnode * 2 + 1 - 1 is the ending index of the internal node
    end_id <- tree$Nnode * 2

    # Index for the new master node
    master_id <- tree$Nnode * 2 + 1

    # Add the master node to the connectivity matrix
    new_part_a <- cbind(start_id:end_id, rep(master_id, times = end_id - start_id + 1))
    new_part_b <- cbind(new_part_a[, 2], new_part_a[, 1])
    new_part_ab <- rbind(new_part_a, new_part_b)

    out <- rbind(out, new_part_ab)
  }

  if (master && !undirected) {
    stop("Master node is currently only supported for undirected trees.")
  }

  return(out)
}


rescale_crown_age <- function(tree, target_crown_age) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  # Calculate the current crown age of the tree
  current_crown_age <- max(ape::node.depth.edgelength(tree))

  # Calculate the scaling factor
  scale_factor <- target_crown_age / current_crown_age

  # Scale the tree
  scaled_tree <- tree
  scaled_tree$edge.length <- scaled_tree$edge.length * scale_factor

  return(scaled_tree)
}


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
