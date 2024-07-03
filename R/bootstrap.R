#' Function to bootstrap the uncertainty of the EvoNN estimates
#'
#' Generates posterior distributions of the speciation rate, extinction rate, and carrying capacity (DDD) of a
#' phylogenetic tree by bootstrapping the tree and estimating the parameters using EvoNN
#' @param estimate The EvoNN estimate of the speciation rate, extinction rate, and carrying capacity (DDD)
#' @param scenario The diversification scenario: \cr \code{"BD"} : Birth-death
#' diversification \cr \code{"DDD"} : Diversity-dependent diversification
#' @param n The number of bootstrap iterations
#' @param timeout The maximum time in seconds for each iteration
#' @return \item{ results_NN }{ A data frame with the following columns: \cr
#' \code{lambda} : Speciation rate \cr \code{mu} : Extinction rate \cr \code{cap} : Carrying capacity
#' (Only for DDD) \cr }
#' @author Tianjian Qin
#' @export nn_bootstrap_uncertainty
nn_bootstrap_uncertainty <- function(estimate, scenario = "DDD", n = 100, timeout = 30) {
  if (scenario != "DDD") {
    stop("Only DDD is supported for boostrapping")
  }
  message("Performing ", n, " bootstrap iterations")
  message("Iteration longer than ", timeout, " seconds will be skipped, and the result will be NA")

  results_NN <- data.frame(lambda = numeric(n), mu = numeric(n), cap = numeric(n), stringsAsFactors = FALSE)

  pb <- utils::txtProgressBar(min = 0,
                       max = n,
                       style = 3,
                       width = n, # Needed to avoid multiple printings
                       char = "=")

  init <- numeric(n)
  end <- numeric(n)

  for (i in seq_len(n)) {
    init[i] <- Sys.time()
    tree <- bootstrap_core(estimate, scenario, n, timeout)

    if (!is.null(tree)) {
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

      system_path <- system.file("model", package = "EvoNN")

      result_NN <- ddd_estimation(system_path, py_tree_nd, py_tree_el, py_tree_st, py_tree_bt, py_scale)
      results_NN[i, ] <- c(result_NN$pred_lambda, result_NN$pred_mu, result_NN$pred_cap)
    } else {
      results_NN[i, ] <- c(NA, NA, NA)
    }

    end[i] <- Sys.time()
    utils::setTxtProgressBar(pb,i)
    time <- round(sum(end - init), 0)
    est <- n * (mean(end[end != 0] - init[init != 0])) - time
    remainining <- round(est, 0)
    cat(paste(" // Execution time:", time, "S",
              " // Estimated time remaining:", remainining, "S"), "")
  }

  close(pb)

  return(results_NN)
}


bootstrap_core <- function(estimate, scenario, n, timeout) {
    tree <- tryCatch({
      R.utils::withTimeout({
        sim <- dd_sim(c(estimate$pred_lambda, estimate$pred_mu, estimate$pred_cap), age = 10, ddmodel = 1)
        tree <- sim$tes
        while ((tree$Nnode + 1) < 10) {
          sim <- dd_sim(c(estimate$pred_lambda, estimate$pred_mu, estimate$pred_cap), age = 10, ddmodel = 1)
          tree <- sim$tes
        }
        tree
      }, timeout = timeout)
    }, TimeoutException = function(ex) {
      warning("Timeout")
      NULL
    }, error = function(e) {
      warning(e$message)
      NULL
    })

  return(tree)
}
