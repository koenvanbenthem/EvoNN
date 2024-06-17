#' @export estimation_bootstrap
estimation_bootstrap <- function(estimate, scenario = "DDD", n = 100) {
  if (scenario != DDD) {
    stop("Only DDD is supported for boostrapping")
  }

  results <- list()
  for (i in seq_len(100)) {
    tree <- dd_sim(c(estimate$pred_lambda, estimate$pred_mu, estimate$pred_cap), age = 10, ddmodel = 1)$tes
    result <- estimate_from_simulation(tree)
    results[[length(results) + 1]] <- result
  }

  results <- unlist(results)

  return(results)
}


