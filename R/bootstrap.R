#' @export estimation_bootstrap
estimation_bootstrap <- function(estimate, scenario = "DDD", n = 100) {
  if (scenario != "DDD") {
    stop("Only DDD is supported for boostrapping")
  }

  results <- data.frame(lambda = numeric(n), mu = numeric(n), cap = numeric(n), stringsAsFactors = FALSE)
  for (i in seq_len(n)) {
    tree <- dd_sim(c(estimate$pred_lambda, estimate$pred_mu, estimate$pred_cap), age = 10, ddmodel = 1)$tes
    result <- suppressMessages(estimate_from_simulation(tree))
    message("Bootstrap iteration ", i, " completed")
    results[i, ] <- c(result$pred_lambda, result$pred_mu, result$pred_cap)
  }

  return(results)
}
