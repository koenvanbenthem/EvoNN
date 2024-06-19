#' @export estimation_bootstrap
estimation_bootstrap <- function(estimate, scenario = "DDD", n = 100) {
  if (scenario != "DDD") {
    stop("Only DDD is supported for boostrapping")
  }
  message("There are ", n, " bootstrap iterations")

  results_NN <- data.frame(lambda = numeric(n), mu = numeric(n), cap = numeric(n), stringsAsFactors = FALSE)
  results_ML_Typ <- data.frame(lambda = numeric(n), mu = numeric(n), cap = numeric(n), stringsAsFactors = FALSE)
  results_ML_Opt <- data.frame(lambda = numeric(n), mu = numeric(n), cap = numeric(n), stringsAsFactors = FALSE)
  for (i in seq_len(n)) {
    sim <- dd_sim(c(estimate$pred_lambda, estimate$pred_mu, estimate$pred_cap), age = 10, ddmodel = 1)
    tree <- sim$tes
    brts <- sim$brts
    result_NN <- suppressMessages(estimate_from_simulation(tree))
    result_ML_Typ <- suppressMessages(DDD::dd_ML(
      brts = brts,
      initparsopt = c(runif(1, 0.1, 4), runif(1, 0, 1.5), runif(1, 10, 1000)),
      idparsopt = c(1, 2, 3),
      btorph = 0,
      soc = 2,
      cond = 1,
      ddmodel = 1,
      num_cycles = Inf,
      optimmethod = 'simplex'
    ))
    result_ML_Opt <- suppressMessages(DDD::dd_ML(
      brts = brts,
      initparsopt = c(estimate$pred_lambda, estimate$pred_mu, estimate$pred_cap),
      idparsopt = c(1, 2, 3),
      btorph = 0,
      soc = 2,
      cond = 1,
      ddmodel = 1,
      num_cycles = Inf,
      optimmethod = 'simplex'
    ))

    message("Computing from neural network...")
    results_NN[i, ] <- c(result$pred_lambda, result$pred_mu, result$pred_cap)
    message("Computing from ML with typical starting values...")
    results_ML_Typ[i, ] <- c(result_ML_Typ$lambda, result_ML_Typ$mu, result_ML_Typ$K)
    message("Computing from ML with optimal starting values...")
    results_ML_Opt[i, ] <- c(result_ML_Opt$lambda, result_ML_Opt$mu, result_ML_Opt$K)
    message("Bootstrap iteration ", i, " completed")
  }

  out <- list(results_NN = results_NN, results_ML_Typ = results_ML_Typ, results_ML_Opt = results_ML_Opt)
  return(out)
}
