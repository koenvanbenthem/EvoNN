#' Function to plot the bootstrap results of the EvoNN estimates
#'
#' Plots the posterior distributions of the speciation rate, extinction rate, and carrying capacity (DDD) of a
#' phylogenetic tree estimated using EvoNN, with the original estimates
#' @param bootstrap The bootstrap results of the EvoNN estimates
#' @param result The EvoNN estimate of the speciation rate, extinction rate, and carrying capacity (DDD)
#' @param scenario The diversification scenario: \cr \code{"BD"} : Birth-death
#' diversification \cr \code{"DDD"} : Diversity-dependent diversification
#' @return \item{ plots }{ A list of plots }
#' @author Tianjian Qin
#' @export nn_plot_bootstrap
nn_plot_bootstrap <- function(bootstrap, result, scenario = "DDD") {
  if (scenario != "DDD") {
    stop("Only DDD is supported for boostrapping")
  }
  if (nrow(bootstrap) == 0) {
    stop("No bootstrap results to plot")
  }
  if (ncol(bootstrap) != 3) {
    stop("Invalid number of columns in the bootstrap results")
  }
  if (length(result) != 5) {
    stop("Invalid number of elements in the result")
  }

  plots <- par(mfrow=c(1,3))
  plots <- plot(density(bootstrap$lambda, na.rm = T), main = "Speciation rate λ")
  plots <- abline(v=result$pred_lambda, col="red",lty="dashed")
  plots <- text(result$pred_lambda,.5*par('usr')[4],labels=round(result$pred_lambda,3), col="blue")
  plots <- plot(density(bootstrap$mu, na.rm = T), main = "Extinction rate μ")
  plots <- abline(v=result$pred_mu, col="red",lty="dashed")
  plots <- text(result$pred_mu,.5*par('usr')[4],labels=round(result$pred_mu,3), col="blue")
  plots <- plot(density(bootstrap$cap, na.rm = T), main = "Carrying capacity K")
  plots <- abline(v=result$pred_cap, col="red",lty="dashed")
  plots <- text(result$pred_cap,.5*par('usr')[4],labels=round(result$pred_cap,3), col="blue")
}