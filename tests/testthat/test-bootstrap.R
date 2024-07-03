test_that("bootstrap core works", {
  test_estimates <- list(pred_lambda = structure(0.47203266620636, dim = 1L), pred_mu = structure(0.281138569116592, dim = 1L),
                         pred_cap = structure(463, dim = 1L), scale = 1, num_nodes = structure(37, dim = 1L))
  # test that the bootstrap core works
  results <- bootstrap_core(test_estimates, "DDD", 10, 30)
  expect_true(inherits(results, "phylo"))
})

test_that("bootstrapping function works", {
  test_estimates <- list(pred_lambda = structure(0.47203266620636, dim = 1L), pred_mu = structure(0.281138569116592, dim = 1L),
                         pred_cap = structure(463, dim = 1L), scale = 1, num_nodes = structure(37, dim = 1L))
  # test that the bootstrapping works
  results <- nn_bootstrap_uncertainty(test_estimates, "DDD", 10)
  expect_equal(dim(results), c(10, 3))
  expect_false(all(is.na(results)))
})
