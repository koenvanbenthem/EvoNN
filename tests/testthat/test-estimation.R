test_that("tree validation works", {
  # Test that the non-phylo object returns an error
  expect_error(check_tree_validity(1), "Error: Tree is not in phylo format", fixed = TRUE)
  # Test that the non-binary tree returns an error
  expect_error(check_tree_validity(ape::rtree(10)), "Error: Tree is not binary or ultrametric", fixed = TRUE)
  # Test that the non-ultrametric tree returns an error
  expect_error(check_tree_validity(ape::rtree(10)), "Error: Tree is not binary or ultrametric", fixed = TRUE)
  # Test that the tree with less than 10 nodes returns an error
  expect_error(check_tree_validity(ape::rphylo(5,0.4,0.1)), "Error: Tree size is not within the range [6, 1000]", fixed = TRUE)
  # Test that the tree with more than 2000 nodes returns an error
  expect_error(check_tree_validity(ape::rphylo(1001,0.6,0.1)), "Error: Tree size is not within the range [6, 1000]", fixed = TRUE)
  # Test that a proper tree returns a success message
  expect_equal(check_tree_validity(ape::rphylo(50,0.4,0.1)), "SIG_SUCCESS")
})

test_that("rescale works", {
  # generate a tree with age 20
  tree <- ape::rlineage(0.3, 0.1,20)
  # test that the scale is computed correctly
  rescaled_tree <- rescale_crown_age(tree, 10)
  rescale_tree_age <- treestats::crown_age(rescaled_tree)
  expect_equal(rescale_tree_age, 10)
})

test_that("compute scale works", {
  # generate a tree with age 20
  tree <- ape::rphylo(20,0.3, 0.1)
  tree_age <- treestats::crown_age(tree)
  # test that the scale is computed correctly
  expect_equal(compute_scale(tree), tree_age/10, tolerance = 1e-6)
})

test_that("tree conversion works", {
  # generate a tree with age 10
  tree <- NULL
  tree_size <- 0
  while(tree_size < 10 || tree_size > 2000) {
    tree <- ape::rlineage(0.4, 0.1,10)
    tree <- ape::drop.fossil(tree)
    tree_size <- 2 * tree$Nnode + 1
  }
  # roughly test that the tree to node map works
  tree_nd <- tree_to_connectivity(tree)
  expect_equal(dim(tree_nd), c(2 * tree$Nnode, 2))
  expect_equal(length(tree_nd), 4 * tree$Nnode)
  expect_equal(tree_nd[2 * tree$Nnode, 2], tree$Nnode)
  # roughly test that the adjacency matrix works
  tree_el <- tree_to_adj_mat(tree)
  expect_equal(dim(tree_el), c(2 * tree$Nnode + 1, 3))
  expect_equal(sum(tree_el[1:(tree$Nnode+1),2:3]), sum(matrix(c(rep(0, tree$Nnode+1), rep(0, tree$Nnode+1)), ncol = 2)))
  expect_true(tree_el[tree$Nnode+2, 1]==0)
  expect_true(all(tree_el[(tree$Nnode+3):(2*tree$Nnode+1), 1:3]!=0))
  # roughly test that the summary statistics are computed correctly
  tree_stats <- tree_to_stats(tree)
  expect_true(is.null(dim(tree_stats)))
  expect_equal(length(tree_stats), 54)
  # roughly test that the branching times are computed correctly
  tree_brts <- tree_to_brts(tree)
  expect_true(length(tree_brts)==tree$Nnode)
})

test_that("parameter estimation works", {
  tree <- NULL
  tree_size <- 0
  while(tree_size < 10 || tree_size > 2000) {
    tree <- ape::rlineage(0.4, 0.1,10)
    tree <- ape::drop.fossil(tree)
    tree_size <- 2 * tree$Nnode + 1
  }
  # test that the estimation works
  estimates <- nn_estimate(tree)
  result_names <- c("pred_lambda", "pred_mu", "pred_cap", "scale", "num_nodes")
  expect_equal(names(estimates), result_names)
})