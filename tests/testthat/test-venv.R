test_that("venv is set properly", {
  pyconfig <- reticulate::py_config()
  correct_venv <- file.path(fs::path_home(), ".virtualenvs", "EvoNN")
  correct_numpy <- file.path(correct_venv, "lib", "site-packages", "numpy")
  expect_equal(pyconfig$virtualenv, correct_venv)
  #expect_true(fs::dir_exists(correct_numpy))
})
