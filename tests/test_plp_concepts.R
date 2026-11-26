source("R/extractPlpConcepts.R")
ref <- data.frame(conceptId = factor(c("20", "20", NA, "30", "0")),
                  covariateId = 1001:1005, analysisId = c(1, 2, 1, 2, 1))
plp <- list(covariateData = list(covariateRef = ref))
stopifnot(identical(extractPlpConcepts(plp, 1), c(20, 441840)))
stopifnot(identical(extractPlpConcepts(plp, 2, NULL), c(20, 30)))
file <- tempfile()
writePlpConcepts(plp, file)
stopifnot(identical(as.numeric(read.csv(file)$conceptId), c(20, 30, 441840)))
unlink(file)
if (requireNamespace("dbplyr", quietly = TRUE) && requireNamespace("RSQLite", quietly = TRUE)) {
  db <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
  dplyr::copy_to(db, ref, "covariateRef")
  lazy <- list(covariateData = list(covariateRef = dplyr::tbl(db, "covariateRef")))
  stopifnot(identical(extractPlpConcepts(lazy, 2, NULL), c(20, 30)))
  DBI::dbDisconnect(db)
}
stopifnot(inherits(try(extractPlpConcepts(plp, 999), silent = TRUE), "try-error"))
cat("PLP concept extraction passed\n")
