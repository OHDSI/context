extractPlpConcepts <- function(plpData, analysisIds = NULL, rootConceptId = 441840) {
  ref <- plpData$covariateData$covariateRef
  if (is.null(ref) || !"conceptId" %in% colnames(ref)) stop("covariateRef requires conceptId")
  if (!is.null(analysisIds)) {
    if (!"analysisId" %in% colnames(ref)) stop("analysisId is required for filtering")
    if (is.data.frame(ref)) {
      ref <- ref[ref$analysisId %in% analysisIds, , drop = FALSE]
    } else {
      ref <- dplyr::filter(ref, .data$analysisId %in% .env$analysisIds)
    }
  }
  if (!is.data.frame(ref)) ref <- dplyr::collect(dplyr::distinct(ref, .data$conceptId))
  values <- ref$conceptId
  ids <- suppressWarnings(as.numeric(as.character(values[!is.na(values)])))
  if (any(!is.finite(ids) | ids != floor(ids) | ids < 0)) stop("Invalid mapped concept ID")
  ids <- ids[ids > 0]
  if (!length(ids)) stop("PLP input contains no mapped concepts")
  if (!is.null(rootConceptId) &&
      (length(rootConceptId) != 1L || !is.numeric(rootConceptId) || !is.finite(rootConceptId) ||
       rootConceptId <= 0 || rootConceptId != floor(rootConceptId))) stop("Invalid root concept ID")
  sort(unique(c(ids, rootConceptId)))
}

writePlpConcepts <- function(plpData, file, analysisIds = NULL, rootConceptId = 441840) {
  ids <- extractPlpConcepts(plpData, analysisIds, rootConceptId)
  utils::write.csv(data.frame(conceptId = ids), file, row.names = FALSE, quote = FALSE)
  invisible(ids)
}
