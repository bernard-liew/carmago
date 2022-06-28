library(R.matlab)

### read data -----------------------------------------------------------------
data <- readMat("data/filtered_LiveEditor_all_data_v3.mat")
# labels <- readMat("data/filtered_LiveEditor_all_data_v3_labels.mat")

data_list <- data$outputs
names(data_list) <- dimnames(data$outputs)[[1]]
data_list <- c(data_list, do.call(c, data$inputs))
names(data_list)[6:length(data_list)] <- 
  paste0(rep(dimnames(data$inputs[[1]])[[1]], 3), 
         "_",
         rep(dimnames(data$inputs)[[1]], each = 8))

saveRDS(data_list, "data/filtered_LiveEditor_all_data_v2_list.RDS")

rm(data); gc()
