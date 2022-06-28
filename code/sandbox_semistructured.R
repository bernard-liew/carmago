### Load libraries ------------------------------------------------------------
library(reticulate)
library(devtools)
library(tidyverse)
library(refund)
library(FDboost)
library(abind)
library(keras)

### Options for running the code
on_server <- FALSE

### Code for semi-structure part ----------------------------------------------
# Although deepregression is on CRAN, a newer version is available 
# on Github. Clone the repo into a folder of your choice and 
# also clone the add-on package funnel. Then load both packages
path_to_repos <- "~/NSL/"
load_all(paste0(path_to_repos, "deepregression"))
load_all(paste0(path_to_repos, "funnel"))
devtools::load_all("~/NSL/deepoptim")

### Load data and split in train / test  --------------------------------------
data_list <- readRDS("data/filtered_LiveEditor_all_data_v2_list.RDS")

# get names
response_names <- names(data_list)[1:5]
predictor_names <- names(data_list)[-1*1:5]

# define data split
set.seed(42)
split_size <- 0.3
test_indices <- sample(1:nrow(data_list[[1]]), 
                       round(split_size * nrow(data_list[[1]])))
train_indices <- setdiff(1:nrow(data_list[[1]]), test_indices)
train <- lapply(data_list, function(x) x[train_indices,])
test <- lapply(data_list, function(x) x[test_indices,])

# train[predictor_names] <- lapply(train[predictor_names], function(x) I(x))
# test[predictor_names] <- lapply(test[predictor_names], function(x) I(x))

# define time variable
train$cycle <- test$cycle <- cycle <- 1:101

# create multivariate time series format for deep models
train_x_mts_format <- abind(train[predictor_names], rev.along=0)
test_x_mts_format <- abind(test[predictor_names], rev.along=0)

### Loop over all response names ----------------------------------------------
# for(resp in response_names){

resp <- response_names[4]
train_y <- train[[resp]]
test_y <- test[[resp]]

### Load inception ------------------------------------------------------------
# Also load the InceptionNet architecture from the py file
inceptionNet <- import_from_path(
  "inceptionnet", path = "code/")$inceptionnet_architecture(
    nb_filters=50L, use_residual=TRUE, use_bottleneck=TRUE, depth=6L, c_out=101L,
    kernel_size=40L, stride=1L, activation='linear', bottleneck_size = 32L
  )
## test architecture
# inp <- keras::layer_input(c(3L,3L))
# outp <- inceptionNet(inp)
# mod <- keras_model(inp, outp)
# mod$summary()

### Stand-alone InceptionNet
inp <- layer_input(dim(train_x_mts_format)[-1])
outp <- inceptionNet(inp)
mod <- keras_model(inp, outp)
mod %>% compile(loss = "mse", 
                optimizer = optimizer_adam(lr = exp(-5.6)))

callbacks <- list(
  callback_early_stopping(patience = 60L, 
                          restore_best_weights = TRUE),
  callback_reduce_lr_on_plateau(monitor = "loss", factor = 0.5, patience = 10L,
                                min_lr = 0.00001)
)

mod %>% fit(x = train_x_mts_format,
            y = train_y,
            batch_size = 64L,
            epochs = 100L,
            verbose = 1,
            view_metrics = FALSE,
            validation_split = 0.3,
            callbacks = callbacks
            )
# best val_loss: 0.0183 for me

prediction_inceptionnet <- mod %>% predict(test_x_mts_format)

saveRDS(prediction_inceptionnet, file=paste0("output/prediction_", 
                                             resp, "_inceptionnet.RDS"))
# compare
par(mfrow=c(1,2))
matplot(t(test_y), type="l")
matplot(t(prediction_inceptionnet), type="l")

rm(mod, prediction_inceptionnet); gc()

### Function-on-function regression
if(on_server){ # will take too much mem on a PC
  
  form <- paste(resp, " ~ 1 + ", paste(
    paste0("ff(", predictor_names,
           ", yind=cycle, xind=cycle)"),
    collapse = " + ")
  )
  
  # prep data for pffr
  train <- lapply(train, function(x) I(x))
  train$cycle <- cycle
  
  # initialize the model
  mod <- pffr(as.formula(form),
              yind = 1:101,
              algorithm = "bam",
              # discrete = TRUE,
              data = train)
  
  prediction_pffr <- mod %>% predict(test)
  
  saveRDS(prediction_pffr, file=paste0("output/prediction_", resp, "_pffr.RDS"))
  
  rm(mod, prediction_pffr); gc()
}

### Functional Boosting
if(on_server){ # will take too much mem on a PC
  form <- paste(resp, " ~ 1 + ", paste(
    paste0("bsignal(", predictor_names,
           ", cycle)"),
    collapse = " + ")
  )
  
  train[predictor_names] <- lapply(train[predictor_names], 
                                   function(x) scale(x, scale=F))
  test[predictor_names] <- lapply(test[predictor_names], 
                                  function(x) scale(x, scale=F))
  
  # initialize the model
  mod <- FDboost(as.formula(form),
                 data = train,
                 timeformula = ~ bbs(cycle, df = 5),
                 control=boost_control(mstop = 1000, nu = 0.1))
  
  set.seed(123)
  appl1 <- applyFolds(mod, folds = mboost::cv(rep(1, length(unique(m$id))), 
                                            B = 5), 
                      grid = 1:1000)
  ## plot(appl1)
  mod[mstop(mod)]
  
  prediction_fdboost <- mod %>% predict(test)
  par(mfrow=c(1,2))
  matplot(t(test_y), type="l")
  matplot(t(prediction_fdboost), type="l")
  
  saveRDS(prediction_fdboost, file=paste0("output/prediction_", 
                                          resp, "_fdboost.RDS"))
  
  rm(mod, prediction_fdboost); gc()

}
### Function-on-function Regression in a neural network
fun_part <- paste0("~ ", paste(predictor_names, 
                               collapse=" + "))

# this first fits a structured model only
# we then copy weights 
mod <- funnel(y = train_y,
              data = train, 
              list_of_formulas = list(as.formula(fun_part), 
                                      ~ 1),
              auto_convert_formulas = TRUE,
              time_variable_outcome = train$cycle,
              name_outcome_time = "cycle", 
              name_feature_time = "cycle",
              optimizer = optimizer_sgd(lr = 0.01),
              monitor_metrics = list("mse"),
              fun_options = fun_controls(
                k_t = 20,
                df_t = 20,
                k_s = 7,
                df_s = 7,
                intercept_k = 20
              )
)

if(!file.exists(paste0("models/weights_structured_",resp,".hdf5"))){
  
  mod %>% fit(epochs = 2500L, 
              batch_size = 10L,
              callbacks = list(# callback_reduce_lr_on_plateau(patience = 40),
                callback_early_stopping(monitor = "val_mse", 
                                        patience = 15,
                                        restore_best_weights = T)
              )
  )
  
  save_model_weights_hdf5(mod$model, filepath=paste0("models/weights_structured_",resp,".hdf5"))
  
}else{
  
  mod$model$load_weights(filepath=paste0("models/weights_structured_",resp,".hdf5"),
                         by_name = FALSE)
  
}

prediction_neural_structured <- mod %>% predict(test)
# pe <- mod %>% get_partial_effect()
# # compare
# par(mfrow=c(1,2))
# matplot(t(test_y), type="l")
# matplot(t(pe[[2]]), type="l", add=TRUE, col="red", lwd=2)
# matplot(t(prediction_neural_structured), type="l")
# matplot(t(pe[[2]]), type="l", add=TRUE, col="red", lwd=2)

# par(mfrow=c(1,1))
# matplot(t(test_y), type="l")
# matplot(t(prediction_neural_structured), type="l", add=T, col="red", lwd=2)

saveRDS(prediction_neural_structured, file=paste0("output/prediction_", 
                                                  resp, "_neural_structured.RDS"))

rm(mod, prediction_neural_structured); gc()


fun_part <- paste0("~ ", paste(predictor_names, 
                               collapse=" + "))

frm_mean <- paste0(form2text(convert_lof_to_loff(fun_part, 
                                                 formula_outcome_time = "cycle",
                                                 formula_feature_time = "cycle",
                                                 type = "matrix",
                                                 controls = fun_controls(
                                                   k_t = 20,
                                                   df_t = 20,
                                                   k_s = 7,
                                                   df_s = 7,
                                                   intercept_k = 20
                                                 ))[[1]]), 
                   " + inceptionNet(train_x_mts_format)")

train$train_x_mts_format <- I(train_x_mts_format)

mod <- funnel(y = train_y,
              data = train, 
              list_of_formulas = list(as.formula(frm_mean), ~ 1),
              list_of_deep_models = list(inceptionNet = inceptionNet),
              auto_convert_formulas = FALSE,
              time_variable_outcome = train$cycle,
              name_outcome_time = "cycle", 
              name_feature_time = "cycle",
              # optimizer = optimizer_sgd(lr = 0.1),
              monitor_metrics = list("mse"),
              fun_options = fun_controls(
                k_t = 20,
                df_t = 20,
                k_s = 7,
                df_s = 7,
                intercept_k = 20
              )
              # model_fun = build_bcdKeras("cyclic") 
)

if(!file.exists(paste0("models/weights_semistructured_",resp,".hdf5"))){
  
  mod %>% fit(epochs = 2500L, 
              batch_size = 30L,
              callbacks = list(
                callback_early_stopping(monitor = "val_mse", 
                                        patience = 5,
                                        restore_best_weights = T)
              )
  )
  
  save_model_weights_hdf5(mod$model, filepath=paste0("models/weights_semistructured_",resp,".hdf5"))
  
}else{
  
  mod$model$load_weights(filepath=paste0("models/weights_semistructured_",resp,".hdf5"),
                         by_name = FALSE)
  
}

test$train_x_mts_format <- I(test_x_mts_format)
prediction_neural_semistructured <- mod %>% predict(test)

# par(mfrow=c(1,2))
# matplot(t(test_y), type="l")
# matplot(t(prediction_neural_semistructured), type="l")


# }