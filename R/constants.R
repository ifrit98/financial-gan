# reticulate::use_python("/home/home/tf1/bin/python")
reticulate::use_python("/home/home/financial-time-series/bin/python")

library(tidyverse)
library(magrittr)
library(reticulate)
library(tensorflow)
library(keras)
library(pryr)
library(utils)
library(purrr)
library(dplyr)
library(docopt)
library(tibble)
library(listarrays)
library(reticulate)

source('utils/JGutils.R')

# Sys.getenv() %>% print()
sessionInfo() %>% print()
py_config() %>% print()
tf_config() %>% print()

message("Hey motherfucker, welcome to R.  Yes that's a penis prompt...")
options(prompt = "8==> ")

x <- tf$random$normal(shape = list(16L, 8192L, 6L))
y <- tf$random$normal(shape = list(16L, 8192L))
inp <- layer_input(shape = list(784))
sess <- tf$compat$v1$Session()
r <- sess$run

# options(device = "X11")
# options(device = "RStudioGD")
