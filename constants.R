reticulate::use_python("/home/home/financial-time-series/bin/python")

library(reticulate)
library(tensorflow)
library(keras)
library(tidyverse)

print(py_config())
print(tf_config())

# options(device = "X11")
# options(device = "RStudioGD")
