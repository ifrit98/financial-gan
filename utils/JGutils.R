# Is x explicitly a list type
is_list <- function(x) is.vector(x) && !is.atomic(x)

# Is x explicitly a vector with length(vector) > 1
is_vec  <- function(x) is.vector(x) & length(x) != 1L

# Is x an (array, list, vector) with length(x) > 1
is_vec2 <- function(x) is_list(x) | is_vec(x) 


# Convencience fun to one-liner a model with in-outs
build_and_compile <-
  function(input,
           output,
           optimizer = 'adam',
           loss = "mse",
           metric = 'acc') {
    model <- keras::keras_model(input, output) %>%
      keras::compile(optimizer = optimizer,
                     loss = loss,
                     metric = metric)
    model
  }
