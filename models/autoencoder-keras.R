Autoencoder_keras <-
  R6::R6Class("Autoencoder_keras",
              
              inherit = KerasLayer,
              
              public = list(
                hidden_dim = NULL,
                encoder_W = NULL,
                encoder_b = NULL,
                decoder_W = NULL,
                decoder_b = NULL,
                
                initialize = function(hidden_dim) {
                  self$hidden_dim <- hidden_dim
                },
                
                build = function(input_shape) {
                },
                
                call = function(x, mask = NULL) {

                },
                
                compute_output_shape = function(input_shape) {
                  input_shape
                } 
                
              )
  )


layer_autoencoder <-
  function(object,
           hidden_dim = 64,
           name = NULL,
           trainable = TRUE) {
    create_layer(Autoencoder,
                 object,
                 list(
                   hidden_dim = as.integer(hidden_dim),
                   name = name,
                   trainable = trainable)
    )
  }


make_autoencoder <- 
  function() {
    input <- layer_input(shape = list(784L))
    
    output <- input %>% 
      layer_autoencoder()
    
    build_and_compile(input, output, metric = 'mse')
  }


source_python("utils/get_keras_dataset.py")

(model <- make_autoencoder())
model$fit(x_train, 
          x_train,
          validation_data = list(x_test, x_test),
          # steps_per_epoch = 1024L,
          epochs = 30L)

