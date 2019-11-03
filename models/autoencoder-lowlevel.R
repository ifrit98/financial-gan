source('constants.R')


Autoencoder <-
  R6::R6Class("Autoencoder",
    
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
        browser()
        # Encoder
        self$encoder_W <- self$add_weight(
          name = 'encoder_W',
          shape = list(input_shape[[2]], self$hidden_dim),
          initializer = self$kernel_initializer,
          regularizer = self$kernel_regularizer,
          trainable = TRUE
        )
        
        self$encoder_b <- self$add_weight(
          name = 'encoder_b',
          shape = list(self$hidden_dim),
          initializer = self$bias_initializer,
          regularizer = self$bias_regularizer,
          trainable = TRUE
        )
        
        # # Decoder
        # self$decoder_W <- self$add_weight(
        #   name = 'decoder_W',
        #   shape = list(self$encoder_W$shape[1], self$encoder_W$shape[0]),
        #   initializer = self$my_initializer,
        #   regularizer = self$kernel_regularizer,
        #   trainable = TRUE
        # )
        self$decoder_W <- tf$transpose(self$encoder_W)
        
        self$decoder_b <- self$add_weight(
          name = 'decoder_b',
          shape = list(input_shape[[length(input_shape)]]),
          initializer = self$bias_initializer,
          regularizer = self$bias_regularizer,
          trainable = TRUE
        )
      },
      
      call = function(x, mask = NULL) {
        encoder_out <-
          tf$nn$tanh(
            tf$nn$bias_add(tf$matmul(x, self$encoder_W), self$encoder_b))
        
        decoder_out <-
          tf$nn$tanh(
            tf$nn$bias_add(
              tf$matmul(encoder_out, self$decoder_W), self$decoder_b))
        
        decoder_out
      },
      
      compute_output_shape = function(input_shape) {
        input_shape
      },
      
      my_initializer = 
        function(shape, dtype = NULL) tf$transpose(self$encoder_W) 
      
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

















# 
# 
# 
# la <- layer_autoencoder()
# la$build(list(784, 784) %>% as.integer())
# la$call(x)
# 
# # units for encoder layers
# dimensions <- c(784, 512, 256, 64)
# 
# input <- x <- layer_input(shape = list(dimensions[1]))
# 
# 
# # Encoder
# encoder <- vector('list', length(dimensions))
# n_inp <- input$get_shape()[1] 
# n_out <- dimensions[1]
# 
# W <- tf$random_uniform(shape = list(784L, 784L), # list(n_inp, n_out)
#                        minval = tf$divide(-1, tf$math$sqrt(784)),
#                        maxval = tf$divide(-1, tf$math$sqrt(784)))
# b <- tf$zeros(list(n_out))
# 
# output <-
#   tf$nn$tanh(tf$nn$bias_add(tf$matmul(input, W), b))
# 
# 
# # Latent
# z <- output
# 
# 
# # Decoder
# W <- tf$transpose(W)
# b <- tf$zeros(list(n_out))
# 
# output <- 
#   tf$nn$tanh(tf$nn$bias_add(tf$matmul(output, W), b))
# 
# y <- output
# 
# cost <- tf$sqrt(tf$reduce_mean(tf$square(y - x)))
# 
# returnval <- list(x = x, z = z, y = y, cost = cost)
