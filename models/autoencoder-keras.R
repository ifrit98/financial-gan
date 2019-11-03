source('R/constants.R')
source_python("utils/get_keras_dataset.py")


EncoderDecoder <- 
  R6::R6Class("EncoderDecoder",
    
    inherit = KerasLayer,
    
    public = list(
      hidden_dim = NULL,
      mode = NULL,
      original_dim = NULL,
      hidden_layer = NULL,
      output_layer = NULL,
      
      initialize = function(mode, hidden_dim, original_dim) {
        self$mode <- mode 
        self$hidden_dim <- hidden_dim
        self$original_dim <- original_dim # Infer from input_shape?
      },
      
      build = function(input_shape) {
        
        if (rlang::is_empty(self$original_dim))
          self$original_dim <- input_shape[[length(input_shape)]]
        
        self$hidden_layer <- layer_dense(
          units = self$hidden_dim,
          activation = 'relu',
          kernel_initializer = 'he_uniform'
        )
        
        self$output_layer <- layer_dense(
          units = 
            if (self$mode == 'decoder')
              self$original_dim else self$hidden_dim,
          activation = 'relu',
          kernel_initializer = 'he_uniform'
        )
      },
      
      call = function(x, mask = NULL) {
        
        activation <- self$hidden_layer(x)
        
        output <- self$output_layer(activation)
        
        output
      },
      
      compute_output_shape = function(input_shape) {
        input_shape
      }
      
    )
  )


layer_encoder_decoder <-
  function(object,
           mode = 'encoder',
           hidden_dim = 64,
           original_dim = NULL,
           name = NULL,
           trainable = TRUE) {
    
    create_layer(EncoderDecoder,
                 object,
                 list(
                   mode = tolower(mode),
                   hidden_dim = as.integer(hidden_dim),
                   original_dim = as.integer(original_dim),
                   name = name,
                   trainable = trainable
                 ))
  }




# Model version of above
autoencoder_model <- 
  function(num_layers,
           hidden_dim,
           original_dim, # need to know at model instantiation
           name = NULL) {

    is_lst <- is_vec2(hidden_dim)
        
    hidden_dim <- 
      if(!is_lst) as.integer(hidden_dim) else hidden_dim
    
    num_layers <- as.integer(num_layers)
    original_dim <- as.integer(original_dim)
    
    stopifnot(is_lst & length(hidden_dim) == num_layers)
    
    keras_model_custom(name = name, function(self) {

      map2_fn <- function(mode, hidden) 
        layer_encoder_decoder(mode = mode, hidden_dim = hidden)
      
      self$encoder_layers <- purrr::map2(
        .x = rep('encoder', num_layers),
        .y = if (is_lst)
          hidden_dim
        else
          rep(hidden_dim, num_layers),
        .f = map2_fn
      )
      
      self$decoder_layers <- purrr::map2(
        .x = rep('decoder', num_layers + 1L),
        .y = if (is_lst)
          c(rev(hidden_dim), original_dim)
        else
          rep(hidden_dim, num_layers + 1L),
        .f = map2_fn
      )
      
      # Call
      function(x, mask = NULL) {

        output <- x
        
        # TODO: Shape issue??
        #'  Error in py_call_impl(callable, dots$args, dots$keywords) : 
        #InvalidArgumentError: 2 root error(s) found.
        #(0) Invalid argument: Incompatible shapes: [32,784] vs. [32,64]
        #[[{{node loss_2/output_1_loss/SquaredDifference}}]]
        #[[metrics_4/acc/Identity/_181]]
        #(1) Invalid argument: Incompatible shapes: [32,784] vs. [32,64]
        #[[{{node loss_2/output_1_loss/SquaredDifference}}]]
        #0 successful operations.
        #0 derived errors ignored. 
        
        for (i in 1L:length(self$encoder_layers) - 1L) { 
          output <- self$encoder_layers[i](output) 
        }
        
        for (j in 1L:length(self$decoder_layers) - 1L) { 
          output <- self$decoder_layers[i](output) 
        }
        
        output
      }
      
    })
  }

model <- 
  autoencoder_model(
    num_layers = 3,
    hidden_dim = list(512L, 256L, 64L),
    original_dim = 784
  )

model %>% compile(
  loss = 'mse',
  optimizer = 'adagrad',
  metrics = c('accuracy')
)



model %>% fit(x_train, 
              x_train, 
              epochs = 10, 
              batch_size = 128, 
              validation_data = list(x_test, x_test))






############################################################################
############################################################################



# 
# 
# Autoencoder_keras <-
#   R6::R6Class(
#     "Autoencoder_keras",
#     
#     inherit = KerasLayer,
#     
#     public = list(
#       hidden_dim = NULL,
#       encoder = NULL,
#       decoder = NULL,
#       original_dim = NULL,
#       
#       initialize = function(hidden_dim) {
#         self$hidden_dim <- hidden_dim
#       },
#       
#       build = function(input_shape) {
#         self$original_dim <- input_shape[[length(input_shape)]]
#         
#         self$encoder <-
#           layer_encoder_decoder(mode = 'encoder',
#                                 hidden_dim = self$hidden_dim)
#         
#         self$decoder <-
#           layer_encoder_decoder(
#             mode = 'decoder',
#             hidden_dim = self$hidden_dim,
#             original_dim = self$original_dim
#           )
#       },
#       
#       call = function(x, mask = NULL) {
#         encoder_out <- self$encoder(x)
#         decoder_out <- self$decoder(encoder_out)
#         
#         decoder_out
#       },
#       
#       compute_output_shape = function(input_shape) {
#         input_shape
#       }
#       
#     )
#   )


# layer_autoencoder <-
#   function(object,
#            hidden_dim,
#            name = NULL,
#            trainable = TRUE) {
#     create_layer(Autoencoder_keras,
#                  object,
#                  list(
#                    hidden_dim = as.integer(hidden_dim),
#                    name = name,
#                    trainable = trainable)
#     )
#   }

# la <- layer_autoencoder()
# la$build(y$shape)
# la$call(y)


# make_autoencoder <- 
#   function(input_dim = list(8192L)) {
#     input <- layer_input(shape = input_dim)
#     
#     output <- input %>% 
#       layer_autoencoder()
#     
#     build_and_compile(input, output, metric = 'mse')
#   }
# 
# make_autoencoder()
