
source('constants.R')


encoding_dim <- 32L
input_img <- layer_input(shape = list(784))

encoded <- layer_dense(input_img, units = encoding_dim, activation = 'relu')
decoded <- layer_dense(encoded, units = 784, activation = 'sigmoid')

autoencoder <- keras_model(input_img, decoded)

encoder <- keras_model(input_img, encoded)

encoded_input <- layer_input(shape = list(encoding_dim))
decoder_layer <- autoencoder$layers[length(autoencoder$layers)]

deocder <- keras_model(encoded_input, decoder_layers(encoded_input))

autoencoder$compile(
  optimizer = 'adadelta', loss = 'binary_crossentropy')

# Load MNIST
source_python('utils/get_keras_dataset.py')

autoencoder$fit(x_train, x_train,
                epochs = 50L,
                batch_size = 256L,
                shuffle = TRUE,
                validation_data = list(x_test, x_test))
# Doesn't learn shit??
