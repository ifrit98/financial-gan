

add_unif_noise <- function(x) {
  tf$multiply(x, tf$cast(
    tf$random_uniform(
      shape = tf$shape(x),
      minval = 0,
      maxval = 2
    ),
    tf$float32
  ))
}


add_awgn <- function(x, std = 0.1) {
  tf$add(x, tf$cast(
    tf$random_normal(
      shape = tf$shape(x),
      dtype = tf$float32,
      mean = 0,
      stddev = std
    ),
    tf$float32
  ))
}


units <- c(784, 512, 256, 64)

n_inputs <- inp$get_shape()[1]
n_output <- units[1]


W