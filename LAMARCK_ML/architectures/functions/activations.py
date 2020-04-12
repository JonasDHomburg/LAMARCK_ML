from enum import Enum

class Activations(Enum):
  sigmoid = 1,
  tanh = 2,
  linear = 3,
  relu = 4,
  selu = 5,
  elu = 6,
  exponential = 7,
  hard_sigmoid = 8,
  softmax = 9,
  softplus = 10,
  softsign = 11,
  sign = 12,

  sine = 20,
  cosine = 21,
  absolute = 22,
  inverse  = 23,

  gaussian = 30,

  # TODO: correct values
  flops_weight = {
    sigmoid: 1,
    tanh: 1,
    linear: 0,
    relu: 1,
    selu: 1,
    elu: 1,
    exponential: 1,
    hard_sigmoid: 1,
    softmax: 1,
    softplus: 1,
    softsign: 1,
    sign: 1,

    sine: 2,
    cosine: 2,
    absolute: 1,
    inverse: 1,

    gaussian: 2,
  }

# TODO: add it to function attributes