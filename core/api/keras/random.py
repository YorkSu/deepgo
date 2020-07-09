# -*- coding: utf-8 -*-
"""Keras Random API
  ======

  Keras API, containing Random methods
"""


from tensorflow.keras import backend as K


def random_binomial(shape, p=0.0, dtype=None, seed=None):
  """Keras.backend.random_binomial

    Returns a tensor with random binomial distribution of values.
  
    Args:
      shape: A tuple of integers, the shape of tensor to create.
      p: A float, 0. <= p <= 1, probability of binomial distribution.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.
  """
  return K.random_binomial(shape, p=p, dtype=dtype, seed=seed)


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  """Keras.backend.random_normal

    Returns a tensor with normal distribution of values.
  
    Args:
      shape: A tuple of integers, the shape of tensor to create.
      mean: A float, the mean value of the normal distribution to draw 
          samples. Default to 0.0.
      stddev: A float, the standard deviation of the normal distribution 
          to draw samples. Default to 1.0.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.
  """
  return K.random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, 
      seed=seed)


def random_normal_variable(shape, mean, scale, dtype=None, name=None,
    seed=None):
  """Keras.backend.random_normal_variable

    Instantiates a variable with values drawn from a normal distribution.
  
    Args:
      shape: Tuple of integers, shape of returned Keras variable.
      mean: Float, mean of the normal distribution.
      scale: Float, standard deviation of the normal distribution.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed. 
  """
  return K.random_normal_variable(shape, mean=mean, scale=scale, 
      dtype=dtype, name=name, seed=seed)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
  """Keras.backend.random_uniform

    Returns a tensor with uniform distribution of values.
  
    Args:
      shape: A tuple of integers, the shape of tensor to create.
      minval: float, lower boundary of the uniform distribution.
      maxval: float, upper boundary of the uniform distribution.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.
  """
  return K.random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype,
      seed=seed)


def random_normal_variable(shape, low, high, dtype=None, name=None,
    seed=None):
  """Keras.backend.random_normal_variable

    Instantiates a variable with values drawn from a uniform distribution.
  
    Args:
      shape: Tuple of integers, shape of returned Keras variable.
      low: Float, lower boundary of the output interval.
      high: Float, upper boundary of the output interval.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed. 
  """
  return K.random_normal_variable(shape, low=low, high=high, dtype=dtype, 
      name=name, seed=seed)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  """Keras.backend.truncated_normal

    Returns a tensor with truncated random normal distribution of values.
  
    Args:
      shape: A tuple of integers, the shape of tensor to create.
      mean: Mean of the values.
      stddev: Standard deviation of the values.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.
  """
  return K.truncated_normal(shape, mean=mean, stddev=stddev, dtype=dtype, 
      seed=seed)

