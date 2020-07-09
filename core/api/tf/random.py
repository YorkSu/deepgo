# -*- coding: utf-8 -*-
"""TensorFlow Random API
  =====

  TensorFlow API, containing Random methods
"""


import tensorflow as tf


def normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None,
    name=None):
  """TensorFlow.random.normal

    Outputs random values from a normal distribution.
  
    Args:
      shape: A 1-D integer Tensor or Python array. 
          The shape of the output tensor.
      mean: A Tensor or Python value of type dtype, broadcastable with 
          stddev. The mean of the normal distribution.
      stddev: A Tensor or Python value of type dtype, broadcastable with 
          mean. The standard deviation of the normal distribution.
      dtype: The type of the output.
      seed: A Python integer. Used to create a random seed for the 
          distribution. See tf.random.set_seed for behavior.
      name: A name for the operation (optional).
  """
  return tf.random.normal(shape, mean=mean, stddev=stddev, dtype=dtype, 
      seed=seed, name=name)


def poisson(shape, lam, dtype=tf.dtypes.float32, seed=None, name=None):
  """TensorFlow.random.poisson

    Draws shape samples from each of the given Poisson distribution(s).
  
    Args:
      shape: A 1-D integer Tensor or Python array. 
          The shape of the output tensor.
      lam: A Tensor or Python value or N-D array of type dtype. lam 
          provides the rate parameter(s) describing the poisson 
          distribution(s) to sample.
      dtype: The type of the output.
      seed: A Python integer. Used to create a random seed for the 
          distribution. See tf.random.set_seed for behavior.
      name: A name for the operation (optional).
  """
  return tf.random.poisson(shape, lam=lam, dtype=dtype, seed=seed,
      name=name)


def set_seed(seed):
  """TensorFlow.random.set_seed
    
    Sets the global random seed.

    Args:
      seed: integer.
  """
  return tf.random.set_seed(seed)


def shuffle(value, seed=None, name=None):
  """TensorFlow.random.shuffle
    
    Randomly shuffles a tensor along its first dimension.

    Args:
      value: A Tensor to be shuffled.
      seed: A Python integer. Used to create a random seed for the 
          distribution. See tf.random.set_seed for behavior.
      name: A name for the operation (optional).
  """
  return tf.random.shuffle(value, seed=seed, name=name)


def uniform(shape, minval=0, maxval=None, dtype=tf.dtypes.float32,
    seed=None, name=None):
  """TensorFlow.random.uniform

    Outputs random values from a uniform distribution.
  
    Args:
      shape: A 1-D integer Tensor or Python array. 
          The shape of the output tensor.
      minval: A Tensor or Python value of type dtype, broadcastable 
          with maxval. The lower bound on the range of random values 
          to generate (inclusive). Defaults to 0.
      maxval: A Tensor or Python value of type dtype, broadcastable 
          with minval. The upper bound on the range of random values 
          to generate (exclusive). Defaults to 1 if dtype is floating 
          point.
      dtype: The type of the output.
      seed: A Python integer. Used to create a random seed for the 
          distribution. See tf.random.set_seed for behavior.
      name: A name for the operation (optional).
  """
  return tf.random.uniform(shape, minval=minval, maxval=maxval, 
      dtype=dtype, seed=seed, name=name)

