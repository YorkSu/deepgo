# -*- coding: utf-8 -*-
"""Keras Tensor API
  ======

  Keras API, containing methods for handling Tensor
"""


from tensorflow.keras import backend as K


def arange(start, stop=None, step=1, dtype='int32'):
  """Keras.backend.arange
    
    Creates a 1D tensor containing a sequence of integers.
  """
  return K.arange(start, stop=stop, step=step, dtype=dtype)


def argmax(x, axis=-1):
  """Keras.backend.argmax
    
    Returns the index of the maximum value along an axis.

    Args:
      x: Tensor
      axis: int, axis along which to perform the reduction.
  """
  return K.argmax(x, axis=axis)


def argmin(x, axis=-1):
  """Keras.backend.argmin
    
    Returns the index of the minimum value along an axis.

    Args:
      x: Tensor
      axis: int, axis along which to perform the reduction.
  """
  return K.argmin(x, axis=axis)


def batch_dot(x, y, axes=None):
  """Keras.backend.batch_dot
    
    Batchwise dot product.

    Args:
      x: Tensor, ndim >= 2
      y: Tensor, ndim >= 2
      axes: Tuple or list of integers with target dimensions, or single 
          integer. The sizes of x.shape[axes[0]] and y.shape[axes[1]] 
          should be equal.
  """
  return K.batch_dot(x, y, axes=axes)


def batch_flatten(x):
  """Keras.backend.batch_flatten
    
    Turn a nD tensor into a 2D tensor with same 0th dimension.

    Args:
      x: Tensor
  """
  return K.batch_flatten(x)


def batch_get_value(tensors):
  """Keras.backend.batch_get_value
    
    Returns the value of more than one tensor variable.

    Args:
      tensors: list of Tensor
    
    Returns:
      A list of Numpy arrays.
  """
  return K.batch_get_value(tensors)


def batch_normalization(x, mean, var, beta, gamma, axis=-1,
    epsilon=0.001):
  """Keras.backend.batch_normalization
    
    Applies batch normalization on x given mean, var, beta and gamma.

    Args:
      x: Tensor
      mean: Mean of batch.
      var: Variance of batch.
      beta: Tensor with which to center the input.
      gamma: Tensor by which to scale the input.
      axis: Integer, the axis that should be normalized. (typically the 
          features axis).
      epsilon: Fuzz factor.

    Returns:
      output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta
  """
  return K.batch_normalization(x, mean, var, beta, gamma,
      axis = axis, epsilon = epsilon)
      

def batch_set_value(tuples):
  """Keras.backend.batch_set_value
    
    Sets the values of many tensor variables at once.

    Args:
      tuples: a list of tuples(tensor, value)
          value should be a Numpy array
  """
  return K.batch_set_value(tuples)


def bias_add(x, bias, data_format=None):
  """Keras.backend.bias_add
    
    Adds a bias vector to a tensor.

    Args:
      x: Tensor
      bias: Bias tensor to add.
      data_format: str, "channels_last" or "channels_first".
  """
  return K.bias_add(x, bias, data_format=data_format)


def concatenate(tensors, axis=-1):
  """Keras.backend.concatenate
    
    Concatenates a list of tensors alongside the specified axis.

    Args:
      tensors: list of tensors to concatenate.
      axis: concatenation axis.
  """
  return K.concatenate(tensors, axis=axis)


def constant(value, dtype=None, shape=None, name=None):
  """Keras.backend.constant
    
    Creates a constant tensor.

    Args:
      value: A constant value (or list)
      dtype: The type of the elements of the resulting tensor.
      shape: Optional dimensions of resulting tensor.
      name: Optional name for the tensor.
  """
  return K.constant(value, dtype=dtype, shape=shape, name=name)


def expand_dims(x, axis=-1):
  """Keras.backend.expand_dims

    Adds a 1-sized dimension at index "axis".
  
    Args:
      x: Tensor
      axis: Position where to add a new axis.
  """
  return K.expand_dims(x, axis=axis)


def eye(size, dtype=None, name=None):
  """Keras.backend.eye

    Instantiate an identity matrix and returns it.
  
    Args:
      size: Integer, number of rows/columns.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.
  """
  return K.eye(size, dtype=dtype, name=name)


def flatten(x):
  """Keras.backend.flatten

    Flatten a tensor.
  
    Args:
      x: Tensor
  """
  return K.flatten(x)


def foldl(fn, elems, initializer=None, name=None):
  """Keras.backend.foldl

    Reduce elems using fn to combine them from left to right.

    Args:
      fn: Callable that will be called upon each element in elems 
          and an accumulator, for instance lambda acc, x: acc + x
      elems: Tensor
      initializer: The first value used (elems[0] in case of None)
      name: A string name for the foldl node in the graph
  """
  return K.foldl(fn, elems, initializer=initializer, name=name)


def foldr(fn, elems, initializer=None, name=None):
  """Keras.backend.foldr

    Reduce elems using fn to combine them from right to left.

    Args:
      fn: Callable that will be called upon each element in elems 
          and an accumulator, for instance lambda acc, x: acc + x
      elems: Tensor
      initializer: The first value used (elems[0] in case of None)
      name: A string name for the foldr node in the graph
  """
  return K.foldr(fn, elems, initializer=initializer, name=name)


def gather(reference, indices):
  """Keras.backend.gather

    Retrieves the elements of indices in the tensor reference.
  
    Args:
      reference: Tensor
      indices: An integer tensor of indices.
  """
  return K.gather(reference, indices)


def map_fn(fn, elems, name=None, dtype=None):
  """Keras.backend.map_fn

    Map the function fn over the elements elems and return the outputs.
  
    Args:
      fn: Callable that will be called upon each element in elems
      elems: Tensor
      name: A string name for the map node in the graph
      dtype: Output data type.
  """
  return K.map_fn(fn, elems, name=name, dtype=dtype)


def ones(shape, dtype=None, name=None):
  """Keras.backend.ones

    Instantiates an all-ones variable and returns it.
  
    Args:
      shape: Tuple of integers, shape of returned Keras variable.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.
  """
  return K.ones(shape, dtype=dtype, name=name)


def ones_like(x, dtype=None, name=None):
  """Keras.backend.ones_like

    Instantiates an all-ones variable of the same shape as another tensor.
  
    Args:
      x: Tensor
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.
  """
  return K.ones_like(x, dtype=dtype, name=name)


def one_hot(indices, num_classes):
  """Keras.backend.one_hot

    Computes the one-hot representation of an integer tensor.
  
    Args:
      indices: nD integer tensor of shape (batch_size, dim1, dim2, ... 
          dim(n-1))
      num_classes: Integer, number of classes to consider.
  """
  return K.one_hot(indices, num_classes)


def permute_dimensions(x, pattern):
  """Keras.backend.permute_dimensions

    Permutes axes in a tensor.
  
    Args:
      x: Tensor
      pattern: A tuple of dimension indices, e.g. (0, 2, 1).
  """
  return K.permute_dimensions(x, pattern)


def repeat(x, n):
  """Keras.backend.repeat

    Repeats a 2D tensor.
  
    Args:
      x: Tensor
      n: Python integer, number of times to repeat.

    Returns:
      if x has shape (samples, dim) and n is 2, 
      the output will have shape (samples, 2, dim).
  """
  return K.repeat(x, n)


def repeat_elements(x, rep, axis):
  """Keras.backend.repeat_elements

    Repeats a 2D tensor.
  
    Args:
      x: Tensor
      rep: Python integer, number of times to repeat.
      axis: Axis along which to repeat.

    Returns:
      If x has shape (s1, s2, s3) and axis is 1, 
      the output will have shape (s1, s2 * rep, s3).
  """
  return K.repeat_elements(x, rep, axis)


def reshape(x, shape):
  """Keras.backend.reshape

    Reshapes a tensor to the specified shape.
  
    Args:
      x: Tensor
      shape: Target shape tuple.
  """
  return K.reshape(x, shape)


def resize_images(x, height_factor, width_factor, data_format,
    interpolation='nearest'):
  """Keras.backend.resize_images

    Resizes the images contained in a 4D tensor.
  
    Args:
      x: Tensor
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of "channels_first", "channels_last".
      interpolation: A string, one of nearest or bilinear.
  """
  return K.resize_images(x, height_factor, width_factor, data_format, 
      interpolation=interpolation)


def resize_volumes(x, depth_factor, height_factor, width_factor,
    data_format):
  """Keras.backend.resize_volumes

    Resizes the volume contained in a 5D tensor.
  
    Args:
      x: Tensor
      depth_factor: Positive integer.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of "channels_first", "channels_last".
  """
  return K.resize_volumes(x, depth_factor, height_factor, width_factor,
      data_format)


def reverse(x, axes):
  """Keras.backend.reverse

    Reverse a tensor along the specified axes.
  
    Args:
      x: Tensor
      axes: Integer or iterable of integers. Axes to reverse.
  """
  return K.reverse(x, axes)


def shape(x):
  """Keras.backend.shape

    Returns the symbolic shape of a tensor or variable.
  
    Args:
      x: Tensor
  """
  return K.shape(x)


def squeeze(x, axis):
  """Keras.backend.squeeze

    Removes a 1-dimension from the tensor at index "axis".
  
    Args:
      x: Tensor
      axis: Axis to drop.
  """
  return K.squeeze(x, axis)


def stack(x, axis=0):
  """Keras.backend.stack

    Stacks a list of rank R tensors into a rank R+1 tensor.
  
    Args:
      x: List of tensors.
      axis: Axis along which to perform stacking.
  """
  return K.stack(x, axis=axis)


def tile(x, n):
  """Keras.backend.tile

    Creates a tensor by tiling x by n.
  
    Args:
      x: Tensor
      n: A list of integer. The length must be the same as the number of 
          dimensions in x.
  """
  return K.tile(x, n)


def transpose(x):
  """Keras.backend.transpose

    Transposes a tensor and returns it.
  
    Args:
      x: Tensor
  """
  return K.transpose(x)


def zeros(shape, dtype=None, name=None):
  """Keras.backend.zeros

    Instantiates an all-zeros variable and returns it.
  
    Args:
      shape: Tuple of integers, shape of returned Keras variable.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.
  """
  return K.zeros(shape, dtype=dtype, name=name)


def zeros_like(x, dtype=None, name=None):
  """Keras.backend.zeros_like

    Instantiates an all-zeros variable of the same shape as another tensor.
  
    Args:
      x: Tensor
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.
  """
  return K.zeros_like(x, dtype=dtype, name=name)

