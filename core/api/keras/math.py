# -*- coding: utf-8 -*-
"""Keras Math API
  ======

  Keras API, containing MATH Operations
"""


from tensorflow.keras import backend as K


def abs(x):
  """Keras.backend.abs
    
    Element-wise absolute value.

    Args:
      x: Tensor
  """
  return K.abs(x)


def all(x, axis=None, keepdims=False):
  """Keras.backend.all
    
    Bitwise reduction (logical AND).

    Args:
      x: Tensor
      axis: int, axis along which to perform the reduction.
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
  """Keras.backend.any
    
    Bitwise reduction (logical OR).

    Args:
      x: Tensor
      axis: int, axis along which to perform the reduction.
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.any(x, axis=axis, keepdims=keepdims)


def clip(x, min_value, max_value):
  """Keras.backend.clip
    
    Element-wise value clipping.

    Args:
      x: Tensor
      min_value: Python float, integer, or tensor.
      max_value: Python float, integer, or tensor.
  """
  return K.clip(x, min_value, max_value)


def cos(x):
  """Keras.backend.cos

    Computes cos of x element-wise.
  
    Args:
      x: Tensor
  """
  return K.cos(x)


def cumprod(x, axis=0):
  """Keras.backend.cumprod

    Cumulative product of the values in a tensor, alongside the specified 
    axis.
  
    Args:
      x: Tensor
      axis: An integer, the axis to compute the product.
  """
  return K.cumprod(x, axis=axis)


def cumsum(x, axis=0):
  """Keras.backend.cumsum

    Cumulative sum of the values in a tensor, alongside the specified axis.
  
    Args:
      x: Tensor
      axis: An integer, the axis to compute the sum.
  """
  return K.cumsum(x, axis=axis)


def dot(x, y):
  """Keras.backend.dot

    Multiplies 2 tensors (and/or variables) and returns a tensor.
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.dot(x, y)


def equal(x, y):
  """Keras.backend.equal

    Element-wise equality between two tensors.
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.equal(x, y)


def exp(x):
  """Keras.backend.exp

    Element-wise exponential.
  
    Args:
      x: Tensor
  """
  return K.exp(x)


def greater(x, y):
  """Keras.backend.greater

    Element-wise truth value of (x > y).
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.greater(x, y)


def greater_equal(x, y):
  """Keras.backend.greater_equal

    Element-wise truth value of (x >= y).
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.greater_equal(x, y)


def in_top_k(predictions, targets, k):
  """Keras.backend.in_top_k

    Selects x in test phase, and alt otherwise.
  
    Args:
      predictions: A tensor of shape (batch_size, classes) and type float32.
      targets: A 1D tensor of length batch_size and type int32 or int64.
      k: An int, number of top elements to consider.
  """
  return K.in_top_k(predictions, targets, k)


def less(x, y):
  """Keras.backend.less

    Element-wise truth value of (x < y).
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.less(x, y)


def less_equal(x, y):
  """Keras.backend.less_equal

    Element-wise truth value of (x <= y).
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.less_equal(x, y)


def log(x):
  """Keras.backend.log

    Element-wise log.
  
    Args:
      x: Tensor
  """
  return K.log(x)


def max(x, axis=None, keepdims=False):
  """Keras.backend.max

    Maximum value in a tensor.
  
    Args:
      x: Tensor
      axis: An integer, the axis to find maximum values.
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.max(x, axis=axis, keepdims=keepdims)


def maximum(x, y):
  """Keras.backend.maximum

    Element-wise maximum of two tensors.
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.maximum(x, y)


def mean(x, axis=None, keepdims=False):
  """Keras.backend.mean

    Mean of a tensor, alongside the specified axis.
  
    Args:
      x: Tensor
      axis: A list of integer. Axes to compute the mean.
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.mean(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
  """Keras.backend.min

    Minimum value in a tensor.
  
    Args:
      x: Tensor
      axis: An integer, the axis to find minimum values.
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.min(x, axis=axis, keepdims=keepdims)


def minimum(x, y):
  """Keras.backend.minimum

    Element-wise minimum of two tensors.
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.minimum(x, y)


def not_equal(x, y):
  """Keras.backend.not_equal

    Element-wise inequality between two tensors.
  
    Args:
      x: Tensor
      y: Tensor
  """
  return K.not_equal(x, y)


def pow(x, a):
  """Keras.backend.pow

    Element-wise exponentiation.
  
    Args:
      x: Tensor
      a: Python integer.
  """
  return K.pow(x, a)


def prod(x, axis=None, keepdims=False):
  """Keras.backend.prod

    Multiplies the values in a tensor, alongside the specified axis.
  
    Args:
      x: Tensor
      axis: An integer, the axis to compute the product.
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.prod(x, axis=axis, keepdims=keepdims)


def round(x):
  """Keras.backend.round

    Element-wise rounding to the closest integer.
  
    Args:
      x: Tensor
  """
  return K.round(x)


def sign(x):
  """Keras.backend.sign

    Element-wise sign.
  
    Args:
      x: Tensor
  """
  return K.sign(x)


def sin(x):
  """Keras.backend.sin

    Computes sin of x element-wise.
  
    Args:
      x: Tensor
  """
  return K.sin(x)


def softmax(x, axis=-1):
  """Keras.backend.softmax

    Softmax of a tensor.
  
    Args:
      x: Tensor
      axis: The dimension softmax would be performed on. 
          The default is -1 which indicates the last dimension.
  """
  return K.softmax(x, axis=axis)


def softplus(x):
  """Keras.backend.softplus

    Softplus of a tensor.
  
    Args:
      x: Tensor
  """
  return K.softplus(x)


def softsign(x):
  """Keras.backend.softsign

    Softsign of a tensor.
  
    Args:
      x: Tensor
  """
  return K.softsign(x)


def sqrt(x):
  """Keras.backend.sqrt

    Element-wise square root.
  
    Args:
      x: Tensor
  """
  return K.sqrt(x)


def square(x):
  """Keras.backend.square

    Element-wise square.
  
    Args:
      x: Tensor
  """
  return K.square(x)


def std(x, axis=None, keepdims=False):
  """Keras.backend.std

    Standard deviation of a tensor, alongside the specified axis.
  
    Args:
      x: A tensor or variable. It should have numerical dtypes. 
          Boolean type inputs will be converted to float.
      axis: An integer, the axis to compute the standard deviation. 
          If None (the default), reduces all dimensions. 
          Must be in the range [-rank(x), rank(x)).
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.std(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
  """Keras.backend.sum

    Sum of the values in a tensor, alongside the specified axis.
  
    Args:
      x: A tensor or variable.
      axis: An integer, the axis to sum over.
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.sum(x, axis=axis, keepdims=keepdims)


def tanh(x):
  """Keras.backend.tanh

    Element-wise tanh.
  
    Args:
      x: Tensor
  """
  return K.tanh(x)


def var(x, axis=None, keepdims=False):
  """Keras.backend.var

    Sum of the values in a tensor, alongside the specified axis.
  
    Args:
      x: A tensor or variable.
      axis: An integer, the axis to compute the variance.
      keepdims: A boolean, whether to keep the dimensions or not. 
          If False, the rank of the tensor is reduced by 1. 
          If True, the reduced dimension is retained with length 1.
  """
  return K.var(x, axis=axis, keepdims=keepdims)

