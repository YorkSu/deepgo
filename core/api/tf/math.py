# -*- coding: utf-8 -*-
"""TensorFlow Math API
  =====

  TensorFlow API, containing MATH Operations
"""


import tensorflow as tf


def acos(x, name=None):
  """TensorFlow.math.acos
    
    Computes acos of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.acos(x, name=name)


def acosh(x, name=None):
  """TensorFlow.math.acosh
    
    Computes inverse hyperbolic cosine of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.acosh(x, name=name)


def add(x, y, name=None):
  """TensorFlow.math.add
    
    Returns x + y element-wise.

    Args:
      x: Tensor
      y: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.add(x, y, name=name)


def add_n(inputs, name=None):
  """TensorFlow.math.add_n
    
    Adds all input tensors element-wise.

    Args:
      inputs: list of Tensor
      name: A name for the operation (optional).
  """
  return tf.math.add_n(inputs, name=name)


def angle(x, name=None):
  """TensorFlow.math.angle
    
    Returns the element-wise argument of a complex (or real) tensor.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.angle(x, name=name)


def asin(x, name=None):
  """TensorFlow.math.asin
    
    Computes the trignometric inverse sine of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.asin(x, name=name)


def asinh(x, name=None):
  """TensorFlow.math.asinh
    
    Computes inverse hyperbolic sine of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.asinh(x, name=name)


def atan(x, name=None):
  """TensorFlow.math.atan
    
    Computes the trignometric inverse tangent of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.atan(x, name=name)


def atan2(y, x, name=None):
  """TensorFlow.math.atan2
    
    Computes arctangent of y/x element-wise, respecting signs of the arguments.

    Args:
      y: Tensor
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.atan2(y, x, name=name)


def atanh(x, name=None):
  """TensorFlow.math.atanh
    
    Computes inverse hyperbolic tangent of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.atanh(x, name=name)


def ceil(x, name=None):
  """TensorFlow.math.ceil
    
    Return the ceiling of the input, element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.ceil(x, name=name)


def cosh(x, name=None):
  """TensorFlow.math.cosh
    
    Computes hyperbolic cosine of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.cosh(x, name=name)


def divide(x, y, name=None):
  """TensorFlow.math.divide
    
    Computes Python style division of x by y.

    Args:
      x: Tensor
      y: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.divide(x, y, name=name)


def expm1(x, name=None):
  """TensorFlow.math.expm1
    
    Computes exp(x) - 1 element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.expm1(x, name=name)


def floor(x, name=None):
  """TensorFlow.math.floor
    
    Returns element-wise largest integer not greater than x.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.floor(x, name=name)


def imag(input, name=None):
  """TensorFlow.math.imag
    
    Returns the imaginary part of a complex (or real) tensor.

    Args:
      x: complex Tensor
      name: A name for the operation (optional).
  """
  return tf.math.imag(input, name=name)


def is_finite(x, name=None):
  """TensorFlow.math.is_finite
    
    Returns which elements of x are finite.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.is_finite(x, name=name)


def is_inf(x, name=None):
  """TensorFlow.math.is_inf
    
    Returns which elements of x are Inf.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.is_inf(x, name=name)


def is_nan(x, name=None):
  """TensorFlow.math.is_nan
    
    Returns which elements of x are NaN.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.is_nan(x, name=name)


def logical_and(x, y, name=None):
  """TensorFlow.math.logical_and
    
    Logical AND function.

    Args:
      x: Tensor
      y: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.logical_and(x, y, name=name)


def logical_not(x, name=None):
  """TensorFlow.math.logical_not
    
    Logical NOT function.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.logical_not(x, name=name)


def logical_or(x, y, name=None):
  """TensorFlow.math.logical_or
    
    Logical OR function.

    Args:
      x: Tensor
      y: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.logical_or(x, y, name=name)


def logical_xor(x, y, name=None):
  """TensorFlow.math.logical_xor
    
    Logical XOR function.

    Args:
      x: Tensor
      y: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.logical_xor(x, y, name=name)


def multiply(x, y, name=None):
  """TensorFlow.math.multiply
    
    Returns an element-wise x * y.

    Args:
      x: Tensor
      y: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.multiply(x, y, name=name)


def negative(x, name=None):
  """TensorFlow.math.negative
    
    Computes numerical negative value element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.negative(x, name=name)


def real(input, name=None):
  """TensorFlow.math.real
    
    Returns the real part of a complex (or real) tensor.

    Args:
      x: complex Tensor
      name: A name for the operation (optional).
  """
  return tf.math.real(input, name=name)


def sinh(x, name=None):
  """TensorFlow.math.sinh
    
    Computes hyperbolic sine of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.sinh(x, name=name)


def tan(x, name=None):
  """TensorFlow.math.tan
    
    Computes tan of x element-wise.

    Args:
      x: Tensor
      name: A name for the operation (optional).
  """
  return tf.math.tan(x, name=name)


def top_k(input, k=1, sorted=True, name=None):
  """TensorFlow.math.top_k
    
    Finds values and indices of the k largest entries for the last dimension.

    Args:
      x: Tensor
      k: int Tensor. Number of top elements to look for 
          along the last dimension (along each row for matrices).
      sorted: If true the resulting k elements will be sorted by 
          the values in descending order.
      name: A name for the operation (optional).
  """
  return tf.math.top_k(x, k=k, sorted=sorted, name=name)

