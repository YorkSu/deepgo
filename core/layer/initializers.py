# -*- coding: utf-8 -*-
"""Keras Initializer
  =====

  Keras Initializers
"""


from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops


class Initializer(object):
  """Initializer base class: all initializers inherit from this class."""

  def __call__(self, shape, dtype=None, partition_info=None):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided use the initializer
        dtype.
      partition_info: Optional information about the possible partitioning of a
        tensor.
    """
    raise NotImplementedError

  def get_config(self):
    """Returns the configuration of the initializer as a JSON-serializable dict.

    Returns:
      A JSON-serializable Python dict.
    """
    return {}

  @classmethod
  def from_config(cls, config):
    """Instantiates an initializer from a configuration dictionary.

    Example:

    ```python
    initializer = RandomUniform(-1, 1)
    config = initializer.get_config()
    initializer = RandomUniform.from_config(config)
    ```

    Args:
      config: A Python dictionary. It will typically be the output of
        `get_config`.

    Returns:
      An Initializer instance.
    """
    return cls(**config)


def Zeros(dtype=dtypes.float32):
  """Initializer that generates tensors initialized to 0."""
  return init_ops.Zeros(dtype=dtype)


def Ones(dtype=dtypes.float32):
  """Initializer that generates tensors initialized to 1."""
  return init_ops.Ones(dtype=dtype)


def Constant(value=0, dtype=dtypes.float32, verify_shape=False):
  """Initializer that generates tensors with constant values.

  The resulting tensor is populated with values of type `dtype`, as
  specified by arguments `value` following the desired `shape` of the
  new tensor (see examples below).

  The argument `value` can be a constant value, or a list of values of type
  `dtype`. If `value` is a list, then the length of the list must be less
  than or equal to the number of elements implied by the desired shape of the
  tensor. In the case where the total number of elements in `value` is less
  than the number of elements required by the tensor shape, the last element
  in `value` will be used to fill the remaining entries. If the total number of
  elements in `value` is greater than the number of elements required by the
  tensor shape, the initializer will raise a `ValueError`.

  Args:
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy
      array. All elements of the initialized variable will be set to the
      corresponding value in the `value` argument.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer.
    verify_shape: Boolean that enables verification of the shape of `value`. If
      `True`, the initializer will throw an error if the shape of `value` is not
      compatible with the shape of the initialized tensor.

  Raises:
    TypeError: If the input `value` is not one of the expected types.

  Examples:
    The following example can be rewritten using a numpy.ndarray instead
    of the `value` list, even reshaped, as shown in the two commented lines
    below the `value` list initialization.

  >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
  >>> init = tf.compat.v1.constant_initializer(value)
  >>> # fitting shape
  >>> with tf.compat.v1.Session():
  ...   x = tf.compat.v1.get_variable('x', shape=[2, 4], initializer=init)
  ...   x.initializer.run()
  ...   print(x.eval())
  [[0. 1. 2. 3.]
   [4. 5. 6. 7.]]
  >>> # Larger shape
  >>> with tf.compat.v1.Session():
  ...   y = tf.compat.v1.get_variable('y', shape=[3, 4], initializer=init)
  ...   y.initializer.run()
  ...   print(y.eval())
  [[0.  1.  2.  3.]
   [4.  5.  6.  7.]
   [7.  7.  7.  7.]]
  >>> # Smaller shape
  >>> with tf.compat.v1.Session():
  ...   z = tf.compat.v1.get_variable('z', shape=[2, 3], initializer=init)
  Traceback (most recent call last):
  ...
  ValueError: Too many elements provided. Needed at most 6, but received 8
  >>> # Shape verification
  >>> init_verify = tf.compat.v1.constant_initializer(value, verify_shape=True)
  >>> with tf.compat.v1.Session():
  ...  u = tf.compat.v1.get_variable('u', shape=[3, 4],
  ...                                initializer=init_verify)
  Traceback (most recent call last):
  ...
  TypeError: Expected Tensor's shape: (3, 4), got (8,).
  """
  return init_ops.Constant(
      value=value, 
      dtype=dtype, 
      verify_shape=verify_shape)


def RandomUniform(
      minval=-0.05,
      maxval=0.05,
      seed=None,
      dtype=dtypes.float32):
  """Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate. Defaults to -0.05.
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate. Defaults to 0.05.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type.
    
  Returns:
    A RandomUniform instance.
  """
  return init_ops.RandomUniform(
      minval=minval, 
      maxval=maxval, 
      seed=seed,
      dtype=dtype)


def RandomNormal(
      mean=0.0,
      stddev=1.0,
      seed=None,
      dtype=dtypes.float32):
  """Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate. Defaults to 0.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate. Defaults to 0.05.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
      RandomNormal instance.
  """
  return init_ops.RandomNormal(
      mean=mean, 
      stddev=stddev, 
      seed=seed,
      dtype=dtype)


def GlorotUniform(seed=None, dtype=dtypes.float32):
  """The Glorot uniform initializer, also called Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """
  return init_ops.GlorotUniform(
      seed=seed,
      dtype=dtype)


def GlorotNormal(seed=None, dtype=dtypes.float32):
  """The Glorot normal initializer, also called Xavier normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with standard deviation (after truncation) given by
  `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number
  of input units in the weight tensor and `fan_out` is the number of
  output units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """
  return init_ops.GlorotNormal(
      seed=seed,
      dtype=dtype)

