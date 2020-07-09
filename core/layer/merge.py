# -*- coding: utf-8 -*-
"""Keras Merge Layer
  =====

  Keras Layer, containing Merge Layers
"""


from tensorflow.keras import layers as kl
from tensorflow.python.keras.layers.merge import _Merge as _M


class _Merge(_M):
  """Generic merge layer for elementwise merge functions.

    Used to implement `Sum`, `Average`, etc.

    Arguments:
        **kwargs: standard layer keyword arguments.
  """


def Add(**kwargs):
  """Layer that adds a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).

    Examples:

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # equivalent to `added = keras.layers.add([x1, x2])`
        added = keras.layers.Add()([x1, x2])
        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
  """
  return kl.Add(**kwargs)


def Average(**kwargs):
  """Layer that averages a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
  """
  return kl.Average(**kwargs)


def Concatenate(axis=-1, **kwargs):
  """Layer that concatenates a list of inputs.

    It takes as input a list of tensors,
    all of the same shape except for the concatenation axis,
    and returns a single tensor, the concatenation of all inputs.

    Arguments:
        axis: Axis along which to concatenate.
        **kwargs: standard layer keyword arguments.
  """
  return kl.Concatenate(axis=axis, **kwargs)


def Dot(axes, normalize=False, **kwargs):
  """Layer that computes a dot product between samples in two tensors.

    E.g. if applied to a list of two tensors `a` and `b` of shape
    `(batch_size, n)`, the output will be a tensor of shape `(batch_size, 1)`
    where each entry `i` will be the dot product between
    `a[i]` and `b[i]`.

    Arguments:
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
  """
  return kl.Dot(axes, normalize=normalize, **kwargs)


def Maximum(**kwargs):
  """Layer that computes the maximum (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
  """
  return kl.Maximum(**kwargs)


def Minimum(**kwargs):
  """Layer that computes the minimum (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
  """
  return kl.Minimum(**kwargs)


def Multiply(**kwargs):
  """Layer that multiplies (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
  """
  return kl.Multiply(**kwargs)


def Subtract(**kwargs):
  """Layer that subtracts two inputs.

    It takes as input a list of tensors of size 2,
    both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]),
    also of the same shape.

    Examples:

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # Equivalent to subtracted = keras.layers.subtract([x1, x2])
        subtracted = keras.layers.Subtract()([x1, x2])

        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
  """
  return kl.Subtract(**kwargs)

