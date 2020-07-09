# -*- coding: utf-8 -*-
"""Keras Activation Layer
  =====

  Keras Layer, containing Activations
"""


__all__ = [
    'ELU',
    'LeakyReLU',
    'PReLU',
    'ReLU',
    'Softmax',
    'ThresholdedReLU',]


from tensorflow.keras import layers as kl


def ELU(
      alpha=1.0,
      **kwargs):
  """Exponential Linear Unit.

    It follows:
    `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
    `f(x) = x for x >= 0`.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Arguments:
      alpha: Scale for the negative factor.
  """
  return kl.ELU(
      alpha=alpha,
      **kwargs)


def LeakyReLU(
      alpha=0.3,
      **kwargs):
  """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Arguments:
      alpha: Float >= 0. Negative slope coefficient.
  """
  return kl.LeakyReLU(
      alpha=alpha,
      **kwargs)


def PReLU(
      alpha_initializer='zeros',
      alpha_regularizer=None,
      alpha_constraint=None,
      shared_axes=None,
      **kwargs):
  """Parametric Rectified Linear Unit.

    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alpha` is a learned array with the same shape as x.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Arguments:
      alpha_initializer: Initializer function for the weights.
      alpha_regularizer: Regularizer for the weights.
      alpha_constraint: Constraint for the weights.
      shared_axes: The axes along which to share learnable
        parameters for the activation function.
        For example, if the incoming feature maps
        are from a 2D convolution
        with output shape `(batch, height, width, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1, 2]`.
  """
  return kl.PReLU(
      alpha_initializer=alpha_initializer,
      alpha_regularizer=alpha_regularizer,
      alpha_constraint=alpha_constraint,
      shared_axes=shared_axes,
      **kwargs)


def ReLU(
      max_value=None,
      negative_slope=0,
      threshold=0,
      **kwargs):
  """Rectified Linear Unit activation function.

    With default values, it returns element-wise `max(x, 0)`.

    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = negative_slope * (x - threshold)` otherwise.
    
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Arguments:
      max_value: Float >= 0. Maximum activation value.
      negative_slope: Float >= 0. Negative slope coefficient.
      threshold: Float. Threshold value for thresholded activation.
  """
  return kl.ReLU(
      max_value=max_value,
      negative_slope=negative_slope,
      threshold=threshold,
      **kwargs)


def Softmax(
      axis=-1,
      **kwargs):
  """Softmax activation function.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Arguments:
      axis: Integer, axis along which the softmax normalization is applied.
  """
  return kl.Softmax(
      axis=axis,
      **kwargs)


def ThresholdedReLU(
      theta=1.0,
      **kwargs):
  """Thresholded Rectified Linear Unit.

    It follows:
    `f(x) = x for x > theta`,
    `f(x) = 0 otherwise`.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Arguments:
      theta: Float >= 0. Threshold location of activation.
  """
  return kl.ThresholdedReLU(
      theta=theta,
      **kwargs)

