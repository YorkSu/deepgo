# -*- coding: utf-8 -*-
"""Keras Pooling Layer
  =====

  Keras Layer, containing Pooling Layers
"""


from tensorflow.keras import layers as kl


def AveragePooling1D(
      pool_size=2,
      strides=None,
      padding='valid',
      data_format=None,
      **kwargs):
  """Average pooling for temporal data.

    Arguments:
      pool_size: Integer, size of the average pooling windows.
      strides: Integer, or None. Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, steps, features)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, steps)`.

    Output shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, downsampled_steps, features)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, downsampled_steps)`.
  """
  return kl.AveragePooling1D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      **kwargs)


def AveragePooling2D(
      pool_size=(2, 2),
      strides=None,
      padding='valid',
      data_format=None,
      **kwargs):
  """Average pooling operation for spatial data.

    Arguments:
      pool_size: integer or tuple of 2 integers,
        factors by which to downscale (vertical, horizontal).
        `(2, 2)` will halve the input in both spatial dimension.
        If only one integer is specified, the same window length
        will be used for both dimensions.
      strides: Integer, tuple of 2 integers, or None.
        Strides values.
        If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
  """
  return kl.AveragePooling2D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      **kwargs)


def AveragePooling3D(
      pool_size=(2, 2, 2),
      strides=None,
      padding='valid',
      data_format=None,
      **kwargs):
  """Average pooling operation for 3D data (spatial or spatio-temporal).

    Arguments:
      pool_size: tuple of 3 integers,
        factors by which to downscale (dim1, dim2, dim3).
        `(2, 2, 2)` will halve the size of the 3D input in each dimension.
      strides: tuple of 3 integers, or None. Strides values.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
  """
  return kl.AveragePooling3D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      **kwargs)


def MaxPooling1D(
      pool_size=2,
      strides=None,
      padding='valid',
      data_format=None,
      **kwargs):
  """Max pooling operation for temporal data.

    Arguments:
      pool_size: Integer, size of the max pooling windows.
      strides: Integer, or None. Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, steps, features)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, steps)`.

    Output shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, downsampled_steps, features)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, downsampled_steps)`.
  """
  return kl.MaxPooling1D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      **kwargs)


def MaxPooling2D(
      pool_size=(2, 2),
      strides=None,
      padding='valid',
      data_format=None,
      **kwargs):
  """Max pooling operation for spatial data.

    Arguments:
      pool_size: integer or tuple of 2 integers,
        factors by which to downscale (vertical, horizontal).
        `(2, 2)` will halve the input in both spatial dimension.
        If only one integer is specified, the same window length
        will be used for both dimensions.
      strides: Integer, tuple of 2 integers, or None.
        Strides values.
        If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
  """
  return kl.MaxPooling2D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      **kwargs)


def MaxPooling3D(
      pool_size=(2, 2, 2),
      strides=None,
      padding='valid',
      data_format=None,
      **kwargs):
  """Max pooling operation for 3D data (spatial or spatio-temporal).

    Arguments:
      pool_size: Tuple of 3 integers,
        factors by which to downscale (dim1, dim2, dim3).
        `(2, 2, 2)` will halve the size of the 3D input in each dimension.
      strides: tuple of 3 integers, or None. Strides values.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
  """
  return kl.MaxPooling3D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      **kwargs)


def GlobalAveragePooling1D(
      data_format=None,
      **kwargs):
  """Global average pooling operation for temporal data.

    Arguments:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.

    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(batch_size, steps)` indicating whether
        a given step should be masked (excluded from the average).

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`

    Output shape:
      2D tensor with shape `(batch_size, features)`.
  """
  return kl.GlobalAveragePooling1D(
      data_format=data_format,
      **kwargs)


def GlobalAveragePooling2D(
      data_format=None,
      **kwargs):
  """Global average pooling operation for spatial data.

    Arguments:
        data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      2D tensor with shape `(batch_size, channels)`.
  """
  return kl.GlobalAveragePooling2D(
      data_format=data_format,
      **kwargs)


def GlobalAveragePooling3D(
      data_format=None,
      **kwargs):
  """Global Average pooling operation for 3D data.

    Arguments:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      2D tensor with shape `(batch_size, channels)`.
  """
  return kl.GlobalAveragePooling3D(
      data_format=data_format,
      **kwargs)


def GlobalMaxPooling1D(
      data_format=None,
      **kwargs):
  """Global max pooling operation for temporal data.

    Arguments:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`

    Output shape:
      2D tensor with shape `(batch_size, features)`.
  """
  return kl.GlobalMaxPooling1D(
      data_format=data_format,
      **kwargs)


def GlobalMaxPooling2D(
      data_format=None,
      **kwargs):
  """Global max pooling operation for spatial data.

    Arguments:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      2D tensor with shape `(batch_size, channels)`.
  """
  return kl.GlobalMaxPooling2D(
      data_format=data_format,
      **kwargs)


def GlobalMaxPooling3D(
      data_format=None,
      **kwargs):
  """Global Max pooling operation for 3D data.

    Arguments:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      2D tensor with shape `(batch_size, channels)`.
  """
  return kl.GlobalMaxPooling3D(
      data_format=data_format,
      **kwargs)


# Aliases

AvgPool1D = AveragePooling1D
AvgPool2D = AveragePooling2D
AvgPool3D = AveragePooling3D
MaxPool1D = MaxPooling1D
MaxPool2D = MaxPooling2D
MaxPool3D = MaxPooling3D
GlobalAvgPool1D = GlobalAveragePooling1D
GlobalAvgPool2D = GlobalAveragePooling2D
GlobalAvgPool3D = GlobalAveragePooling3D
GlobalMaxPool1D = GlobalMaxPooling1D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalMaxPool3D = GlobalMaxPooling3D

