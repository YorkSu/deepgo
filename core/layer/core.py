# -*- coding: utf-8 -*-
"""Keras Core Layer
  =====

  Keras Layer, containing Core Layers
"""


__all__ = [
    'Activation',
    'ActivityRegularization',
    'Dense',
    'Dropout',
    'Flatten',
    'Lambda',
    'Masking',
    'Permute',
    'RepeatVector',
    'Reshape',
    'SpatialDropout1D',
    'SpatialDropout2D',
    'SpatialDropout3D',]


from tensorflow.keras import layers as kl


def Activation(
      activation,
      **kwargs):
  """Applies an activation function to an output.

    Arguments:
      activation: Activation function, such as `tf.nn.relu`, or string name of
        built-in activation function, such as "relu".

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
  """
  return kl.Activation(
      activation,
      **kwargs)


def ActivityRegularization(
      l1=0.,
      l2=0.,
      **kwargs):
  """Layer that applies an update to the cost function based input activity.

    Arguments:
      l1: L1 regularization factor (positive float).
      l2: L2 regularization factor (positive float).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
  """
  return kl.ActivityRegularization(
      l1=l1,
      l2=l2,
      **kwargs)


def Dense(
      units,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs):
  """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    Example:

    ```python
    # as first layer in a sequential model:
    model = Sequential()
    model.add(Dense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)

    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(Dense(32))
    ```

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
  """
  return kl.Dense(
      units,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      **kwargs)


def Dropout(
      rate,
      noise_shape=None,
      seed=None,
      **kwargs):
  """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Arguments:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the
        binary dropout mask that will be multiplied with the input.
        For instance, if your inputs have shape
        `(batch_size, timesteps, features)` and
        you want the dropout mask to be the same for all timesteps,
        you can use `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
  """
  return kl.Dropout(
      rate,
      noise_shape=noise_shape,
      seed=seed,
      **kwargs)


def Flatten(
      data_format=None,
      **kwargs):
  """Flattens the input. Does not affect the batch size.

    If inputs are shaped `(batch,)` without a channel dimension, then flattening
    adds an extra channel dimension and output shapes are `(batch, 1)`.

    Arguments:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Example:

    ```python
    model = Sequential()
    model.add(Convolution2D(64, 3, 3,
                            border_mode='same',
                            input_shape=(3, 32, 32)))
    # now: model.output_shape == (None, 64, 32, 32)

    model.add(Flatten())
    # now: model.output_shape == (None, 65536)
    ```
  """
  return kl.Flatten(
      data_format=data_format,
      **kwargs)


def Lambda(
      function,
      output_shape=None,
      mask=None,
      arguments=None,
      **kwargs):
  """Wraps arbitrary expressions as a `Layer` object.

    The `Lambda` layer exists so that arbitrary TensorFlow functions
    can be used when constructing `Sequential` and Functional API
    models. `Lambda` layers are best suited for simple operations or
    quick experimentation. For more advanced use cases, subclassing
    `keras.layers.Layer` is preferred. One reason for this is that
    when saving a Model, `Lambda` layers are saved by serializing the
    Python bytecode, whereas subclassed Layers are saved via overriding
    their `get_config` method and are thus more portable. Models that rely
    on subclassed Layers are also often easier to visualize and reason
    about.

    Examples:

    ```python
    # add a x -> x^2 layer
    model.add(Lambda(lambda x: x ** 2))
    ```
    ```python
    # add a layer that returns the concatenation
    # of the positive part of the input and
    # the opposite of the negative part

    def antirectifier(x):
        x -= K.mean(x, axis=1, keepdims=True)
        x = K.l2_normalize(x, axis=1)
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=1)

    model.add(Lambda(antirectifier))
    ```

    Variables can be created within a `Lambda` layer. Like with
    other layers, these variables will be created only once and reused
    if the `Lambda` layer is called on new inputs. If creating more
    than one variable in a given `Lambda` instance, be sure to use
    a different name for each variable. Note that calling sublayers
    from within a `Lambda` is not supported.

    Example of variable creation:

    ```python
    def linear_transform(x):
      v1 = tf.Variable(1., name='multiplier')
      v2 = tf.Variable(0., name='bias')
      return x*v1 + v2

    linear_layer = Lambda(linear_transform)
    model.add(linear_layer)
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(linear_layer)  # Reuses existing Variables
    ```

    Note that creating two instances of `Lambda` using the same function
    will *not* share Variables between the two instances. Each instance of
    `Lambda` will create and manage its own weights.

    Arguments:
      function: The function to be evaluated. Takes input tensor as first
        argument.
      output_shape: Expected output shape from function. This argument can be
        inferred if not explicitly provided. Can be a tuple or function. If a
        tuple, it only specifies the first dimension onward;
        sample dimension is assumed either the same as the input: `output_shape =
          (input_shape[0], ) + output_shape` or, the input is `None` and
        the sample dimension is also `None`: `output_shape = (None, ) +
          output_shape` If a function, it specifies the entire shape as a function
          of the
        input shape: `output_shape = f(input_shape)`
      mask: Either None (indicating no masking) or a callable with the same
        signature as the `compute_mask` layer method, or a tensor that will be
        returned as output mask regardless what the input is.
      arguments: Optional dictionary of keyword arguments to be passed to the
        function.
    Input shape: Arbitrary. Use the keyword argument input_shape (tuple of
      integers, does not include the samples axis) when using this layer as the
      first layer in a model.
    Output shape: Specified by `output_shape` argument
  """
  return kl.Lambda(
      mfunction,
      output_shape=output_shape,
      mask=mask,
      arguments=arguments,
      **kwargs)


def Masking(
      mask_value=0.,
      **kwargs):
  """Masks a sequence by using a mask value to skip timesteps.

    For each timestep in the input tensor (dimension #1 in the tensor),
    if all values in the input tensor at that timestep
    are equal to `mask_value`, then the timestep will be masked (skipped)
    in all downstream layers (as long as they support masking).

    If any downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.

    Example:

    Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
    to be fed to an LSTM layer.
    You want to mask timestep #3 and #5 because you lack data for
    these timesteps. You can:

    - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
    - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

    ```python
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
    model.add(LSTM(32))
    ```
  """
  return kl.Masking(
      mask_value=mask_value,
      **kwargs)


def Permute(
      dims,
      **kwargs):
  """Permutes the dimensions of the input according to a given pattern.

    Useful for e.g. connecting RNNs and convnets together.

    Example:

    ```python
    model = Sequential()
    model.add(Permute((2, 1), input_shape=(10, 64)))
    # now: model.output_shape == (None, 64, 10)
    # note: `None` is the batch dimension
    ```

    Arguments:
      dims: Tuple of integers. Permutation pattern, does not include the
        samples dimension. Indexing starts at 1.
        For instance, `(2, 1)` permutes the first and second dimensions
        of the input.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same as the input shape, but with the dimensions re-ordered according
      to the specified pattern.
  """
  return kl.Permute(
      dims,
      **kwargs)


def RepeatVector(
      n,
      **kwargs):
  """Repeats the input n times.

    Example:

    ```python
    model = Sequential()
    model.add(Dense(32, input_dim=32))
    # now: model.output_shape == (None, 32)
    # note: `None` is the batch dimension

    model.add(RepeatVector(3))
    # now: model.output_shape == (None, 3, 32)
    ```

    Arguments:
      n: Integer, repetition factor.

    Input shape:
      2D tensor of shape `(num_samples, features)`.

    Output shape:
      3D tensor of shape `(num_samples, n, features)`.
  """
  return kl.RepeatVector(
      n,
      **kwargs)


def Reshape(
      target_shape,
      **kwargs):
  """Reshapes an output to a certain shape.

    Arguments:
      target_shape: Target shape. Tuple of integers,
        does not include the samples dimension (batch size).

    Input shape:
      Arbitrary, although all dimensions in the input shaped must be fixed.
      Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      `(batch_size,) + target_shape`

    Example:

    ```python
    # as first layer in a Sequential model
    model = Sequential()
    model.add(Reshape((3, 4), input_shape=(12,)))
    # now: model.output_shape == (None, 3, 4)
    # note: `None` is the batch dimension

    # as intermediate layer in a Sequential model
    model.add(Reshape((6, 2)))
    # now: model.output_shape == (None, 6, 2)

    # also supports shape inference using `-1` as dimension
    model.add(Reshape((-1, 2, 2)))
    # now: model.output_shape == (None, None, 2, 2)
    ```
  """
  return kl.Reshape(
      target_shape,
      **kwargs)


def SpatialDropout1D(
      rate,
      **kwargs):
  """Spatial 1D version of Dropout.

    This version performs the same function as Dropout, however it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout1D will help promote independence
    between feature maps and should be used instead.

    Arguments:
      rate: Float between 0 and 1. Fraction of the input units to drop.

    Call arguments:
      inputs: A 3D tensor.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      3D tensor with shape:
      `(samples, timesteps, channels)`

    Output shape:
      Same as input.

    References:
      - [Efficient Object Localization Using Convolutional
        Networks](https://arxiv.org/abs/1411.4280)
  """
  return kl.SpatialDropout1D(
      rate,
      **kwargs)


def SpatialDropout2D(
      rate,
      data_format=None,
      **kwargs):
  """Spatial 2D version of Dropout.

    This version performs the same function as Dropout, however it drops
    entire 2D feature maps instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout2D will help promote independence
    between feature maps and should be used instead.

    Arguments:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      data_format: 'channels_first' or 'channels_last'.
        In 'channels_first' mode, the channels dimension
        (the depth) is at index 1,
        in 'channels_last' mode is it at index 3.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Call arguments:
      inputs: A 4D tensor.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
      Same as input.

    References:
      - [Efficient Object Localization Using Convolutional
        Networks](https://arxiv.org/abs/1411.4280)
  """
  return kl.SpatialDropout2D(
      rate,
      data_format=data_format,
      **kwargs)


def SpatialDropout3D(
      rate,
      data_format=None,
      **kwargs):
  """Spatial 3D version of Dropout.

    This version performs the same function as Dropout, however it drops
    entire 3D feature maps instead of individual elements. If adjacent voxels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout3D will help promote independence
    between feature maps and should be used instead.

    Arguments:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      data_format: 'channels_first' or 'channels_last'.
          In 'channels_first' mode, the channels dimension (the depth)
          is at index 1, in 'channels_last' mode is it at index 4.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

    Call arguments:
      inputs: A 5D tensor.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      5D tensor with shape:
      `(samples, channels, dim1, dim2, dim3)` if data_format='channels_first'
      or 5D tensor with shape:
      `(samples, dim1, dim2, dim3, channels)` if data_format='channels_last'.

    Output shape:
      Same as input.

    References:
      - [Efficient Object Localization Using Convolutional
        Networks](https://arxiv.org/abs/1411.4280)
  """
  return kl.SpatialDropout3D(
      rate,
      data_format=data_format,
      **kwargs)

