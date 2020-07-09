# -*- coding: utf-8 -*-
"""Keras Special Layer
  =====

  Keras Layer, containing Special Layers
"""


from tensorflow.keras import layers as kl


def AlphaDropout(
      rate,
      noise_shape=None,
      seed=None,
      **kwargs):
  """Applies Alpha Dropout to the input.

    Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
    to their original values, in order to ensure the self-normalizing property
    even after this dropout.
    Alpha Dropout fits well to Scaled Exponential Linear Units
    by randomly setting activations to the negative saturation value.

    Arguments:
      rate: float, drop probability (as with `Dropout`).
        The multiplicative noise will have
        standard deviation `sqrt(rate / (1 - rate))`.
      seed: A Python integer to use as random seed.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
  """
  return kl.AlphaDropout(
      rate,
      noise_shape=noise_shape,
      seed=seed,
      **kwargs)


def AdditiveAttention(
      use_scale=True,
      **kwargs):
  """Additive attention layer, a.k.a. Bahdanau-style attention.

    Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor of
    shape `[batch_size, Tv, dim]` and `key` tensor of shape
    `[batch_size, Tv, dim]`. The calculation follows the steps:

    1. Reshape `query` and `value` into shapes `[batch_size, Tq, 1, dim]`
      and `[batch_size, 1, Tv, dim]` respectively.
    2. Calculate scores with shape `[batch_size, Tq, Tv]` as a non-linear
      sum: `scores = tf.reduce_sum(tf.tanh(query + value), axis=-1)`
    3. Use scores to calculate a distribution with shape
      `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
    4. Use `distribution` to create a linear combination of `value` with
      shape `batch_size, Tq, dim]`:
      `return tf.matmul(distribution, value)`.

    Args:
      use_scale: If `True`, will create a variable to scale the attention scores.
      causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
        that position `i` cannot attend to positions `j > i`. This prevents the
        flow of information from the future towards the past.

    Call Arguments:

      inputs: List of the following tensors:
        * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
        * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
        * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
          given, will use `value` for both `key` and `value`, which is the
          most common case.
      mask: List of the following tensors:
        * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
          If given, the output will be zero at the positions where
          `mask==False`.
        * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
          If given, will apply the mask such that values at positions where
          `mask==False` do not contribute to the result.

    Output shape:

      Attention outputs of shape `[batch_size, Tq, dim]`.

    The meaning of `query`, `value` and `key` depend on the application. In the
    case of text similarity, for example, `query` is the sequence embeddings of
    the first piece of text and `value` is the sequence embeddings of the second
    piece of text. `key` is usually the same tensor as `value`.

    Here is a code example for using `AdditiveAttention` in a CNN+Attention
    network:

    ```python
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    value_input = tf.keras.Input(shape=(None,), dtype='int32')

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)
    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(query_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.AdditiveAttention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    # Add DNN layers, and create Model.
    # ...
    ```
  """
  return kl.AdditiveAttention(
      use_scale=use_scale,
      **kwargs)


def Attention(
      use_scale=False,
      **kwargs):
  """Dot-product attention layer, a.k.a. Luong-style attention.

    Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor of
    shape `[batch_size, Tv, dim]` and `key` tensor of shape
    `[batch_size, Tv, dim]`. The calculation follows the steps:

    1. Calculate scores with shape `[batch_size, Tq, Tv]` as a `query`-`key` dot
      product: `scores = tf.matmul(query, key, transpose_b=True)`.
    2. Use scores to calculate a distribution with shape
      `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
    3. Use `distribution` to create a linear combination of `value` with
      shape `batch_size, Tq, dim]`:
      `return tf.matmul(distribution, value)`.

    Args:
      use_scale: If `True`, will create a scalar variable to scale the attention
        scores.
      causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
        that position `i` cannot attend to positions `j > i`. This prevents the
        flow of information from the future towards the past.

    Call Arguments:

      inputs: List of the following tensors:
        * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
        * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
        * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
          given, will use `value` for both `key` and `value`, which is the
          most common case.
      mask: List of the following tensors:
        * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
          If given, the output will be zero at the positions where
          `mask==False`.
        * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
          If given, will apply the mask such that values at positions where
          `mask==False` do not contribute to the result.

    Output shape:

      Attention outputs of shape `[batch_size, Tq, dim]`.

    The meaning of `query`, `value` and `key` depend on the application. In the
    case of text similarity, for example, `query` is the sequence embeddings of
    the first piece of text and `value` is the sequence embeddings of the second
    piece of text. `key` is usually the same tensor as `value`.

    Here is a code example for using `Attention` in a CNN+Attention network:

    ```python
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    value_input = tf.keras.Input(shape=(None,), dtype='int32')

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)
    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(query_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    # Add DNN layers, and create Model.
    # ...
    ```
  """
  return kl.Attention(
      use_scale=use_scale,
      **kwargs)


def BatchNormalization(
      axis=-1,
      momentum=0.99,
      epsilon=1e-3,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      renorm=False,
      renorm_clipping=None,
      renorm_momentum=0.99,
      fused=None,
      trainable=True,
      virtual_batch_size=None,
      adjustment=None,
      **kwargs):
  """Base class of Batch normalization layer (Ioffe and Szegedy, 2014).

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Arguments:
      axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
      momentum: Momentum for the moving average.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
      scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      moving_mean_initializer: Initializer for the moving mean.
      moving_variance_initializer: Initializer for the moving variance.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      beta_constraint: Optional constraint for the beta weight.
      gamma_constraint: Optional constraint for the gamma weight.
      renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during
        training. The inference is the same for either value of this parameter.
      renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
      renorm_momentum: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `momentum` is still applied
        to get the means and variances for inference.
      fused: if `None` or `True`, use a faster, fused implementation if possible.
        If `False`, use the system recommended implementation.
      trainable: Boolean, if `True` the variables will be marked as trainable.
      virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
        which means batch normalization is performed across the whole batch. When
        `virtual_batch_size` is not `None`, instead perform "Ghost Batch
        Normalization", which creates virtual sub-batches which are each
        normalized separately (with shared gamma, beta, and moving statistics).
        Must divide the actual batch size during execution.
      adjustment: A function taking the `Tensor` containing the (dynamic) shape of
        the input tensor and returning a pair (scale, bias) to apply to the
        normalized values (before gamma and beta), only during training. For
        example, if axis==-1,
          `adjustment = lambda shape: (
            tf.random.uniform(shape[-1:], 0.93, 1.07),
            tf.random.uniform(shape[-1:], -0.1, 0.1))`
        will scale the normalized value by up to 7% up or down, then shift the
        result by up to 0.1 (with independent scaling and bias for each feature
        but shared across all examples), and finally apply gamma and/or beta. If
        `None`, no adjustment is applied. Cannot be specified if
        virtual_batch_size is specified.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode.
        - `training=True`: The layer will normalize its inputs using the
          mean and variance of the current batch of inputs.
        - `training=False`: The layer will normalize its inputs using the
          mean and variance of its moving statistics, learned during training.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.

    References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """
  return kl.BatchNormalization(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      **kwargs)


def BatchNormalizationV1(
      axis=-1,
      momentum=0.99,
      epsilon=1e-3,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      renorm=False,
      renorm_clipping=None,
      renorm_momentum=0.99,
      fused=None,
      trainable=True,
      virtual_batch_size=None,
      adjustment=None,
      **kwargs):
  """Base class of Batch normalization layer (Ioffe and Szegedy, 2014).

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Arguments:
      axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
      momentum: Momentum for the moving average.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
      scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      moving_mean_initializer: Initializer for the moving mean.
      moving_variance_initializer: Initializer for the moving variance.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      beta_constraint: Optional constraint for the beta weight.
      gamma_constraint: Optional constraint for the gamma weight.
      renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during
        training. The inference is the same for either value of this parameter.
      renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
      renorm_momentum: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `momentum` is still applied
        to get the means and variances for inference.
      fused: if `True`, use a faster, fused implementation, or raise a ValueError
        if the fused implementation cannot be used. If `None`, use the faster
        implementation if possible. If False, do not used the fused
        implementation.
      trainable: Boolean, if `True` the variables will be marked as trainable.
      virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
        which means batch normalization is performed across the whole batch. When
        `virtual_batch_size` is not `None`, instead perform "Ghost Batch
        Normalization", which creates virtual sub-batches which are each
        normalized separately (with shared gamma, beta, and moving statistics).
        Must divide the actual batch size during execution.
      adjustment: A function taking the `Tensor` containing the (dynamic) shape of
        the input tensor and returning a pair (scale, bias) to apply to the
        normalized values (before gamma and beta), only during training. For
        example, if axis==-1,
          `adjustment = lambda shape: (
            tf.random.uniform(shape[-1:], 0.93, 1.07),
            tf.random.uniform(shape[-1:], -0.1, 0.1))`
        will scale the normalized value by up to 7% up or down, then shift the
        result by up to 0.1 (with independent scaling and bias for each feature
        but shared across all examples), and finally apply gamma and/or beta. If
        `None`, no adjustment is applied. Cannot be specified if
        virtual_batch_size is specified.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode.
        - `training=True`: The layer will normalize its inputs using the
          mean and variance of the current batch of inputs.
        - `training=False`: The layer will normalize its inputs using the
          mean and variance of its moving statistics, learned during training.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.

    References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

    **About setting `layer.trainable = False` on a `BatchNormalization layer:**

    The meaning of setting `layer.trainable = False` is to freeze the layer,
    i.e. its internal state will not change during training:
    its trainable weights will not be updated
    during `fit()` or `train_on_batch()`, and its state updates will not be run.

    Usually, this does not necessarily mean that the layer is run in inference
    mode (which is normally controlled by the `training` argument that can
    be passed when calling a layer). "Frozen state" and "inference mode"
    are two separate concepts.

    However, in the case of the `BatchNormalization` layer, **setting
    `trainable = False` on the layer means that the layer will be
    subsequently run in inference mode** (meaning that it will use
    the moving mean and the moving variance to normalize the current batch,
    rather than using the mean and variance of the current batch).

    This behavior has been introduced in TensorFlow 2.0, in order
    to enable `layer.trainable = False` to produce the most commonly
    expected behavior in the convnet fine-tuning use case.

    Note that:
      - This behavior only occurs as of TensorFlow 2.0. In 1.*,
        setting `layer.trainable = False` would freeze the layer but would
        not switch it to inference mode.
      - Setting `trainable` on an model containing other layers will
        recursively set the `trainable` value of all inner layers.
      - If the value of the `trainable`
        attribute is changed after calling `compile()` on a model,
        the new value doesn't take effect for this model
        until `compile()` is called again.
  """
  return kl.BatchNormalizationV1(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      **kwargs)


def Bidirectional(
      layer,
      merge_mode='concat',
      weights=None,
      backward_layer=None,
      **kwargs):
  """Bidirectional wrapper for RNNs.

    Arguments:
      layer: `Recurrent` instance.
      merge_mode: Mode by which outputs of the
        forward and backward RNNs will be combined.
        One of {'sum', 'mul', 'concat', 'ave', None}.
        If None, the outputs will not be combined,
        they will be returned as a list.
      backward_layer: Optional `Recurrent` instance to be used to handle
        backwards input processing. If `backward_layer` is not provided,
        the layer instance passed as the `layer` argument will be used to
        generate the backward layer automatically.
        Note that the provided `backward_layer` layer should have properties
        matching those of the `layer` argument, in particular it should have the
        same values for `stateful`, `return_states`, `return_sequence`, etc.
        In addition, `backward_layer` and `layer` should have
        different `go_backwards` argument values.
        A `ValueError` will be raised if these requirements are not met.

    Call arguments:
      The call arguments for this layer are the same as those of the wrapped RNN
        layer.

    Raises:
      ValueError:
        1. If `layer` or `backward_layer` is not a `Layer` instance.
        2. In case of invalid `merge_mode` argument.
        3. If `backward_layer` has mismatched properties compared to `layer`.

    Examples:

    ```python
    model = Sequential()
    model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
    model.add(Bidirectional(LSTM(10)))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # With custom backward layer
    model = Sequential()
    forward_layer = LSTM(10, return_sequences=True)
    backard_layer = LSTM(10, activation='relu', return_sequences=True,
                          go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
                            input_shape=(5, 10)))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```
  """
  return kl.Bidirectional(
      layer,
      merge_mode=merge_mode,
      weights=weights,
      backward_layer=backward_layer,
      **kwargs)


def deserialize(config, custom_objects=None):
  """Instantiates a layer from a config dictionary.

    Arguments:
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    Returns:
        Layer instance (may be Model, Sequential, Network, Layer...)
  """
  return kl.deserialize(config, custom_objects=custom_objects)


def DeviceWrapper(
      *args,
      **kwargs):
  """Operator that ensures an RNNCell runs on a particular device."""
  return kl.DeviceWrapper(
      *args,
      **kwargs)


def DropoutWrapper(
      *args,
      **kwargs):
  """Operator adding dropout to inputs and outputs of the given cell."""
  return kl.DropoutWrapper(
      *args,
      **kwargs)


def Embedding(
      input_dim,
      output_dim,
      embeddings_initializer='uniform',
      embeddings_regularizer=None,
      activity_regularizer=None,
      embeddings_constraint=None,
      mask_zero=False,
      input_length=None,
      **kwargs):
  """Turns positive integers (indexes) into dense vectors of fixed size.

    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

    This layer can only be used as the first layer in a model.

    Example:

    ```python
    model = Sequential()
    model.add(Embedding(1000, 64, input_length=10))
    # the model will take as input an integer matrix of size (batch,
    # input_length).
    # the largest integer (i.e. word index) in the input should be no larger
    # than 999 (vocabulary size).
    # now model.output_shape == (None, 10, 64), where None is the batch
    # dimension.

    input_array = np.random.randint(1000, size=(32, 10))

    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    assert output_array.shape == (32, 10, 64)
    ```

    Arguments:
      input_dim: int > 0. Size of the vocabulary,
        i.e. maximum integer index + 1.
      output_dim: int >= 0. Dimension of the dense embedding.
      embeddings_initializer: Initializer for the `embeddings` matrix.
      embeddings_regularizer: Regularizer function applied to
        the `embeddings` matrix.
      embeddings_constraint: Constraint function applied to
        the `embeddings` matrix.
      mask_zero: Whether or not the input value 0 is a special "padding"
        value that should be masked out.
        This is useful when using recurrent layers
        which may take variable length input.
        If this is `True` then all subsequent layers
        in the model need to support masking or an exception will be raised.
        If mask_zero is set to True, as a consequence, index 0 cannot be
        used in the vocabulary (input_dim should equal size of
        vocabulary + 1).
      input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).

    Input shape:
      2D tensor with shape: `(batch_size, input_length)`.

    Output shape:
      3D tensor with shape: `(batch_size, input_length, output_dim)`.
  """
  return kl.Embedding(
      input_dim,
      output_dim,
      embeddings_initializer=embeddings_initializer,
      embeddings_regularizer=embeddings_regularizer,
      activity_regularizer=activity_regularizer,
      embeddings_constraint=embeddings_constraint,
      mask_zero=mask_zero,
      input_length=input_length,
      **kwargs)


def GaussianDropout(
      rate,
      **kwargs):
  """Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    Arguments:
      rate: Float, drop probability (as with `Dropout`).
        The multiplicative noise will have
        standard deviation `sqrt(rate / (1 - rate))`.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
  """
  return kl.GaussianDropout(
      rate,
      **kwargs)


def GaussianNoise(
      stddev,
      **kwargs):
  """Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    Arguments:
      stddev: Float, standard deviation of the noise distribution.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
  """
  return kl.GaussianNoise(
      stddev,
      **kwargs)


def LayerNormalization(
      axis=-1,
      epsilon=1e-3,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      trainable=True,
      **kwargs):
  """Layer normalization layer (Ba et al., 2016).

    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.

    Arguments:
      axis: Integer or List/Tuple. The axis that should be normalized
        (typically the features axis).
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor.
          If False, `beta` is ignored.
      scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      beta_constraint: Optional constraint for the beta weight.
      gamma_constraint: Optional constraint for the gamma weight.
      trainable: Boolean, if `True` the variables will be marked as trainable.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.

    References:
      - [Layer Normalization](https://arxiv.org/abs/1607.06450)
  """
  return kl.LayerNormalization(
      axis=axis,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      trainable=trainable,
      **kwargs)


def LocallyConnected1D(
      filters,
      kernel_size,
      strides=1,
      padding='valid',
      data_format=None,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      implementation=1,
      **kwargs):
  """Locally-connected layer for 1D inputs.

    The `LocallyConnected1D` layer works similarly to
    the `Conv1D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each different patch
    of the input.

    Example:
    ```python
        # apply a unshared weight convolution 1d of length 3 to a sequence with
        # 10 timesteps, with 64 output filters
        model = Sequential()
        model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
        # now model.output_shape == (None, 8, 64)
        # add a new conv1d on top
        model.add(LocallyConnected1D(32, 3))
        # now model.output_shape == (None, 6, 32)
    ```

    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: Currently only supports `"valid"` (case-insensitive).
            `"same"` may be supported in the future.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, length, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, length)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
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
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
        implementation: implementation mode, either `1`, `2`, or `3`.
            `1` loops over input spatial locations to perform the forward pass.
            It is memory-efficient but performs a lot of (small) ops.

            `2` stores layer weights in a dense but sparsely-populated 2D matrix
            and implements the forward pass as a single matrix-multiply. It uses
            a lot of RAM but performs few (large) ops.

            `3` stores layer weights in a sparse tensor and implements the forward
            pass as a single sparse matrix-multiply.

            How to choose:

            `1`: large, dense models,
            `2`: small models,
            `3`: large, sparse models,

            where "large" stands for large input/output activations
            (i.e. many `filters`, `input_filters`, large `input_size`,
            `output_size`), and "sparse" stands for few connections between inputs
            and outputs, i.e. small ratio
            `filters * input_filters * kernel_size / (input_size * strides)`,
            where inputs to and outputs of the layer are assumed to have shapes
            `(input_size, input_filters)`, `(output_size, filters)`
            respectively.

            It is recommended to benchmark each in the setting of interest to pick
            the most efficient one (in terms of speed and memory usage). Correct
            choice of implementation can lead to dramatic speed improvements (e.g.
            50X), potentially at the expense of RAM.

            Also, only `padding="valid"` is supported by `implementation=1`.

    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`

    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
  """
  return kl.LocallyConnected1D(
      filters,
      kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      implementation=implementation,
      **kwargs)


def LocallyConnected2D(
      filters,
      kernel_size,
      strides=(1, 1),
      padding='valid',
      data_format=None,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      implementation=1,
      **kwargs):
  """Locally-connected layer for 2D inputs.

    The `LocallyConnected2D` layer works similarly
    to the `Conv2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.

    Examples:
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters on a
        32x32 image
        # with `data_format="channels_last"`:
        model = Sequential()
        model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64
        parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
    ```

    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: Currently only support `"valid"` (case-insensitive).
            `"same"` will be supported in future.
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
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
        implementation: implementation mode, either `1`, `2`, or `3`.
            `1` loops over input spatial locations to perform the forward pass.
            It is memory-efficient but performs a lot of (small) ops.

            `2` stores layer weights in a dense but sparsely-populated 2D matrix
            and implements the forward pass as a single matrix-multiply. It uses
            a lot of RAM but performs few (large) ops.

            `3` stores layer weights in a sparse tensor and implements the forward
            pass as a single sparse matrix-multiply.

            How to choose:

            `1`: large, dense models,
            `2`: small models,
            `3`: large, sparse models,

            where "large" stands for large input/output activations
            (i.e. many `filters`, `input_filters`, large `np.prod(input_size)`,
            `np.prod(output_size)`), and "sparse" stands for few connections
            between inputs and outputs, i.e. small ratio
            `filters * input_filters * np.prod(kernel_size) / (np.prod(input_size)
            * np.prod(strides))`, where inputs to and outputs of the layer are
            assumed to have shapes `input_size + (input_filters,)`,
            `output_size + (filters,)` respectively.

            It is recommended to benchmark each in the setting of interest to pick
            the most efficient one (in terms of speed and memory usage). Correct
            choice of implementation can lead to dramatic speed improvements (e.g.
            50X), potentially at the expense of RAM.

            Also, only `padding="valid"` is supported by `implementation=1`.

    Input shape:
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
  """
  return kl.LocallyConnected2D(
      filters,
      kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      implementation=implementation,
      **kwargs)


def RandomFourierFeatures(
      output_dim,
      kernel_initializer='gaussian',
      scale=None,
      trainable=False,
      name=None,
      **kwargs):
  r"""Layer that maps its inputs using random Fourier features.

    This layer implements a feature map \\(\phi: \mathbb{R}^d \rightarrow
    \mathbb{R}^D\\) which approximates shift-invariant kernels. A kernel function
    K(x, y) defined over \\(\mathbb{R}^d x \mathbb{R}^d\\) is shift-invariant if
    K(x, y) = k(x-y) for some function defined over \\(\mathbb{R}^d\\). Many
    popular Radial Basis Functions (in short RBF), including gaussian and
    laplacian kernels are shift-invariant.

    The layer approximates a (shift invariant) kernel K in the following sense:
      up to a scaling factor, for all inputs \\(x, y \in \mathbb{R}^d\\)
          \\(\phi(x)^T \cdot \phi(y) \approx K(x, y)\\)

    The implementation of this layer is based on the following paper:
    "Random Features for Large-Scale Kernel Machines" by Ali Rahimi and Ben Recht.
    (link: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

    The distribution from which the parameters of the random features map (layer)
    are sampled, determines which shift-invariant kernel the layer approximates
    (see paper for more details). The users can use the distribution of their
    choice. Due to their popularity, the layer supports the out-of-the-box
    approximation of the following RBF kernels:
    - Gaussian: \\(K(x, y) = e^{-\frac{\|x-y\|_2^2}{2 \cdot scale^2}}\\)
    - Laplacian: \\(K(x, y) = e^{-\frac{\|x-y\|_1}{scale}}\\)

    NOTE: Unlike the map described in the paper and the scikit-learn
    implementation, the output of this layer does not apply the sqrt(2/D)
    normalization factor.

    Usage for ML: Typically, this layer is used to "kernelize" linear models by
    applying a non-linear transformation (this layer) to the input features and
    then training a linear model on top of the transformed features. Depending on
    the loss function of the linear model, the composition of this layer and the
    linear model results to models that are equivalent (up to approximation) to
    kernel SVMs (for hinge loss), kernel logistic regression (for logistic loss),
    kernel linear regression (for squared loss) etc.

    Example of building a kernel multinomial logistic regression model with
    Gaussian kernel in keras:
    ```python
    random_features_layer = RandomFourierFeatures(
        output_dim=500,
        kernel_initializer='gaussian',
        scale=5.0,
        ...)

    model = tf.keras.models.Sequential()
    model.add(random_features_layer)
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax')

    model.compile(
      loss=tf.keras.losses.categorical_crossentropy, optimizer=..., metrics=...)
    ```

    To use another kernel, replace the layer creation command with:
    ```python
    random_features_layer = RandomFourierFeatures(
        output_dim=500,
        kernel_initializer=<my_initializer>,
        scale=...,
        ...)
    ```

    Arguments:
      output_dim: Positive integer, the dimension of the layer's output, i.e., the
        number of random features used to approximate the kernel.
      kernel_initializer: Determines the distribution of the parameters of the
        random features map (and therefore the kernel approximated by the layer).
        It can be either a string or an instance of TensorFlow's Initializer
        class. Currently only 'gaussian' and 'laplacian' are supported as string
        initializers (case insensitive). Note that these parameters are not
        trainable.
      scale: For gaussian and laplacian kernels, this corresponds to a scaling
        factor of the corresponding kernel approximated by the layer (see concrete
        definitions above). When provided, it should be a positive float. If None,
        the implementation chooses a default value (1.0 typically). Both the
        approximation error of the kernel and the classification quality are
        sensitive to this parameter. If trainable is set to True, this parameter
        is learned end-to-end during training and the provided value serves as an
        initialization value.
        NOTE: When this layer is used to map the initial features and then the
          transformed features are fed to a linear model, by making `scale`
          trainable, the resulting optimization problem is no longer convex (even
          if the loss function used by the linear model is convex).
      trainable: Whether the scaling parameter of th layer is trainable. Defaults
        to False.
      name: name for the RandomFourierFeatures layer.

    Raises:
      ValueError: if output_dim or stddev are not positive or if the provided
        kernel_initializer is not supported.
  """
  return kl.RandomFourierFeatures(
      output_dim,
      kernel_initializer=kernel_initializer,
      scale=scale,
      trainable=trainable,
      name=name,
      **kwargs)


def ResidualWrapper(
      *args,
      **kwargs):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""
  return kl.ResidualWrapper(
      *args,
      **kwargs)


def serialize(layer):
  return kl.serialize(layer)


def TimeDistributed(
      layer,
      **kwargs):
  """This wrapper allows to apply a layer to every temporal slice of an input.

    The input should be at least 3D, and the dimension of index one
    will be considered to be the temporal dimension.

    Consider a batch of 32 samples,
    where each sample is a sequence of 10 vectors of 16 dimensions.
    The batch input shape of the layer is then `(32, 10, 16)`,
    and the `input_shape`, not including the samples dimension, is `(10, 16)`.

    You can then use `TimeDistributed` to apply a `Dense` layer
    to each of the 10 timesteps, independently:

    ```python
    # as the first layer in a model
    model = Sequential()
    model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
    # now model.output_shape == (None, 10, 8)
    ```

    The output will then have shape `(32, 10, 8)`.

    In subsequent layers, there is no need for the `input_shape`:

    ```python
    model.add(TimeDistributed(Dense(32)))
    # now model.output_shape == (None, 10, 32)
    ```

    The output will then have shape `(32, 10, 32)`.

    `TimeDistributed` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:

    ```python
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3, 3)),
                              input_shape=(10, 299, 299, 3)))
    ```

    Arguments:
      layer: a layer instance.

    Call arguments:
      inputs: Input tensor.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the
        wrapped layer (only if the layer supports this argument).
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked. This argument is passed to the
        wrapped layer (only if the layer supports this argument).

    Raises:
      ValueError: If not initialized with a `Layer` instance.
  """
  return kl.TimeDistributed(
      layer,
      **kwargs)


class Warpper(kl.Wrapper):
  """Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    Arguments:
      layer: The layer to be wrapped.
  """

