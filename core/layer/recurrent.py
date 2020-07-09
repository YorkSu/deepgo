# -*- coding: utf-8 -*-
"""Keras Recurrent Layer
  =====

  Keras Layer, containing Recurrent Layers
"""


__all__ = [
    'AbstractRNNCell',
    'ConvLSTM2D',
    'CuDNNGRU',
    'CuDNNLSTM',
    'GRU',
    'GRUV1',
    'GRUCell',
    'GRUCellV1',
    'LSTM',
    'LSTMV1',
    'LSTMCell',
    'LSTMCellV1',
    'PeepholeLSTMCell',
    'RNN',
    'SimpleRNN',
    'SimpleRNNCell',
    'StackedRNNCells',]


from tensorflow.keras import layers as kl


class AbstractRNNCell(kl.AbstractRNNCell):
  """Abstract object representing an RNN cell.

    NOTE: This is a doc of Keras.AbstractRNNCell

    This is the base class for implementing RNN cells with custom behavior.

    Every `RNNCell` must have the properties below and implement `call` with
    the signature `(output, next_state) = call(input, state)`.

    Examples:

    ```python
      class MinimalRNNCell(AbstractRNNCell):

        def __init__(self, units, **kwargs):
          self.units = units
          super(MinimalRNNCell, self).__init__(**kwargs)

        @property
        def state_size(self):
          return self.units

        def build(self, input_shape):
          self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer='uniform',
                                        name='kernel')
          self.recurrent_kernel = self.add_weight(
              shape=(self.units, self.units),
              initializer='uniform',
              name='recurrent_kernel')
          self.built = True

        def call(self, inputs, states):
          prev_output = states[0]
          h = K.dot(inputs, self.kernel)
          output = h + K.dot(prev_output, self.recurrent_kernel)
          return output, output
    ```

    This definition of cell differs from the definition used in the literature.
    In the literature, 'cell' refers to an object with a single scalar output.
    This definition refers to a horizontal array of such units.

    An RNN cell, in the most abstract setting, is anything that has
    a state and performs some operation that takes a matrix of inputs.
    This operation results in an output matrix with `self.output_size` columns.
    If `self.state_size` is an integer, this operation also results in a new
    state matrix with `self.state_size` columns.  If `self.state_size` is a
    (possibly nested tuple of) TensorShape object(s), then it should return a
    matching structure of Tensors having shape `[batch_size].concatenate(s)`
    for each `s` in `self.batch_size`.
  """


def ConvLSTM2D(
      filters,
      kernel_size,
      strides=(1, 1),
      padding='valid',
      data_format=None,
      dilation_rate=(1, 1),
      activation='tanh',
      recurrent_activation='hard_sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      unit_forget_bias=True,
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      return_sequences=False,
      go_backwards=False,
      stateful=False,
      dropout=0.,
      recurrent_dropout=0.,
      **kwargs):
  """Convolutional LSTM.

    It is similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Arguments:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, time, ..., channels)`
        while `channels_first` corresponds to
        inputs with shape `(batch, time, channels, ...)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
        By default hyperbolic tangent activation function is applied
        (`tanh(x)`).
      recurrent_activation: Activation function to use
        for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Use in combination with `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et al.]
        (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.

    Call arguments:
      inputs: A 5D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or `recurrent_dropout`
        are set.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.

    Input shape:
      - If data_format='channels_first'
          5D tensor with shape:
          `(samples, time, channels, rows, cols)`
      - If data_format='channels_last'
          5D tensor with shape:
          `(samples, time, rows, cols, channels)`

    Output shape:
      - If `return_sequences`
        - If data_format='channels_first'
            5D tensor with shape:
            `(samples, time, filters, output_row, output_col)`
        - If data_format='channels_last'
            5D tensor with shape:
            `(samples, time, output_row, output_col, filters)`
      - Else
        - If data_format ='channels_first'
            4D tensor with shape:
            `(samples, filters, output_row, output_col)`
        - If data_format='channels_last'
            4D tensor with shape:
            `(samples, output_row, output_col, filters)`
        where `o_row` and `o_col` depend on the shape of the filter and
        the padding

    Raises:
      ValueError: in case of invalid constructor arguments.

    References:
      - [Convolutional LSTM Network: A Machine Learning Approach for
      Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
      The current implementation does not include the feedback loop on the
      cells output.
  """
  return kl.ConvLSTM2D(
      filters,
      kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      unit_forget_bias=unit_forget_bias,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      return_sequences=return_sequences,
      go_backwards=go_backwards,
      stateful=stateful,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      **kwargs)


def CuDNNGRU(
      units,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      return_sequences=False,
      return_state=False,
      go_backwards=False,
      stateful=False,
      **kwargs):
  """Fast GRU implementation backed by cuDNN.

    More information about cuDNN can be found on the [NVIDIA
    developer website](https://developer.nvidia.com/cudnn).
    Can only be run on GPU.

    Arguments:
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the `kernel` weights matrix, used for
          the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights
          matrix, used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
          matrix.
        recurrent_regularizer: Regularizer function applied to the
          `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the
          layer (its "activation").
        kernel_constraint: Constraint function applied to the `kernel` weights
          matrix.
        recurrent_constraint: Constraint function applied to the
          `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        return_sequences: Boolean. Whether to return the last output in the output
          sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state in addition to the
          output.
        go_backwards: Boolean (default False). If True, process the input sequence
          backwards and return the reversed sequence.
        stateful: Boolean (default False). If True, the last state for each sample
          at index i in a batch will be used as initial state for the sample of
          index i in the following batch.
  """
  return kl.CuDNNGRU(
      units,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      **kwargs)


def CuDNNLSTM(
      units,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      unit_forget_bias=True,
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      return_sequences=False,
      return_state=False,
      go_backwards=False,
      stateful=False,
      **kwargs):
  """Fast LSTM implementation backed by cuDNN.

    More information about cuDNN can be found on the [NVIDIA
    developer website](https://developer.nvidia.com/cudnn).
    Can only be run on GPU.

    Arguments:
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the `kernel` weights matrix, used for
          the linear transformation of the inputs.
        unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate
          at initialization. Setting it to true will also force
          `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        recurrent_initializer: Initializer for the `recurrent_kernel` weights
          matrix, used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
          matrix.
        recurrent_regularizer: Regularizer function applied to the
          `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the
          layer (its "activation").
        kernel_constraint: Constraint function applied to the `kernel` weights
          matrix.
        recurrent_constraint: Constraint function applied to the
          `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        return_sequences: Boolean. Whether to return the last output. in the
          output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state in addition to the
          output.
        go_backwards: Boolean (default False). If True, process the input sequence
          backwards and return the reversed sequence.
        stateful: Boolean (default False). If True, the last state for each sample
          at index i in a batch will be used as initial state for the sample of
          index i in the following batch.
  """
  return kl.CuDNNLSTM(
      units,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer = bias_initializer,
      unit_forget_bias=unit_forget_bias,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      **kwargs)


def GRU(
      units,
      activation='tanh',
      recurrent_activation='hard_sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=1,
      return_sequences=False,
      return_state=False,
      go_backwards=False,
      stateful=False,
      unroll=False,
      reset_after=False,
      **kwargs):
  """Gated Recurrent Unit - Cho et al. 2014.

    There are two variants. The default one is based on 1406.1078v3 and
    has reset gate applied to hidden state before matrix multiplication. The
    other one is based on original 1406.1078v1 and has the order reversed.

    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. Use `'reset_after'=True` and
    `recurrent_activation='sigmoid'`.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
      time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `(timesteps, batch, ...)`, whereas in the False case, it will be
        `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form. 
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before" (default),
        True = "after" (CuDNN compatible).

    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
  """
  return kl.GRU(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      unroll=unroll,
      reset_after=reset_after,
      **kwargs)


def GRUV1(
      units,
      activation='tanh',
      recurrent_activation='sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=2,
      return_sequences=False,
      return_state=False,
      go_backwards=False,
      stateful=False,
      unroll=False,
      time_major=False,
      reset_after=False,
      **kwargs):
  """Gated Recurrent Unit - Cho et al. 2014.

    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or pure-TensorFlow)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the CuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation.

    The requirements to use the cuDNN implementation are:

    1. `activation` == 'tanh'
    2. `recurrent_activation` == 'sigmoid'
    3. `recurrent_dropout` == 0
    4. `unroll` is False
    5. `use_bias` is True
    6. `reset_after` is True
    7. Inputs are not masked or strictly right padded.

    There are two variants of the GRU implementation. The default one is based on
    [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to hidden
    state before matrix multiplication. The other one is based on
    [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. To use this variant, set `'reset_after'=True` and
    `recurrent_activation='sigmoid'`.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: sigmoid (`sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before",
        True = "after" (default and CuDNN compatible).

    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
  """
  return kl.GRUV1(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      unroll=unroll,
      time_major=time_major,
      reset_after=reset_after,
      **kwargs)


def GRUCell(
      units,
      activation='tanh',
      recurrent_activation='hard_sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=1,
      reset_after=False,
      **kwargs):
  """Cell class for the GRU layer.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before" (default),
        True = "after" (CuDNN compatible).

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
  """
  return kl.GRUCell(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      reset_after=reset_after,
      **kwargs)


def GRUCellV1(
      units,
      activation='tanh',
      recurrent_activation='sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=2,
      reset_after=False,
      **kwargs):
  """Cell class for the GRU layer.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use. Default: hyperbolic tangent
        (`tanh`). If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use for the recurrent step.
        Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
        applied (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the `recurrent_kernel`
        weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 (default) will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before",
        True = "after" (default and CuDNN compatible).

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
  """
  return kl.GRUCellV1(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      reset_after=reset_after,
      **kwargs)


def LSTM(
      units,
      activation='tanh',
      recurrent_activation='hard_sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      unit_forget_bias=True,
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=1,
      return_sequences=False,
      return_state=False,
      go_backwards=False,
      stateful=False,
      unroll=False,
      **kwargs):
  """Long Short-Term Memory layer - Hochreiter 1997.

    Note that this cell is not optimized for performance on GPU. Please use
    `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Setting it to true will also force `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
      return_sequences: Boolean. Whether to return the last output.
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
      time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `(timesteps, batch, ...)`, whereas in the False case, it will be
        `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form.

    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
  """
  return kl.LSTM(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      unit_forget_bias=unit_forget_bias,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      unroll=unroll,
      **kwargs)


def LSTMV1(
      units,
      activation='tanh',
      recurrent_activation='sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      unit_forget_bias=True,
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=2,
      return_sequences=False,
      return_state=False,
      go_backwards=False,
      stateful=False,
      time_major=False,
      unroll=False,
      **kwargs):
  """Long Short-Term Memory layer - Hochreiter 1997.

    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or pure-TensorFlow)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the CuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation.

    The requirements to use the cuDNN implementation are:

    1. `activation` == 'tanh'
    2. `recurrent_activation` == 'sigmoid'
    3. `recurrent_dropout` == 0
    4. `unroll` is False
    5. `use_bias` is True
    6. Inputs are not masked or strictly right padded.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
        is applied (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use for the recurrent step.
        Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
        applied (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state..
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at
        initialization. Setting it to true will also force
        `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation")..
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the `recurrent_kernel`
        weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1. Fraction of the units to drop for the linear
        transformation of the inputs.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2. Mode 1 will structure
        its operations as a larger number of smaller dot products and additions,
        whereas mode 2 will batch them into fewer, larger operations. These modes
        will have different performance profiles on different hardware and for
        different applications.
      return_sequences: Boolean. Whether to return the last output. in the output
        sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state in addition to the
        output.
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
      unroll: Boolean (default False). If True, the network will be unrolled, else
        a symbolic loop will be used. Unrolling can speed-up a RNN, although it
        tends to be more memory-intensive. Unrolling is only suitable for short
        sequences.

    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
  """
  return kl.LSTMV1(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      unit_forget_bias=unit_forget_bias,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      time_major=time_major,
      unroll=unroll,
      **kwargs)


def LSTMCell(
      units,
      activation='tanh',
      recurrent_activation='hard_sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      unit_forget_bias=True,
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=1,
      **kwargs):
  """Cell class for the LSTM layer.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Setting it to true will also force `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
  """
  return kl.LSTMCell(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      unit_forget_bias=unit_forget_bias,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      **kwargs)


def LSTMCellV1(
      units,
      activation='tanh',
      recurrent_activation='sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      unit_forget_bias=True,
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=2,
      **kwargs):
  """Cell class for the LSTM layer.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use. Default: hyperbolic tangent
        (`tanh`). If you pass `None`, no activation is applied (ie. "linear"
        activation: `a(x) = x`).
      recurrent_activation: Activation function to use for the recurrent step.
        Default: sigmoid (`sigmoid`). If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at
        initialization. Setting it to true will also force
        `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the `recurrent_kernel`
        weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1. Fraction of the units to drop for the linear
        transformation of the inputs.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of smaller dot
        products and additions, whereas mode 2 (default) will batch them into
        fewer, larger operations. These modes will have different performance
        profiles on different hardware and for different applications.

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
  """
  return kl.LSTMCellV1(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      unit_forget_bias=unit_forget_bias,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      **kwargs)


def PeepholeLSTMCell(
      units,
      activation='tanh',
      recurrent_activation='hard_sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      unit_forget_bias=True,
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      implementation=1,
      **kwargs):
  """Equivalent to LSTMCell class but adds peephole connections.

    Peephole connections allow the gates to utilize the previous internal state as
    well as the previous hidden state (which is what LSTMCell is limited to).
    This allows PeepholeLSTMCell to better learn precise timings over LSTMCell.

    From [Gers et al.](http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf):

    "We find that LSTM augmented by 'peephole connections' from its internal
    cells to its multiplicative gates can learn the fine distinction between
    sequences of spikes spaced either 50 or 49 time steps apart without the help
    of any short training exemplars."

    The peephole implementation is based on:

    [Long short-term memory recurrent neural network architectures for
    large scale acoustic modeling.
    ](https://research.google.com/pubs/archive/43905.pdf)

    Example:

    ```python
    # Create 2 PeepholeLSTMCells
    peephole_lstm_cells = [PeepholeLSTMCell(size) for size in [128, 256]]
    # Create a layer composed sequentially of the peephole LSTM cells.
    layer = RNN(peephole_lstm_cells)
    input = keras.Input((timesteps, input_dim))
    output = layer(input)
    ```
  """
  return kl.PeepholeLSTMCell(
      units,
      activation=activation,
      recurrent_activation=recurrent_activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      unit_forget_bias=unit_forget_bias,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      **kwargs)


def RNN(
      cells,
      return_sequences=False,
      return_state=False,
      go_backwards=False,
      stateful=False,
      unroll=False,
      time_major=False,
      **kwargs):
  """Base class for recurrent layers.

    Arguments:
      cell: A RNN cell instance or a list of RNN cell instances.
        A RNN cell is a class that has:
        - A `call(input_at_t, states_at_t)` method, returning
          `(output_at_t, states_at_t_plus_1)`. The call method of the
          cell can also take the optional argument `constants`, see
          section "Note on passing external constants" below.
        - A `state_size` attribute. This can be a single integer
          (single state) in which case it is the size of the recurrent
          state. This can also be a list/tuple of integers (one size per
          state).
          The `state_size` can also be TensorShape or tuple/list of
          TensorShape, to represent high dimension state.
        - A `output_size` attribute. This can be a single integer or a
          TensorShape, which represent the shape of the output. For backward
          compatible reason, if this attribute is not available for the
          cell, the value will be inferred by the first element of the
          `state_size`.
        - A `get_initial_state(inputs=None, batch_size=None, dtype=None)`
          method that creates a tensor meant to be fed to `call()` as the
          initial state, if the user didn't specify any initial state via other
          means. The returned initial state should have a shape of
          [batch_size, cell.state_size]. The cell might choose to create a
          tensor full of zeros, or full of other values based on the cell's
          implementation.
          `inputs` is the input tensor to the RNN layer, which should
          contain the batch size as its shape[0], and also dtype. Note that
          the shape[0] might be `None` during the graph construction. Either
          the `inputs` or the pair of `batch_size` and `dtype` are provided.
          `batch_size` is a scalar tensor that represents the batch size
          of the inputs. `dtype` is `tf.DType` that represents the dtype of
          the inputs.
          For backward compatible reason, if this method is not implemented
          by the cell, the RNN layer will create a zero filled tensor with the
          size of [batch_size, cell.state_size].
        In the case that `cell` is a list of RNN cell instances, the cells
        will be stacked on top of each other in the RNN, resulting in an
        efficient stacked RNN.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled, else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
      time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `(timesteps, batch, ...)`, whereas in the False case, it will be
        `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form.

    Call arguments:
      inputs: Input tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is for use with cells that use dropout.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
      constants: List of constant tensors to be passed to the cell at each
        timestep.

    Input shape:
      N-D tensor with shape `(batch_size, timesteps, ...)` or
      `(timesteps, batch_size, ...)` when time_major is True.

    Output shape:
      - If `return_state`: a list of tensors. The first tensor is
        the output. The remaining tensors are the last states,
        each with shape `(batch_size, state_size)`, where `state_size` could
        be a high dimension tensor shape.
      - If `return_sequences`: N-D tensor with shape
        `(batch_size, timesteps, output_size)`, where `output_size` could
        be a high dimension tensor shape, or
        `(timesteps, batch_size, output_size)` when `time_major` is True.
      - Else, N-D tensor with shape `(batch_size, output_size)`, where
        `output_size` could be a high dimension tensor shape.

    Masking:
      This layer supports masking for input data with a variable number
      of timesteps. To introduce masks to your data,
      use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
      set to `True`.

    Note on using statefulness in RNNs:
      You can set RNN layers to be 'stateful', which means that the states
      computed for the samples in one batch will be reused as initial states
      for the samples in the next batch. This assumes a one-to-one mapping
      between samples in different successive batches.

      To enable statefulness:
        - Specify `stateful=True` in the layer constructor.
        - Specify a fixed batch size for your model, by passing
          If sequential model:
            `batch_input_shape=(...)` to the first layer in your model.
          Else for functional model with 1 or more Input layers:
            `batch_shape=(...)` to all the first layers in your model.
          This is the expected shape of your inputs
          *including the batch size*.
          It should be a tuple of integers, e.g. `(32, 10, 100)`.
        - Specify `shuffle=False` when calling fit().

      To reset the states of your model, call `.reset_states()` on either
      a specific layer, or on your entire model.

    Note on specifying the initial state of RNNs:
      You can specify the initial state of RNN layers symbolically by
      calling them with the keyword argument `initial_state`. The value of
      `initial_state` should be a tensor or list of tensors representing
      the initial state of the RNN layer.

      You can specify the initial state of RNN layers numerically by
      calling `reset_states` with the keyword argument `states`. The value of
      `states` should be a numpy array or list of numpy arrays representing
      the initial state of the RNN layer.

    Note on passing external constants to RNNs:
      You can pass "external" constants to the cell using the `constants`
      keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
      requires that the `cell.call` method accepts the same keyword argument
      `constants`. Such constants can be used to condition the cell
      transformation on additional static inputs (not changing over time),
      a.k.a. an attention mechanism.

    Examples:

    ```python
    # First, let's define a RNN Cell, as a layer subclass.

    class MinimalRNNCell(keras.layers.Layer):

        def __init__(self, units, **kwargs):
            self.units = units
            self.state_size = units
            super(MinimalRNNCell, self).__init__(**kwargs)

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='uniform',
                                          name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')
            self.built = True

        def call(self, inputs, states):
            prev_output = states[0]
            h = K.dot(inputs, self.kernel)
            output = h + K.dot(prev_output, self.recurrent_kernel)
            return output, [output]

    # Let's use this cell in a RNN layer:

    cell = MinimalRNNCell(32)
    x = keras.Input((None, 5))
    layer = RNN(cell)
    y = layer(x)

    # Here's how to use the cell to build a stacked RNN:

    cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
    x = keras.Input((None, 5))
    layer = RNN(cells)
    y = layer(x)
    ```
  """
  return kl.RNN(
      cells,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      unroll=unroll,
      time_major=time_major,
      **kwargs)


def SimpleRNN(
      units,
      activation='tanh',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      return_sequences=False,
      return_state=False,
      go_backwards=False,
      stateful=False,
      unroll=False,
      **kwargs):
  """Fully-connected RNN where the output is to be fed back to input.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.

    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
  """
  return kl.SimpleRNN(
      units,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      unroll=unroll,
      **kwargs)


def SimpleRNNCell(
      units,
      activation='tanh',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      recurrent_initializer='orthogonal',
      bias_initializer='zeros',
      kernel_regularizer=None,
      recurrent_regularizer=None,
      bias_regularizer=None,
      kernel_constraint=None,
      recurrent_constraint=None,
      bias_constraint=None,
      dropout=0.,
      recurrent_dropout=0.,
      **kwargs):
  """Cell class for SimpleRNN.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
  """
  return kl.SimpleRNNCell(
      units,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      **kwargs)


def StackedRNNCells(
      cells,
      **kwargs):
  """Wrapper allowing a stack of RNN cells to behave as a single cell.

    Used to implement efficient stacked RNNs.

    Arguments:
      cells: List of RNN cell instances.

    Examples:

    ```python
    cells = [
        keras.layers.LSTMCell(output_dim),
        keras.layers.LSTMCell(output_dim),
        keras.layers.LSTMCell(output_dim),
    ]

    inputs = keras.Input((timesteps, input_dim))
    x = keras.layers.RNN(cells)(inputs)
    ```
  """
  return kl.StackedRNNCells(
      cells,
      **kwargs)

