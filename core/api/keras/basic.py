# -*- coding: utf-8 -*-
"""Keras Basic API
  =====

  Keras API, containing basic methods
"""


from tensorflow.keras import backend as K


def binary_crossentropy(target, output, from_logits=False):
  """Keras.backend.binary_crossentropy
    
    Binary crossentropy between an output tensor and a target tensor.

    Args:
      target: A tensor with the same shape as output.
      output: A tensor.
      from_logits: Whether output is expected to be a logits tensor. 
          By default, we consider that output encodes a probability 
          distribution.
  """
  return K.binary_crossentropy(target, output, from_logits=from_logits)


def cast(x, dtype):
  """Keras.backend.cast
    
    Casts a tensor to a different dtype and returns it.

    Args:
      x: Tensor
      dtype: str, either ('float16', 'float32', or 'float64').
  """
  return K.cast(x, dtype)


def cast_to_floatx(x):
  """Keras.backend.cast_to_floatx
    
    Cast a Numpy array to the default Keras float type.

    Args:
      x: Tensor
  """
  return K.cast_to_floatx(x)


def categorical_crossentropy(target, output, from_logits=False,
    axis=-1):
  """Keras.backend.categorical_crossentropy
    
    Categorical crossentropy between an output tensor and a target tensor.

    Args:
      target: A tensor of the same shape as output.
      output: A tensor resulting from a softmax (unless from_logits is True, 
          in which case output is expected to be the logits).
      from_logits: Boolean, whether output is the result of a softmax, or is 
          a tensor of logits.
      axis: Int specifying the channels axis. 
          axis=-1 corresponds to data format channels_last', 
          axis=1 corresponds to data format channels_first`.
  """
  return K.categorical_crossentropy(target, output,
      from_logits=from_logits, axis=axis)


def conv1d(x, kernel, strides=1, padding='valid', data_format=None,
    dilation_rate=1):
  """Keras.backend.conv1d
    
    1D convolution.

    Args:
      x: Tensor
      kernel: kernel tensor.
      strides: stride integer.
      padding: string, "same", "causal" or "valid".
      data_format: string, one of "channels_last", "channels_first".
      dilation_rate: integer dilate rate.
  """
  return K.conv1d(x, kernel, strides=strides, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate)


def conv2d(x, kernel, strides=(1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1)):
  """Keras.backend.conv2d
    
    2D convolution.

    Args:
      x: Tensor
      kernel: kernel tensor.
      strides: strides tuple.
      padding: string, "same" or "valid".
      data_format: string, one of "channels_last", "channels_first".
      dilation_rate: tuple of 2 integers.
  """
  return K.conv2d(x, kernel, strides=strides, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate)


def conv2d_transpose(x, kernel, output_shape, strides=(1, 1),
    padding='valid', data_format=None, dilation_rate=(1, 1)):
  """Keras.backend.conv2d_transpose
    
    2D deconvolution (i.e. transposed convolution).

    Args:
      x: Tensor
      kernel: kernel tensor.
      output_shape: 1D int tensor for the output shape.
      strides: strides tuple.
      padding: string, "same" or "valid".
      data_format: string, one of "channels_last", "channels_first".
      dilation_rate: tuple of 2 integers.
  """
  return K.conv2d_transpose(x, kernel, output_shape, strides = strides,
      padding=padding, data_format=data_format, dilation_rate=dilation_rate)


def conv3d(x, kernel, strides=(1, 1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1, 1)):
  """Keras.backend.conv3d
    
    3D convolution.

    Args:
      x: Tensor
      kernel: kernel tensor.
      strides: strides tuple.
      padding: string, "same" or "valid".
      data_format: string, one of "channels_last", "channels_first".
      dilation_rate: tuple of 3 integers.
  """
  return K.conv3d(x, kernel, strides=strides, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate)


def count_params(x):
  """Keras.backend.count_params

    Returns the static number of elements in a variable or tensor.
  
    Args:
      x: Tensor
  """
  return K.count_params(x)


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
  """Keras.backend.ctc_batch_cost

    Runs CTC loss algorithm on each batch element.
  
    Args:
      y_true: tensor (samples, max_string_length) containing the 
          truth labels.
      y_pred: tensor (samples, time_steps, num_categories) containing 
          the prediction, or output of the softmax.
      input_length: tensor (samples, 1) containing the sequence length 
          for each batch item in y_pred.
      label_length: tensor (samples, 1) containing the sequence length 
          for each batch item in y_true.
  """
  return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100,
    top_paths=1):
  """Keras.backend.ctc_decode

    Decodes the output of a softmax.
  
    Args:
      y_true: tensor (samples, time_steps, num_categories) containing 
          the prediction, or output of the softmax.
      input_length: tensor (samples, ) containing the sequence length 
          for each batch item in y_pred.
      greedy: perform much faster best-path search if true. This does 
          not use a dictionary.
      beam_width: if greedy is false: a beam search decoder will be used 
          with a beam of this width.
      top_paths: if greedy is false, how many of the most probable paths 
          will be returned.
  """
  return K.ctc_decode(y_pred, input_length, greedy=greedy,
      beam_width=beam_width, top_paths=top_paths)


def ctc_label_dense_to_sparse(labels, label_lengths):
  """Keras.backend.ctc_label_dense_to_sparse

    Converts CTC labels from dense to sparse.
  
    Args:
      labels: dense CTC labels.
      label_lengths: length of the labels.
  """
  return K.ctc_label_dense_to_sparse(labels, label_lengths)


def dropout(x, level, noise_shape=None, seed=None):
  """Keras.backend.dropout

    Sets entries in x to zero at random, while scaling the entire tensor.
  
    Args:
      x: Tensor
      level: fraction of the entries in the tensor that will be set to 0.
      noise_shape: 	shape for randomly generated keep/drop flags, must be 
          broadcastable to the shape of x
      seed: random seed to ensure determinism.
  """
  return K.dropout(x, level, noise_shape=noise_shape, seed=seed)


def dtype(x):
  """Keras.backend.dtype

    Returns the dtype of a Keras tensor or variable, as a string.
  
    Args:
      x: Tensor
  """
  return K.dtype(x)


def elu(x, alpha=1.0):
  """Keras.backend.elu

    Exponential linear unit.
  
    Args:
      x: Tensor
      alpha: A scalar, slope of negative section.
  """
  return K.elu(x, alpha=alpha)


def eval(x):
  """Keras.backend.eval

    Evaluates the value of a variable.
  
    Args:
      x: Tensor
  """
  return K.eval(x)


def function(inputs, outputs, updates=None, name=None, **kwargs):
  """Keras.backend.function

    Instantiates a Keras function.
    SEE: https://tensorflow.google.cn/versions/r2.1/api_docs/python/tf/keras/backend/function

    Args:
      inputs: List of placeholder tensors.
      outputs: List of output tensors.
      updates: List of update ops.
      name: String, name of function.
      **kwargs: Passed to tf.Session.run.
  """
  return K.function(inputs, outputs, updates=updates, name=name,
      **kwargs)


def get_value(x):
  """Keras.backend.get_value

    Returns the value of a variable.
  
    Args:
      x: Tensor
  """
  return K.get_value(x)


def gradients(loss, variables):
  """Keras.backend.gradients

    Returns the gradients of loss w.r.t. variables.
  
    Args:
      loss: Scalar tensor to minimize.
      variables: List of variables.
  """
  return K.gradients(loss, variables)


def hard_sigmoid(x):
  """Keras.backend.hard_sigmoid

    Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.
  
    Args:
      x: Tensor

    Returns:
      0, if (x < -2.5) 
      1, if x > 2.5
      0.2 * x + 0.5, In -2.5 <= x <= 2.5
  """
  return K.hard_sigmoid(x)


def int_shape(x):
  """Keras.backend.int_shape

    Returns the shape of tensor or variable as a tuple of int or None 
    entries.
  
    Args:
      x: Tensor
  """
  return K.int_shape(x)


def in_test_phase(x, alt, training=None):
  """Keras.backend.in_test_phase

    Selects x in test phase, and alt otherwise.
  
    Args:
      x: What to return in test phase (tensor or callable that returns a 
          tensor).
      alt: What to return otherwise (tensor or callable that returns a 
          tensor).
      training: Optional scalar tensor (or Python boolean, or Python 
          integer) specifying the learning phase.
  """
  return K.in_test_phase(x, alt, training=training)


def in_train_phase(x, alt, training=None):
  """Keras.backend.in_train_phase

    Selects x in train phase, and alt otherwise.
  
    Args:
      x: What to return in train phase (tensor or callable that returns a 
          tensor).
      alt: What to return otherwise (tensor or callable that returns a 
          tensor).
      training: Optional scalar tensor (or Python boolean, or Python 
          integer) specifying the learning phase.
  """
  return K.in_train_phase(x, alt, training=training)


def is_keras_tensor(x):
  """Keras.backend.is_keras_tensor

    Returns whether x is a Keras tensor.
  
    Args:
      x: object
  """
  return K.is_keras_tensor(x)


def is_sparse(tensor):
  """Keras.backend.is_sparse

    Returns whether a tensor is a sparse tensor.
  
    Args:
      tensor: Tensor
  """
  return K.is_sparse(tensor)


def l2_normalize(x, axis=None):
  """Keras.backend.l2_normalize

    Normalizes a tensor wrt the L2 norm alongside the specified axis.
  
    Args:
      x: Tensor
      axis: axis along which to perform normalization.
  """
  return K.l2_normalize(x, axis=axis)


def learning_phase_scope(value):
  """Keras.backend.learning_phase_scope

    Provides a scope within which the learning phase is equal to value.
  
    Args:
      value: Learning phase value, either 0 or 1 (integers). 
          0 = test, 
          1 = train
  """
  return K.learning_phase_scope(value)


def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
  """Keras.backend.local_conv1d
    
    Apply 1D conv with un-shared weights.

    Args:
      inputs: 3D tensor with shape: 
          (batch_size, steps, input_dim) if "channels_last" or 
          (batch_size, input_dim, steps) if "channels_first".
      kernel: the unshared weight for convolution, 
          with shape (output_length, feature_dim, filters).
      kernel_size: a tuple of a single integer, specifying the length of 
          the 1D convolution window.
      strides: a tuple of a single integer, specifying the stride length of 
          the convolution.
      data_format: string, one of "channels_last", "channels_first".
  """
  return K.local_conv1d(inputs, kernel, kernel_size, strides,
      data_format=data_format)


def local_conv2d(inputs, kernel, kernel_size, strides, output_shape,
    data_format=None):
  """Keras.backend.local_conv2d
    
    Apply 2D conv with un-shared weights.

    Args:
      inputs: 	4D tensor with shape: 
          (batch_size, filters, new_rows, new_cols) if 'channels_first' or 
          (batch_size, new_rows, new_cols, filters) if 'channels_last'.
      kernel: the unshared weight for convolution, 
          with shape (output_items, feature_dim, filters).
      kernel_size: a tuple of 2 integers, specifying the width and height of 
          the 2D convolution window.
      strides: a tuple of 2 integers, specifying the strides of the 
          convolution along the width and height.
      output_shape: a tuple with (output_row, output_col).
      data_format: string, one of "channels_last", "channels_first".
  """
  return K.local_conv2d(inputs, kernel, kernel_size, strides, output_shape,
      data_format=data_format)


def manual_variable_initialization(value):
  """Keras.backend.manual_variable_initialization

    Sets the manual variable initialization flag.

    This boolean flag determines whether variables should be initialized 
    as they are instantiated (default), or if the user should handle the 
    initialization (e.g. via tf.compat.v1.initialize_all_variables()).
  
    Args:
      value: Python boolean.
  """
  K.manual_variable_initialization(value)


def moving_average_update(x, value, momentum):
  """Keras.backend.moving_average_update

    Compute the moving average of a variable.
  
    Args:
      x: Tensor Variable
      value: A tensor with the same shape as variable.
      momentum: The moving average momentum.
  """
  return K.moving_average_update(x, value, momentum)


def name_scope(name):
  """Keras.backend.name_scope

    A context manager for use when defining a Python op.
  
    Args:
      name: The prefix to use on all names created within the name scope.
  """
  return K.name_scope(name)


def ndim(x):
  """Keras.backend.ndim

    Returns the number of axes in a tensor, as an integer.
  
    Args:
      x: Tensor
  """
  return K.ndim(x)


def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.001):
  """Keras.backend.normalize_batch_in_training
    
    Computes mean and std for batch then apply batch_normalization on batch.

    Args:
      x: Tensor
      gamma: Tensor by which to scale the input.
      beta: Tensor with which to center the input.
      reduction_axes: iterable of integers, axes over which to normalize.
      epsilon: Fuzz factor.
  """
  return K.normalize_batch_in_training(x, gamma, beta,
      reduction_axes=reduction_axes, epsilon=epsilon)


def placeholder(shape=None, ndim=None, dtype=None, sparse=False,
    name=None, ragged=False):
  """Keras.backend.placeholder

    Instantiates a placeholder tensor and returns it.
  
    Args:
      shape: Shape of the placeholder (integer tuple, may include None 
          entries).
      ndim: Number of axes of the tensor. At least one of {shape, ndim} must
          be specified. If both are specified, shape is used.
      dtype: Placeholder type.
      sparse: Boolean, whether the placeholder should have a sparse type.
      name: Optional name string for the placeholder.
      ragged: Boolean, whether the placeholder should have a ragged type. 
          In this case, values of 'None' in the 'shape' argument represent 
          ragged dimensions. For more information about RaggedTensors, see 
          this guide.
  """
  return K.placeholder(shape=shape, ndim=ndim, dtype=dtype, sparse=sparse, 
      name=name, ragged=ragged)


def pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None,
    pool_mode='max'):
  """Keras.backend.pool2d

    2D Pooling.
  
    Args:
      x: Tensor
      pool_size: tuple of 2 integers.
      strides: tuple of 2 integers.
      padding: string, "same" or "valid".
      data_format: string, "channels_last" or "channels_first".
      pool_mode: string, "max" or "avg".
  """
  return K.pool2d(x, pool_size, strides=strides, padding=padding, 
      data_format=data_format, pool_mode=pool_mode)


def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid',
    data_format=None, pool_mode='max'):
  """Keras.backend.pool3d

    3D Pooling.
  
    Args:
      x: Tensor
      pool_size: tuple of 3 integers.
      strides: tuple of 3 integers.
      padding: string, "same" or "valid".
      data_format: string, "channels_last" or "channels_first".
      pool_mode: string, "max" or "avg".
  """
  return K.pool3d(x, pool_size, strides=strides, padding=padding, 
      data_format=data_format, pool_mode=pool_mode)


def print_tensor(x, message=''):
  """Keras.backend.print_tensor

    Prints message and the tensor value when evaluated.
  
    Args:
      x: Tensor
      message: Message to print jointly with the tensor.
  """
  K.print_tensor(x, message=message)


def relu(x, alpha=0.0, max_value=None, threshold=0):
  """Keras.backend.relu

    Rectified linear unit.
  
    Args:
      x: Tensor
      alpha: A scalar, slope of negative section (default=0.).
      max_value: float. Saturation threshold.
      threshold: float. Threshold value for thresholded activation.

    Returns:
      With default values, it returns element-wise max(x, 0).
      Otherwise, it follows: 
        f(x) = max_value for x >= max_value, 
        f(x) = x for threshold <= x < max_value, 
        f(x) = alpha * (x - threshold) otherwise.
  """
  return K.relu(x, alpha=alpha, max_value=max_value, threshold=threshold)


def rnn(step_function, inputs, initial_states, go_backwards=False, mask=None,
    constants=None, unroll=False, input_length=None, time_major=False,
    zero_output_for_mask=False):
  """Keras.backend.rnn

    Iterates over the time dimension of a tensor.
  
    Args:
      step_function: RNN step function. Args; input; Tensor with shape 
          (samples, ...) (no time dimension), representing input for the 
          batch of samples at a certain time step. states; List of tensors. 
          Returns; output; Tensor with shape (samples, output_dim) 
          (no time dimension). new_states; List of tensors, same length and 
          shapes as 'states'. The first state in the list must be the 
          output tensor at the previous timestep.
      inputs: Tensor of temporal data of shape (samples, time, ...) 
          (at least 3D), or nested tensors, and each of which has shape 
          (samples, time, ...).
      initial_states: Tensor with shape (samples, state_size) 
          (no time dimension), containing the initial values for the states 
          used in the step function. In the case that state_size is in a 
          nested shape, the shape of initial_states will also follow the 
          nested structure.
      go_backwards: Boolean. If True, do the iteration over the time 
          dimension in reverse order and return the reversed sequence.
      mask: Binary tensor with shape (samples, time, 1), with a zero for 
          every element that is masked.
      constants: List of constant values passed at each step.
      unroll: Whether to unroll the RNN or to use a symbolic while_loop.
      input_length: An integer or a 1-D Tensor, depending on whether the 
          time dimension is fixed-length or not. In case of variable length 
          input, it is used for masking in case there's no mask specified.
      time_major: Boolean. If true, the inputs and outputs will be in shape 
          (timesteps, batch, ...), whereas in the False case, it will be 
          (batch, timesteps, ...). Using time_major = True is a bit more 
          efficient because it avoids transposes at the beginning and end of 
          the RNN calculation. However, most TensorFlow data is batch-major, 
          so by default this function accepts input and emits output in 
          batch-major form.
      zero_output_for_mask: Boolean. If True, the output for masked timestep 
          will be zeros, whereas in the False case, output from previous 
          timestep is returned.
  """
  return K.rnn(step_function, inputs, initial_states,
      go_backwards = go_backwards, mask = mask, constants = constants,
      unroll = unroll, input_length = input_length, time_major = time_major,
      zero_output_for_mask=zero_output_for_mask)


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
    padding='valid', data_format=None, dilation_rate=(1, 1)):
  """Keras.backend.separable_conv2d
    
    2D convolution with separable filters.

    Args:
      x: Tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      pointwise_kernel: kernel for the 1x1 convolution.
      strides: strides tuple.
      padding: string, "same" or "valid".
      data_format: string, one of "channels_last", "channels_first".
      dilation_rate: tuple of 2 integers.
  """
  return K.separable_conv2d(x, depthwise_kernel, pointwise_kernel,
      strides=strides, padding=padding, data_format=data_format,
      dilation_rate=dilation_rate)


def set_value(x, value):
  """Keras.backend.set_value

    Sets the value of a variable, from a Numpy array.
  
    Args:
      x: Tensor
      value: Numpy array.
  """
  K.set_value(x, value)


def sigmoid(x):
  """Keras.backend.sigmoid

    Element-wise sigmoid.
  
    Args:
      x: Tensor
  """
  return K.sigmoid(x)


def sparse_categorical_crossentropy(target, output, from_logits=False,
    axis=-1):
  """Keras.backend.sparse_categorical_crossentropy

    Softsign of a tensor.
  
    Args:
      target: An integer tensor.
      output: A tensor resulting from a softmax (unless from_logits is True, 
          in which case output is expected to be the logits).
      from_logits: Boolean, whether output is the result of a softmax, 
          or is a tensor of logits.
      axis: Int specifying the channels axis. 
          axis=-1 corresponds to data format channels_last, 
          axis=1 corresponds to data format channels_first.
  """
  return K.sparse_categorical_crossentropy(target, output,
      from_logits=from_logits, axis=axis)


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
  """Keras.backend.spatial_2d_padding

    Softsign of a tensor.
  
    Args:
      x: Tensor
      padding: Tuple of 2 tuples, padding pattern.
      data_format: One of channels_last or channels_first.
  """
  return K.spatial_2d_padding(x, padding=padding, data_format=data_format)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
  """Keras.backend.spatial_3d_padding

    Softsign of a tensor.
  
    Args:
      x: Tensor
      padding: Tuple of 3 tuples, padding pattern.
      data_format: One of channels_last or channels_first.
  """
  return K.spatial_3d_padding(x, padding=padding, data_format=data_format)


def stop_gradient(variables):
  """Keras.backend.stop_gradient

    Returns variables but with zero gradient w.r.t. every other variable.
  
    Args:
      variables: Tensor or list of tensors to consider constant with 
          respect to any other variable.
  """
  return K.stop_gradient(variables)


def switch(condition, then_expression, else_expression):
  """Keras.backend.switch

    Switches between two operations depending on a scalar value.
  
    Args:
      condition: tensor (int or bool).
      then_expression: either a tensor, or a callable that returns a tensor.
      else_expression: either a tensor, or a callable that returns a tensor.
  """
  return K.switch(condition, then_expression, else_expression)


def temporal_padding(x, padding=(1, 1)):
  """Keras.backend.temporal_padding

    Pads the middle dimension of a 3D tensor.
  
    Args:
      x: Tensor
      padding: Tuple of 2 integers, how many zeros to add at the start and 
          end of dim 1.
  """
  return K.temporal_padding(x, padding=padding)


def to_dense(tensor):
  """Keras.backend.to_dense

    Converts a sparse tensor into a dense tensor and returns it.
  
    Args:
      tensor: Tensor
  """
  return K.to_dense(tensor)


def update(x, new_x):
  """Keras.backend.update

    Update the value of x by new_x.
  
    Args:
      x: Tensor
      new_x: A tensor of same shape as x.
  """
  return K.update(x, new_x)


def update_add(x, increment):
  """Keras.backend.update_add

    Update the value of x by adding increment.
  
    Args:
      x: Tensor
      increment: A tensor of same shape as x.
  """
  return K.update_add(x, increment)


def update_sub(x, decrement):
  """Keras.backend.update_sub

    Update the value of x by subtracting decrement.
  
    Args:
      x: Tensor
      decrement: A tensor of same shape as x.
  """
  return K.update_sub(x, decrement)


def variable(value, dtype=None, name=None, constraint=None):
  """Keras.backend.variable

    Instantiates a variable and returns it.
  
    Args:
      value: Numpy array, initial value of the tensor.
      dtype: Tensor type.
      name: Optional name string for the tensor.
      constraint: Optional projection function to be applied to the 
          variable after an optimizer update.
  """
  return K.variable(value, dtype=dtype, name=name, constraint=constraint)

