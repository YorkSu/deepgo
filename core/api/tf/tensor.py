# -*- coding: utf-8 -*-
"""TensorFlow Tensor API
  ======

  TensorFlow API, containing methods for handling Tensor
"""


import tensorflow as tf


def case(pred_fn_pairs, default=None, exclusive=False, strict=False,
    name=None):
  """TensorFlow.case

    Create a case operation.

    Args:
      pred_fn_pairs: List of pairs of a boolean scalar tensor and a callable 
          which returns a list of tensors.
      default: Optional callable that returns a list of tensors.
      exclusive: True iff at most one predicate is allowed to evaluate to 
          True.
      strict: A boolean that enables/disables 'strict' mode; see above.
      name: A name for this operation (optional).
  """
  return tf.case(pred_fn_pairs, default=default, exclusive=exclusive, 
      strict=strict, name=name)


def concat(values, axis, name=None):
  """TensorFlow.concat

    Concatenates tensors along one dimension.

    Args:
      values: A list of Tensor objects or a single Tensor.
      axis: 0-D int32 Tensor. Dimension along which to concatenate. 
          Must be in the range [-rank(values), rank(values)). As in Python, 
          indexing for axis is 0-based. Positive axis in the rage of 
          [0, rank(values)) refers to axis-th dimension. And negative axis 
          refers to axis + rank(values)-th dimension.
      name: A name for this operation (optional).
  """
  return tf.concat(values, axis, name=name)


def eig(tensor, name=None):
  """TensorFlow.eig

    Computes the eigen decomposition of a batch of matrices.

    Args:
      tensor: Tensor of shape [..., N, N]. Only the lower triangular part 
          of each inner inner matrix is referenced.
      name: A name for this operation (optional).
  """
  return tf.eig(tensor, name=name)


def eigvals(tensor, name=None):
  """TensorFlow.eigvals

    Computes the eigenvalues of one or more matrices.

    Args:
      tensor: Tensor of shape [..., N, N].
      name: A name for this operation (optional).
  """
  return tf.eigvals(tensor, name=name)


def ensure_shape(x, shape, name=None):
  """TensorFlow.ensure_shape

    Updates the shape of a tensor and checks at runtime that the shape holds.

    Args:
      x: Tensor
      shape: A TensorShape representing the shape of this tensor, 
          a TensorShapeProto, a list, a tuple, or None.
      name: A name for this operation (optional).
  """
  return tf.ensure_shape(x, shape, name=name)


def identity(input, name=None):
  """TensorFlow.identity

    Return a tensor with the same shape and contents as input.

    Args:
      input: Tensor
      name: A name for this operation (optional).
  """
  return tf.identity(input, name=name)


def pad(tensor, paddings, mode='CONSTANT', constant_values=0, name=None):
  """TensorFlow.pad

    Pads a tensor.

    Args:
      tensor: Tensor
      paddings: A Tensor of type int32.
      mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
      constant_values: In "CONSTANT" mode, the scalar pad value to use. 
          Must be same type as tensor.
      name: A name for this operation (optional).
  """
  return tf.pad(tensor, paddings, mode=mode, 
      constant_values=constant_values, name=name)


def range(*args, **kwargs):
  """TensorFlow.range

    Creates a sequence of numbers.

    Args:
      The same as python.range
  """
  return tf.range(*args, **kwargs)


def rank(input, name=None):
  """TensorFlow.rank

    Returns the rank of a tensor.

    Args:
      input: Tensor
      name: A name for this operation (optional).
  """
  return tf.rank(input, name=name)


def slice(input_, begin, size, name=None):
  """TensorFlow.slice

    Extracts a slice from a tensor.

    Args:
      input_: Tensor
      begin: An int32 or int64 Tensor.
      size: An int32 or int64 Tensor.
      name: A name for this operation (optional).
  """
  return tf.slice(input_, begin, size, name=name)


def sort(values, axis=-1, direction='ASCENDING', name=None):
  """TensorFlow.sort

    Sorts a tensor.

    Args:
      values: Tensor
      axis: The axis along which to sort. The default is -1, 
          which sorts the last axis.
      direction: The direction in which to sort the values 
          ('ASCENDING' or 'DESCENDING').
      name: A name for this operation (optional).
  """
  return tf.sort(values, axis=axis, direction=direction, name=name)


def split(value, num_or_size_splits, axis=0, num=None, name=None):
  """TensorFlow.split

    Splits a tensor into sub tensors.

    Args:
      value: Tensor
      num_or_size_splits: Either an integer indicating the number of splits 
          along axis or a 1-D integer Tensor or Python list containing the 
          sizes of each output tensor along axis. If a scalar, then it must 
          evenly divide value.shape[axis]; otherwise the sum of sizes along 
          the split axis must match that of the value.
      axis: An integer or scalar int32 Tensor. The dimension along which to 
          split. Must be in the range [-rank(value), rank(value)). Defaults 
          to 0.
      num: Optional, used to specify the number of outputs when it cannot 
          be inferred from the shape of size_splits.
      name: A name for this operation (optional).
  """
  return tf.split(value, num_or_size_splits, axis=axis, num=num, name=name)


def switch_case(branch_index, branch_fns, default=None, name=None):
  """TensorFlow.switch_case

    Create a switch/case operation, i.e. an integer-indexed conditional.

    Args:
      branch_index: An int Tensor specifying which of branch_fns should be 
          executed.
      branch_fns: A dict mapping ints to callables, or a list of (int, 
          callable) pairs, or simply a list of callables (in which case 
          the index serves as the key). Each callable must return a 
          matching structure of tensors.
      default: Optional callable that returns a structure of tensors.
      name: A name for this operation (optional).
  """
  return tf.switch_case(branch_index, branch_fns, default=default, name=name)


def unique(x, out_idx=tf.dtypes.int32, name=None):
  """TensorFlow.unique

    Finds unique elements in a 1-D tensor.

    Args:
      x: Tensor
      out_idx: An optional tf.DType from: tf.int32, tf.int64. 
          Defaults to tf.int32.
      name: A name for this operation (optional).
  
    Returns:
      Tuple of Tensor:
      * y: A Tensor contains unique value
      * idx: A Tensor of type out_idx.
  """
  return tf.unique(x, out_idx=out_idx, name=name)


def unstack(value, num=None, axis=0, name=None):
  """TensorFlow.unstack

    Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

    Args:
      value: Tensor
      num: An int. The length of the dimension axis. Automatically inferred 
          if None (the default).
      axis: An int. The axis to unstack along. Defaults to the first 
          dimension. Negative values wrap around, so the valid range is 
          [-R, R).
      name: A name for this operation (optional).
  """
  return tf.unstack(value, num=num, axis=axis, name=name)


def where(condition, x=None, y=None, name=None):
  """TensorFlow.where

    Return the elements, either from x or y, depending on the condition.

    Args:
      condition: Tensor of type bool
      x: A Tensor which is of the same type as y, and may be broadcastable 
          with condition and y.
      y: A Tensor which is of the same type as x, and may be broadcastable 
          with condition and x.
      name: A name for this operation (optional).
  """
  return tf.where(condition, x=x, y=y, name=name)

