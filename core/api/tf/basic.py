# -*- coding: utf-8 -*-
"""TensorFlow Basic API
  =====

  TensorFlow API, containing basic methods
"""


import tensorflow as tf


def cond(pred, true_fn=None, false_fn=None, name=None):
  """TensorFlow.cond

    Return true_fn() if the predicate pred is true else false_fn().

    Args:
      pred: A scalar determining whether to return the result of 
          true_fn or false_fn.
      true_fn: The callable to be performed if pred is true.
      false_fn: The callable to be performed if pred is false.
      name: A name for this operation (optional).
  """
  return tf.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)


def device(device_name):
  """TensorFlow.device

    Specifies the device for ops created/executed in this context.

    Args:
      device_name: The device name to use in the context.
  """
  return tf.device(device_name)


def Graph(device_name):
  """TensorFlow.Graph

    A TensorFlow computation, represented as a dataflow graph.

    Returns:
      tf.Graph
  """
  return tf.Graph()


def no_op(name=None):
  """TensorFlow.no_op

    Does nothing. Only useful as a placeholder for control edges.

    Args:
      name: A name for this operation (optional).
  """
  return tf.no_op(name=name)


def numpy_function(func, inp, Tout, name=None):
  """TensorFlow.numpy_function

    Wraps a python function and uses it as a TensorFlow op.
    SEE: https://tensorflow.google.cn/versions/r2.1/api_docs/python/tf/numpy_function

    Args:
      func: A Python function, which accepts numpy.ndarray objects as 
          arguments and returns a list of numpy.ndarray objects 
          (or a single numpy.ndarray). 
      inp: A list of tf.Tensor objects.
      Tout: A list or tuple of tensorflow data types or a single tensorflow 
          data type if there is only one, indicating what func returns.
      name: A name for this operation (optional).
  """
  return tf.numpy_function(func, inp, Tout, name=name)


def print(*inputs, **kwargs):
  """TensorFlow.print

    Print the specified inputs.

    Args:
      *inputs: Positional arguments that are the inputs to print.
      output_stream: The output stream, logging level, or file to print to.
          Defaults to sys.stderr, but sys.stdout, tf.logging
      summarize: The first and last summarize elements within each 
          dimension are recursively printed per Tensor.
      sep: The string to use to separate the inputs. Defaults to " ".
      end: End character that is appended at the end the printed string. 
          Defaults to the newline character.
      name: A name for this operation (optional).
  """
  return tf.print(*inputs, **kwargs)


def py_function(func, inp, Tout, name=None):
  """TensorFlow.py_function

    Wraps a python function into a TensorFlow op that executes it eagerly.
    SEE: https://tensorflow.google.cn/versions/r2.1/api_docs/python/tf/py_function

    Args:
      func: A Python function which accepts a list of Tensor objects having 
          element types that match the corresponding tf.Tensor objects in 
          inp and returns a list of Tensor objects (or a single Tensor, or 
          None) having element types that match the corresponding values in 
          Tout.
      inp: A list of tf.Tensor objects.
      Tout: A list or tuple of tensorflow data types or a single tensorflow 
          data type if there is only one, indicating what func returns; an 
          empty list if no value is returned (i.e., if the return value is 
          None).
      name: A name for this operation (optional).
  """
  return tf.py_function(func, inp, Tout, name=name)


def tf_function(func=None, input_signature=None, autograph=True,
    experimental_implements=None, experimental_autograph_options=None,
    experimental_relax_shapes=False, experimental_compile=None):
  """TensorFlow.function

    Compiles a function into a callable TensorFlow graph.
    SEE: https://tensorflow.google.cn/versions/r2.1/api_docs/python/tf/function

    Args:
      func: the function to be compiled.
  """
  return tf.function(
      func=func, 
      input_signature=input_signature, 
      autograph=autograph, 
      experimental_implements=experimental_implements, 
      experimental_autograph_options=experimental_autograph_options,
      experimental_relax_shapes=experimental_relax_shapes, 
      experimental_compile=experimental_compile)


def while_loop(cond, body, loop_vars, shape_invariants=None,
    parallel_iterations=10, back_prop=True, swap_memory=False,
    maximum_iterations=None, name=None):
  """TensorFlow.while_loop

    Repeat body while the condition cond is true.
    SEE: https://tensorflow.google.cn/versions/r2.1/api_docs/python/tf/while_loop

    Args:
      cond: A callable that represents the termination condition of the loop.
      body: A callable that represents the loop body.
      loop_vars: A (possibly nested) tuple, namedtuple or list of numpy 
          array, Tensor, and TensorArray objects.
      shape_invariants: The shape invariants for the loop variables.
      name: A name for this operation (optional).
  """
  return tf.while_loop(cond, body, loop_vars,
      shape_invariants=shape_invariants,
      parallel_iterations=parallel_iterations,
      back_prop=back_prop, 
      swap_memory=swap_memory,
      maximum_iterations=maximum_iterations, 
      name=name)

