# -*- coding: utf-8 -*-
"""Keras Config API
  =====

  Keras API, containing mathods for handling Config
"""


from tensorflow.keras import backend as K


def clear_session():
  """Keras.backend.clear_session
    
    Destroys the current TF graph and session, and creates a new one.
  """
  return K.clear_session()


def get_backend():
  """Keras.backend.backend
    
    Publicly accessible method for determining the current backend.

    Returns:
      The string "tensorflow".
  """
  return K.backend()


def get_data_format():
  """Keras.backend.image_data_format

    Returns the default image data format convention.

    Returns:
      A string, either 'channels_first' or 'channels_last'
  """
  return K.image_data_format()


def get_epsilon():
  """Keras.backend.epsilon

    Returns the value of the fuzz factor used in numeric expressions.
  """
  return K.epsilon()


def get_floatx():
  """Keras.backend.floatx

    Returns the default float type, as a string.
  """
  return K.floatx()


def get_learning_phase():
  """Keras.backend.learning_phase

    Returns the learning phase flag.
  """
  return K.learning_phase()


def get_uid(prefix=''):
  """Keras.backend.get_uid

    Associates a string prefix with an integer counter in a TensorFlow graph.
  
    Args:
      prefix: String prefix to index.
  """
  return K.get_uid(prefix=prefix)


def reset_uids():
  """Keras.backend.reset_uids

    Resets graph identifiers.
  """
  K.reset_uids()


def set_data_format(data_format):
  """Keras.backend.set_image_data_format

    Sets the value of the image data format convention.
  
    Args:
      data_format: string. 'channels_first' or 'channels_last'.
  """
  K.set_image_data_format(data_format)


def set_epsilon(value):
  """Keras.backend.set_epsilon

    Sets the value of the fuzz factor used in numeric expressions.
  
    Args:
      value: float. New value of epsilon.
  """
  K.set_epsilon(value)


def set_floatx(value):
  """Keras.backend.set_floatx

    Sets the default float type.
  
    Args:
      value: String; 'float16', 'float32', or 'float64'.
    
    Note:
      Note: It is not recommended to set this to float16 for training, 
      as this will likely cause numeric stability issues. Instead, 
      mixed precision, which is using a mix of float16 and float32, 
      can be used by calling 
      `tf.keras.mixed_precision.experimental.set_policy('mixed_float16')`. 
      See the mixed precision guide for details.
  """
  K.set_floatx(value)


def set_learning_phase(value):
  """Keras.backend.set_learning_phase

    Sets the learning phase to a fixed value.
  
    Args:
      value: Learning phase value, either 0 or 1 (integers). 
          0 = test, 1 = train
  """
  K.set_learning_phase(value)

