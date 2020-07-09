# -*- coding: utf-8 -*-
"""TensorFlow Config API
  ======

  TensorFlow API, containing mathods for handling Config
"""


import tensorflow as tf


def executing_eagerly():
  """TensorFlow.executing_eagerly
  
    Checks whether the current thread has eager execution enabled.
  """
  return tf.executing_eagerly()


def get_run_eagerly():
  """TensorFlow.config.experimental_functions_run_eagerly
  
    Returns the value of the experimental_run_functions_eagerly setting.
  """
  return tf.config.experimental_functions_run_eagerly()


def get_visible_devices(device_type=None):
  """TensorFlow.config.get_visible_devices
  
    Get the list of visible physical devices.

    Args:
      device_type: (optional string) Only include devices matching 
          this device type. For example "CPU" or "GPU".
  """
  return tf.config.get_visible_devices(device_type)


def set_run_eagerly(run_eagerly):
  """TensorFlow.config.experimental_run_functions_eagerly
  
    Enables / disables eager execution of tf.functions.

    Args:
      run_eagerly: Boolean. Whether to run functions eagerly.
  """
  tf.config.experimental_run_functions_eagerly(run_eagerly)


def set_visible_devices(devices, device_type=None):
  """TensorFlow.config.set_visible_devices
  
    Set the list of visible devices.

    Args:
      devices: List of PhysicalDevices to make visible
      device_type: (optional string) Only include devices matching 
          this device type. For example "CPU" or "GPU".
  """
  tf.config.set_visible_devices(devices, device_type)


# ================
# Experimental
# ================


def get_device_policy():
  """TensorFlow.config.experimental.get_device_policy
  
    Gets the current device policy.
  """
  return tf.config.experimental.get_device_policy()


def get_memory_growth(device):
  """TensorFlow.config.experimental.get_memory_growth
  
    Get if memory growth is enabled for a PhysicalDevice.

    Args:
      device: PhysicalDevice to query
  """
  return tf.config.experimental.get_memory_growth(device)


def get_synchronous_execution():
  """TensorFlow.config.experimental.get_synchronous_execution
  
    Gets whether operations are executed synchronously or asynchronously.
  """
  return tf.config.experimental.get_synchronous_execution()


def set_device_policy(device_policy):
  """TensorFlow.config.experimental.set_device_policy
  
    Sets the current thread device policy.

    Args:
      device_policy: A device policy. Valid values:
          None: Switch to a system default.
          'warn': Copies the tensors which are not on the right device and 
              logs a warning.
          'explicit': Raises an error if the placement is not as required.
          'silent': Silently copies the tensors. Note that this may hide 
              performance problems as there is no notification provided when 
              operations are blocked on the tensor being copied between devices.
          'silent_for_int32': silently copies int32 tensors, raising errors 
              on the other ones.
  """
  tf.config.experimental.set_device_policy(device_policy)


def set_memory_growth(device, enable):
  """TensorFlow.config.experimental.set_memory_growth
  
    Set if memory growth should be enabled for a PhysicalDevice.

    Args:
      device: PhysicalDevice to query
      enable: (Boolean) Whether to enable or disable memory growth
  """
  tf.config.experimental.set_memory_growth(device, enable)


def set_synchronous_execution(enable):
  """TensorFlow.config.experimental.set_synchronous_execution
  
    Gets whether operations are executed synchronously or asynchronously.

    Args:
      enable: Whether operations should be dispatched synchronously.
          None: sets the system default.
          True: executes each operation synchronously.
          False: executes each operation asynchronously.
  """
  tf.config.experimental.set_synchronous_execution(enable)


# ================
# Optimizer
# ================


def get_experimental_options():
  """TensorFlow.config.optimizer.get_experimental_options
  
    Get experimental optimizer options.
  """
  return tf.config.optimizer.get_experimental_options()


def get_jit():
  """TensorFlow.config.optimizer.get_jit
  
    Get if JIT compilation is enabled.
  """
  return tf.config.optimizer.get_jit()


def set_experimental_options(options):
  """TensorFlow.config.optimizer.set_experimental_options
  
    Set experimental optimizer options.

    Args:
      options: Dictionary of experimental optimizer options to configure. 
          SEE: https://tensorflow.google.cn/versions/r2.1/api_docs/python/tf/config/optimizer/set_experimental_options
  """
  tf.config.optimizer.set_experimental_options(options)


def get_jit(enabled):
  """TensorFlow.config.optimizer.get_jit
  
    Set if JIT compilation is enabled.

    Args:
      enabled: Whether to enable JIT compilation.
  """
  tf.config.optimizer.get_jit(enabled)

