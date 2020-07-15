# -*- coding: utf-8 -*-
"""Config API
  ======

  Custom API, containing methods for handling Config
"""


__all__ = [
    'set_gpu_memory_growth',]


from deepgo.core.api.tf.config import get_visible_devices
from deepgo.core.api.tf.config import set_memory_growth


def set_gpu_memory_growth(value=True):
  """set gpu memory growth
  
    Set if memory growth should be enabled for GPU.
  """
  devides = get_visible_devices()
  for devide in devides:
    if 'GPU' in devide.name:
      set_memory_growth(devide, value)

