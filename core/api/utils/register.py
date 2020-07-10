# -*- coding: utf-8 -*-
"""Register API
  ======

  Custom API, containing methods for handling register
"""


from tensorflow.keras.utils import get_custom_objects


_CUSTOM_DATASET = {}


def get_dataset(name):
  """api.utils.get_dataset

    Get the dataset class or define function from name

    Args:
      name: str, name of the dataset.

    Returns:
      DeepGo.Dataset or Function or None
  """
  return _CUSTOM_DATASET.get(name)


def register(name, clas):
  """api.utils.register

    Register a custom layer as a Keras object.

    Args:
      name: str, name of the layer
      clas: the custom layer class
  """
  get_custom_objects().update({name: clas})


def register_dataset(name, clas):
  """api.utils.register_dataset

    Register a custom dataset

    Args:
      name: str, name of the layer
      clas: the custom layer class
  """
  global _CUSTOM_DATASET
  _CUSTOM_DATASET.update({name: clas})

