# -*- coding: utf-8 -*-
"""Lazy Loader
  ======

  Lazy Loader Class
"""


import importlib

from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod
from deepgo.core.kernel.logger import logger


class AbstractLazyLoader(AbstractSingleton):
  """Abstract Lazy Loader

    This is a Abstract Singleton Class
  """
  @abstractmethod
  def lazy_import(self, name): ...


class LazyLoader(AbstractLazyLoader):
  """Lazy Loader

    This is a Singleton Class
  """
  def __init__(self):
    self.__modules = {}

  def __import(self, name):
    """Thread-Safe Import Method"""
    if name not in self.__modules:
      with self._lock:
        if name not in self.__modules:
          module = None
          e = None
          try:
            module = importlib.import_module(name)
          except ImportError as _e:
            e = _e
          self.__modules[name] = {
              'module': module,
              'e': e}
    return self.__modules[name]

  def lazy_import(self, name):
    """Lazy Import Method"""
    _module = self.__import(name)
    if _module['e']:
      logger.error(_module['e'], name="LazyLoader")
    return _module['module']

  @property
  def tensorflow(self):
    """Lazy Import TensorFlow"""
    _tensorflow = self.__import("tensorflow")
    if _tensorflow['module'] is None:
      logger.error(_tensorflow['e'], name="LazyLoader")
    elif _tensorflow['module'].__version__ != "2.1.0":
      logger.warning(
          f"Expected tensorflow 2.1.0, "
          f"got {_tensorflow['module'].__version__}, "
          f"may cause problems!", name="LazyLoader")
    return _tensorflow['module']
  
  tf = tensorflow


lazy_loader = LazyLoader()


if __name__ == "__main__":
  tf = lazy_loader.tf

