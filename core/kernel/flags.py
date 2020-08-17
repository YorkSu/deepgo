# -*- coding: utf-8 -*-
"""Flags
  ======

  Global Arguments Manager
"""


from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod


class AbstractFlags(AbstractSingleton):
  """Abstract Flags Class

    This is a Abstract Singleton Class
  """
  @abstractmethod
  def get(self, key): ...

  @abstractmethod
  def set(self, key, value): ...


class Flags(AbstractFlags):
  """Global Arguments Manager

    This is a Singleton Class
  """
  def __getattribute__(self, key, default=None):
    try:
      return super(Flags, self).__getattribute__(key)
    except AttributeError:
      return default

  def get(self, key, default=None):
    """Get the value of an argument

      If not found, return default
    """
    return self.__getattribute__(key, default)

  def set(self, key, value):
    """Set the value of an argument"""
    self.__setattr__(key, value)


flags = Flags()

