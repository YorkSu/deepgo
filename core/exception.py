# -*- coding: utf-8 -*-
"""Exception
  ======

  Deep Go Exceptions and Assertions
"""


import h5py

from deepgo.core.api import np
from deepgo.core.layer import Model


# ================================
# ERROR CODE
# ================================
UNKNOWN = 0
INIT = 1
TYPE = 2
INDEX = 3


class CoreError(Exception):
  """CoreError
  
    内核异常类的父类
    
    Args:
      message: Str. 描述错误的消息
      code: int. 错误代码
      name: Str, default __name__. 可指定名称
        传入`__name__`可用于定位日志的发送位置
  """
  def __init__(self, message: str, code: int, name=__name__):
    super().__init__()
    self._code = code
    self._message = message
    self._name = name

  @property
  def code(self):
    """The integer error code that describes the error."""
    return self._code

  @property
  def message(self):
    """The error message that describes the error."""
    return self._message

  @property
  def name(self):
    """Where the error occurred."""
    return self._name

  def __str__(self):
    return f"{self.__class__.__name__}(CODE: {self._code}, NAME: {self._name}, MSG: {self._message})"


# ================================
# Subclass
# ================================


class UnknownError(CoreError):
  """UnknownError

    未知错误

    Args:
      message: Str. 描述错误的消息
      name: Str, default __name__. 可指定名称
        传入`__name__`可用于定位日志的发送位置
  """
  def __init__(self, message="Unknown", name=__name__):
    super().__init__(message, UNKNOWN, name)


class InitError(CoreError):
  """InitError
  
    初始化错误
    
    Args:
      message: Str. 描述错误的消息
      name: Str, default __name__. 可指定名称
        传入`__name__`可用于定位日志的发送位置
  """
  def __init__(self, message='Not initialized', name=__name__):
    super().__init__(message, INIT, name)


class TypeError(CoreError):
  """TypeError

    类型错误

    Args:
      message: Str. 描述错误的消息
      name: Str, default __name__. 可指定名称
        传入`__name__`可用于定位日志的发送位置
  """
  def __init__(self, message='Type incorrect', name=__name__):
    super().__init__(message, TYPE, name)


class IndexError(CoreError):
  """IndexError

    索引错误

    Args:
      message: Str. 描述错误的消息
      name: Str, default __name__. 可指定名称
        传入`__name__`可用于定位日志的发送位置
  """
  def __init__(self, message='Index out of range', name=__name__):
    super().__init__(message, INDEX, name)


# ================================
# Assertions
# ================================


def assert_h5_dataset(obj):
  """Ensure Object is a h5py.Dataset

    Raises:
      DeepGo.TypeError
  """
  if not isinstance(obj, h5py.Dataset):
    raise TypeError(", ".join([
        f"Expected h5py.Dataset",
        f"got {obj.__class__.__name__}"]))


def assert_h5_group(obj):
  """Ensure Object is a h5py.File or h5py.Group

    Raises:
      DeepGo.TypeError
  """
  if not isinstance(obj, (h5py.File, h5py.Group)):
    raise TypeError(", ".join([
        f"Expected h5py.File or h5py.Group",
        f"got {obj.__class__.__name__}"]))


def assert_model(obj):
  """Ensure Object is a keras.Model

    Raises:
      DeepGo.TypeError
  """
  if not isinstance(obj, Model):
    raise TypeError(", ".join([
        f"Expected Model",
        f"got {obj.__class__.__name__}"]))


def assert_np(obj):
  """Ensure Object is a np.ndarray

    Raises:
      DeepGo.TypeError
  """
  if not isinstance(obj, np.ndarray):
    raise TypeError(", ".join([
        f"Expected ndarray",
        f"got {obj.__class__.__name__}"]))


if __name__ == "__main__":
  try:
    raise UnknownError
  except Exception as e:
    print(e)
    print(repr(e))
  try:
    AssertModel()('123')
  except Exception as e:
    print(e)
    print(repr(e))

