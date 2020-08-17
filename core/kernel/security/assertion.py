# -*- coding: utf-8 -*-
"""Assertion
  ======

  Deep Go Assertions
"""


# from abc import ABCMeta, abstractmethod

from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod
from deepgo.core.kernel.loader import lazy_loader
from deepgo.core.kernel.security import exception


class Assertion(AbstractSingleton):
  """Parent Class for a Assertion Class

    This is a Abstract Singleton Class
  """
  def __call__(self, *args, **kwargs):
    self.call(*args, **kwargs)
    
  @abstractmethod
  def call(self, *args, **kwargs): ...


# Sub-Classes


class VOAssertion(Assertion):
  """Ensure Object is a VO
  
    This is a Singleton Class
  """
  def call(self, obj):
    VO = lazy_loader.lazy_import("deepgo.core.kernel.popo").VO
    if not isinstance(obj, VO):
      raise exception.TypeError(f"Expected VO, got {obj.__class__.__name__}")
    return True


if __name__ == "__main__":
  from deepgo.core.kernel.popo import VO
  try:
    v = VO()
    VOAssertion()(v)
    VOAssertion()(1)
  except Exception as e:
    print(e)

