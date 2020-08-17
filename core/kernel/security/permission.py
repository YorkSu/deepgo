# -*- coding: utf-8 -*-
"""Permission
  ======

  Deep Go Permissions
"""


from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod


class Permission(AbstractSingleton):
  """Parent Class for a Permission Class

    This is a Abstract Singleton Class
  """
  @abstractmethod
  def verify(self): ...

