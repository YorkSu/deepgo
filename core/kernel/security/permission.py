# -*- coding: utf-8 -*-
"""Permission
  ======

  Deep Go Permissions
"""


import abc
# from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod


class Permission(abc.ABC):
  """Parent Class for a Permission Class"""
  @abc.abstractmethod
  def verify(self): ...


class GETPermission(Permission):
  """GET Permission

    Permission to read data
  """
  def verify(self):
    return "GET"


class POSTPermission(Permission):
  """POST Permission

    Permission to read and write data
  """
  def verify(self):
    return "POST"


