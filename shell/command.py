# -*- coding: utf-8 -*-
"""Command
  ======

  Deep Go Commands
"""


import abc
from deepgo import __version__, __codename__, __release_date__
from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod


class Request(abc.ABC):
  """Title

    Description
  """
  @abc.abstractmethod
  def get(self): ...

  @abc.abstractmethod
  def post(self): ...


class Command(abc.ABC):
  """Title

    Description
  """
  @abc.abstractmethod
  def execute(self): ...


class AbstractCommand(Command):
  """Title

    Description
  """
  def __init__(self):
    ...







