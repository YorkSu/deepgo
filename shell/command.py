# -*- coding: utf-8 -*-
"""Command
  ======

  Deep Go Commands
"""


__all__ = [
    'Command',
    'VersionCommand',
    'ExitCommand',]


import abc

from deepgo.shell.response import Response


class Command(abc.ABC):
  """Command Interface"""
  @abc.abstractmethod
  def execute(self, *args, **kwargs): ...


class VersionCommand(Command):
  """Version Command
  
    Provide version information
  """
  def execute(self, *args, **kwargs):
    from deepgo import __version__, __codename__, __release_date__
    response = Response()
    
    count = 1
    for arg in args:
      if arg in ['more']:
        count = 2
    if 'count' in kwargs:
      count = int(kwargs.pop('count'))
    
    if count < 1:
      response.message = f"Version: Invalid count: {count}"
    elif count == 1:
      response.message = f"Deep Go {__version__}"
    else:
      response.message = f"Deep Go {__version__} [{__codename__} {__release_date__}]"
    return response


class ExitCommand(Command):
  """Exit Command"""
  def execute(self, *args, **kwargs):
    import os
    os._exit(0)

