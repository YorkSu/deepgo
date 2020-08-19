# -*- coding: utf-8 -*-
"""Command
  ======

  Deep Go Commands
"""


__all__ = [
    'Command',
    'VersionCommand',
    'ExitCommand',
    'ProjectCommand',]


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


class ProjectCommand(Command):
  """Project Command
  
    Help to Generate a project for Deep Learning
  """
  def execute(self, *args, **kwargs):
    from deepgo.core.kernel.path import path as _path
    response = Response()
    if not args and not kwargs:
      response.message = f"Expected path argument"
      return response
    path = ''
    if args:
      path = args[0]
    if 'path' in kwargs:
      path = kwargs['path']
    path = _path.abs(path)
    response.message = f"Generated Project: {path}"
    return response

