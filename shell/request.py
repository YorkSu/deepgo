# -*- coding: utf-8 -*-
"""Request
  ======

  Deep Go Requests
"""


import abc

from deepgo.core.kernel.popo import VO
from deepgo.core.kernel.security.permission import GETPermission, POSTPermission


class Response(VO):
  def __init__(self):
    self.code = 0

  def json(self):
    return self.__dict__


# class Response(object):
#   def __init__(self, code=0, message=''):
#     self.set_code(code)
#     self.set_message(message)

#   def set_code(self, code):
#     self.code = code

#   def set_message(self, message):
#     self.message = message


class Request(object):
  def __init__(self):
    self._permission = None

  @abc.abstractmethod
  def get(self): ...
    
  @abc.abstractmethod
  def post(self): ...

  @property
  @abc.abstractmethod
  def method(self) -> str: ...


class VersionRequest(Request):
  def get(self, *args):
    # print("Get Permission")  # DEBUG
    # self._permission = GETPermission()
    # print("TODO")  # DEBUG
    from deepgo import __version__, __codename__, __release_date__
    count = args[0]
    response = Response()
    if count == 1:
      response.message = f"Deep Go {__version__}"
      # response.set_message(f"Deep Go {__version__}")
    else:
      response.message = f"Deep Go {__version__} [{__codename__} {__release_date__}]"
      # response.set_message(f"Deep Go {__version__} [{__codename__} {__release_date__}]")
    return response

  def post(self): ...

  @property
  def method(self):
    return "GET"


