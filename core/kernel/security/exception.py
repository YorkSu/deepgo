# -*- coding: utf-8 -*-
"""Exception
  ======

  Deep Go Exceptions
"""


# Error Codes
UNKNOWN = 0
TYPE = 1


CODE2TYPE = {
    UNKNOWN: 'UnknownError',
    TYPE: 'TypeError',}


class BaseError(Exception): pass


class Error(BaseError):
  """Parent Class for a Exception"""
  def __init__(self, msg="", code=UNKNOWN, etype=None):
    super(Error, self).__init__()
    self.msg = msg
    self.code = code
    if etype is None:
      etype = CODE2TYPE[code]
    self.etype = etype
    
  def __str__(self):
    e = f"CODE: {self.code}, {self.etype}"
    if self.msg:
      e += f", {self.msg}"
    return e


# Sub-Classes


class UnknownError(Error):
  """UnknownError"""
  def __init__(self, msg=""):
    super(UnknownError, self).__init__(msg, UNKNOWN)


class TypeError(Error):
  """TypeError"""
  def __init__(self, msg=""):
    super(TypeError, self).__init__(msg, TYPE)


if __name__ == "__main__":
  try:
    raise UnknownError#("test")
    # raise error_factory.get_error(1, "TypeError")
  except Exception as e:
    print(e)
    import logging
    logging.exception('debug')


