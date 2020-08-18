# -*- coding: utf-8 -*-
"""Response
  ======

  Response Class
"""


from deepgo.core.kernel.popo import VO


class Response(VO):
  def __init__(self):
    self.code = 0

  def json(self):
    return self.__dict__

