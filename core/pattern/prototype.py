# -*- coding: utf-8 -*-
"""Prototype
  ======

  Prototype Patterns
"""


import abc
import copy


class CloneNotSupportedException(Exception): ...


class Cloneable(abc.ABC):
  """Cloneable Interface

    This is an Abstract Class
  """
  def clone(self):
    raise CloneNotSupportedException


class Prototype(Cloneable):
  """Title

    Description
  """
  def clone(self, **kwargs):
    obj = copy.deepcopy(self)
    obj.__dict__.update(kwargs)
    return obj


class Test(Cloneable):
  ...


if __name__ == "__main__":
  t = Prototype()
  t.name = "test"
  print(t.name)
  ct = t.clone()
  print(ct.name)

