# -*- coding: utf-8 -*-
"""Environment
  ======

  Deep Go Environment
"""

import os
import json

# from deepgo.core.abcs import Manager


class Conf(object):
  """Title

    Description
  """
  def __init__(self):
    self.json = json.load(open("./conf.json"))


class LazyLoader(object):
  """Lazy Loader

    Load the third-party libraries if necessary.

    Support libraries:
      tensorflow
  """
  @property
  def tensorflow(self):
    """tensorflow Lazy Loader
    
      load tensorflow and check the version
    """
    if 'tensorflow' not in self.__dict__:
      try:
        import tensorflow
        self.__dict__['tensorflow'] = tensorflow
        if tensorflow.__version__ != '2.1.0':
          print("[Warning] tf version is not 2.1.0, may cause problems.")  # TODO: Use Logger
      except ImportError:
        print("[Error] tensorflow cannot be imported.")  # TODO: Use Logger
        self.__dict__['tensorflow'] = None
    return self.__dict__['tensorflow']

  tf = tensorflow


Config = Conf()
Loader = LazyLoader()


if __name__ == "__main__":
  # tf = Loader.tf
  # print(tf.__version__)
  # print(Config.json)
  print(__package__)
  print(__module__)



