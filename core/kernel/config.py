# -*- coding: utf-8 -*-
"""Config
  ======

  Config Manager
"""


from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod
from deepgo.core.kernel.flags import flags
from deepgo.core.ioc.jsonio import jsonio


class AbstractConfig(AbstractSingleton):
  """Abstract Config Class

    This is a Abstract Singleton Class
  """
  @abstractmethod
  def fmdict(self, dictionary): ...
    
  @abstractmethod
  def fmjson(self, filename): ...

  @abstractmethod
  def update(self): ...


class Config(AbstractConfig):
  """Deep Go Config Manager Class

    This is a Singleton Class
  """
  def __init__(self):
    self.temp_dict = {}

  def fmdict(self, dictionary):
    """Title

      Description
    """
    with self._lock:
      self.temp_dict.update(dictionary)
    return self

  def fmjson(self, filename):
    """Title

      Description
    """
    dictionary = jsonio.load(filename)
    return self.fmdict(dictionary)

  def update(self):
    """Title

      Description
    """
    with self._lock:
      for key in self.temp_dict:
        flags.set(key, self.temp_dict[key])


config = Config()


if __name__ == "__main__":
  config.fmdict({"name": "John"}).update()
  print(flags.name)

