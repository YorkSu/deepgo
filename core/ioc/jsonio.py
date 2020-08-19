# -*- coding: utf-8 -*-
"""Json IO
  ======

  JSON File IO Class
"""


import os
import json
from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod


class AbstractJsonIO(AbstractSingleton):
  """Abstract JSON IO Class

    This is a Abstract Singleton Class
  """
  @abstractmethod
  def load(self, filename): ...
  
  @abstractmethod
  def dump(self, obj, filename): ...


class JsonIO(AbstractJsonIO):
  """JSON File IO Class
  
    This is a Singleton Class
  """
  def load(self, filename, encoding='utf-8') -> dict:
    """Load the JSON file from filename

      If file not exist, return {}

      Arguments:
        filename: String. the JSON filename
        encoding: String. the JSON encoding
    """
    if not os.path.exists(filename):
      return {}
    with self._lock:
      dictionary = json.load(open(filename, encoding=encoding))
    return dictionary

  def dump(self, obj, filename, indent=2, encoding='utf-8',
      ensure_ascii=True):
    """Dump the object to the JSON file
    
      Arguments:
        obj: the object to dump
        filename: String. the JSON filename
        indent: Integer. The JSON indent
        encoding: String. the JSON encoding
        ensure_ascii: Boolean. 
    """
    with self._lock:
      json.dump(
          obj,
          open(filename, 'w', encoding=encoding),
          indent=indent,
          ensure_ascii=ensure_ascii)


jsonio = JsonIO()

