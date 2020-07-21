# -*- coding: utf-8 -*-
"""Json IO
  ======

  Deep Go JSON IO Class
"""


import json

from deepgo.core.abcs import IO
from deepgo.core import exception


class JsonIO(IO):
  """Json IO
    ======
  
    Deep Go Json IO Class
  """
  def __init__(self, filename='', **kwargs):
    self._filename = filename
    self._dict = {}
    self.load()
    if not filename:
      self.update(kwargs)

  @property
  def dict(self):
    return self._dict

  def get(self, key):
    """Return the value of the specified key.

      If the key does not exist, return None
    """
    return self._dict.get(key)

  def load(self):
    """Load the JSON file from filename"""
    if os.path.exists(self._filename):
      self._dict = json.load(open(self._filename))

  def merge(self, other):
    """Merge this JsonIO object with another JsonIO object"""
    if not isinstance(other, JsonIO):
      raise exception.TypeError(", ".join([
          f"Excepted JsonIO",
          f"got {other.__class__.__name__}"]))
    self._dict.update(other._dict)

  def save(self, indent=2):
    """Save the JSON file to filename"""
    json.dump(self._dict, open(self._filename, 'w'), indent=indent)

  def set(self, key, value):
    """Set the value of the specified key"""
    self._dict[key] = value

  def update(self, dictionary):
    """Update the key-value pairs from dictionary"""
    self._dict.update(dictionary)

  # Aliases
  dump = save

