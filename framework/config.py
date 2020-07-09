# -*- coding: utf-8 -*-
"""Config
  ======

  Config Handler
"""


from deepgo.core.io import Json


class Config(object):
  """Config

    Config Handler

    Args:
      json: io.Json, Dict, or Filename.
      **kwargs: keyword argument, cover the json config
  """
  def __init__(self, json={}, **kwargs):
    self._dict = {}
    self.update(json)
    self.update(kwargs)

  @property
  def dict(self):
    return self._dict

  def get(self, key):
    return self._dict.get(key)

  def set(self, key, value):
    self._dict[key] = value
  
  def update(self, json):
    if isinstance(json, Json):
        self._dict.update(json.dict)
    elif isinstance(json, dict):
        self._dict.update(json)
    elif isinstance(json, str):
        self._dict.update(Json(json).dict)

