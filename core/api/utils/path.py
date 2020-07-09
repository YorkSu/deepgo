# -*- coding: utf-8 -*-
"""Path API
  =====

  Custom API, containing methods for handling path
"""


import os


def get_current_path(value):
  """api.utils.get_current_path

    Get the current absolute path from the value
  
    Args:
      value: string. Recommended `__file__`
  """
  return os.path.dirname(os.path.abspath(value))


def current_path_filename(value, filename):
  """api.utils.current_path_filename

    Get the absolute path from the value and filename
  
    Args:
      value: string. Recommended `__file__`
      filename: string. The File name
    
    Returns:
      "{current_path}/{filename}"
  """
  return os.path.join(get_current_path(value), filename)

