# -*- coding: utf-8 -*-
"""Stream IO
  ======

  Deep Go Stream IO Class
"""


import os
import sys

from deepgo.core.abcs import IO


class StreamIO(IO):
  """Stream IO
    ======
  
    Deep Go Stream IO Class
  """
  def __init__(self):
    self._stdout = sys.stdout

  def print(self, *args, **kwargs):
    """Title

      Description
    """
    print(*args, **kwargs)

  def input(self, *args, **kwargs):
    """Title

      Description
    """
    return input(*args, **kwargs)

  def clean(self):
    """Title

      Description
    """
    os.system("cls")

  def flush(self):
    """Title

      Description
    """
    self._stdout.flush()

