# -*- coding: utf-8 -*-
"""Tools API
  ======

  Custom API, containing Useful methods and classes
"""


import time

from deepgo.core.abcs import Context


class Timer(Context):
  """Timer

    Output stake in the block of code.

    Args:
      prec: Integer, precision of the output Seconds.

    Usage:
    ```python
    with Timer():
      ...
    ```
  """
  def __init__(self, prec=6):
    self.prec = prec
    self.start = 0.
    self.stop = 0.

  def __enter__(self):
    self.start = time.perf_counter()
    return self

  def __exit__(self, type, value, traceback):
    self.stop = time.perf_counter()
    cost = round(self.stop - self.start, self.prec)
    # TODO: use io.StreamIO
    print(f"This Codes Cost: {cost} Seconds.")


def normalize_tuple(obj, rank):
  """Ensure Object is a tuple of Integer and length equal rank"""
  if isinstance(obj, (list, tuple)):
    if len(obj) == rank:
      return tuple(obj)
    raise ValueError(f"Expected a list/tuple of rank {rank}, got {len(obj)}")
  if isinstance(obj, int):
    return tuple([obj for _ in range(rank)])
  raise ValueError(f"Expected a int/list/tuple, got {obj.__class__.__name__}")

