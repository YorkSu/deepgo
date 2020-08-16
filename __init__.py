# -*- coding: utf-8 -*-
"""Deep Go
  ======

  An framework for Deep Learning
  
  Version:
    Aiur 0.1.0
  
  Author:
    York Su
"""


import os as _os
import sys as _sys
this_dir = _os.path.dirname(_os.path.abspath(__file__))
if this_dir not in _sys.path:
  _sys.path.append(this_dir)
del _os, _sys


__version__ = "0.1.0"
__codename__ = "Aiur"
__release_date__ = "2020-08-16"
__author__ = "York Su"

