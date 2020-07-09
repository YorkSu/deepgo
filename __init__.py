# -*- coding: utf-8 -*-
"""Deep Go
  ======

  An framework for machine learning and deep learning.
  
  Version:
    1.0 - alpha

  Author:
    York Su
"""


import os as _os
import sys as _sys
this_dir = _os.path.dirname(_os.path.abspath(__file__))
if this_dir not in _sys.path:
  _sys.path.append(this_dir)
del _os, _sys


from deepgo import core
from deepgo import framework

from deepgo.core import api
from deepgo.core import exception
from deepgo.core import io
from deepgo.core import layer

from deepgo.framework import log
from deepgo.framework.config import Config
from deepgo.framework.dataset import Dataset

