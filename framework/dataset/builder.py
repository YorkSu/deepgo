# -*- coding: utf-8 -*-
"""Builder
  ======

  Deep Go Dataset Builder
"""


import os
import time

from deepgo.core.io import h5io
from deepgo.core.abcs import Manager
from deepgo.core.api import np
from deepgo.core.api import get_floatx
from deepgo.framework.dataset.dataset import Dataset


class H5Dataset(Manager):
  """HDF5 Dataset Manager

    Deep Go Abstract HDF5 Dataset Manager Class

    This class should not be instantiated.

    See the Deep Go API for more information.

    Definition:
    ```python
    import deepgo as dp
    class Mnist(dp.framework.dataset.H5Dataset):
      def init(self):
        self.name = "mnist.h5"
        self.path = dp.api.get_current_path(__file__)
        self.mode = 'a'
        self.suffix = '.h5'
        self.data = dp.api.dataset.mnist.load_data()
        self.config = {
            'num_train': len(self.data[0][0]),
            'num_val': len(self.data[1][0]),
            'input_shape': (28, 28, 1),
            'classes': 10}
      def train(self):
        train_x = self.data[0][0]
        train_y = self.data[0][1]
        train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
        return train_x, train_y
      def val(self):
        val_x = self.data[1][0]
        val_y = self.data[1][1]
        val_x = val_x.reshape((val_x.shape[0], 28, 28, 1))
        return val_x, val_y
    ```

    Usage:
    ```python
    Manager = Mnist()
    Manager.build()
    data = Manager.load()
    ```
  """
  def __init__(self):
    self.name = ''
    self.path = None
    self.mode = 'a'
    self.suffix = '.h5'
    self.config = {}
    self.init()
    self.open()
    
  def __del__(self):
    try:
      h5io.close(self.f)
    except:
      pass

  def init(self):
    """Title

      Description
    """
    self.name = ''
    self.path = None
    self.mode = 'a'
    self.suffix = '.h5'
    self.config = {}

  def open(self):
    """Title

      Description
    """
    filename = h5io.filename(self.name, self.path, self.suffix)
    if filename == self.suffix:
      return
    self.filename = filename
    self.f = h5io.open(filename, self.mode)

  def read(self, dset, dtype=None):
    """Title

      Description
    """
    if dtype is None:
      dtype = get_floatx()
    length = h5io.len(dset)
    output = h5io.empty(dset, length, dtype=dtype)
    h5io.read_direct(dset, output)
    return output

  def write(self, dset, datas):
    """Title

      Description
    """
    h5io.batch_fill(dset, datas, 0)

  def build(self):
    """Title

      Description
    """
    f = self.f
    h5io.set_attrs(f, self.config)
    data_train_x, data_train_y = self.train()
    data_val_x, data_val_y = self.val()
    data_test_x, data_test_y = self.test()
    if data_train_x is not None and data_train_y is not None:
      train = h5io.create_group(f, "train")
      train_x = h5io.create_dataset(train, "x", data_train_x.shape[1:])
      self.write(train_x, data_train_x)
      train_y = h5io.create_dataset(train, "y", data_train_y.shape[1:])
      self.write(train_y, data_train_y)
    if data_val_x is not None and data_val_y is not None:
      val = h5io.create_group(f, "val")
      val_x = h5io.create_dataset(val, "x", data_val_x.shape[1:])
      self.write(val_x, data_val_x)
      val_y = h5io.create_dataset(val, "y", data_val_y.shape[1:])
      self.write(val_y, data_val_y)
    if data_test_x is not None and data_test_y is not None:
      test = h5io.create_group(f, "test")
      test_x = h5io.create_dataset(test, "x", data_test_x.shape[1:])
      self.write(test_x, data_test_x)
      test_y = h5io.create_dataset(test, "y", data_test_y.shape[1:])
      self.write(test_y, data_test_y)

  def load(self):
    """Title

      Description
    """
    f = self.f
    keys = h5io.keys(f)
    config = h5io.get_attrs(f)
    output = Dataset()
    for key, value in config.items():
      output.set(key, value)
    if 'train' in keys:
      train_x = f["train/x"]
      train_y = f["train/y"]
      output.set('train_x', self.read(train_x))
      output.set('train_y', self.read(train_y))
    if 'val' in keys:
      val_x = f["val/x"]
      val_y = f["val/y"]
      output.set('val_x', self.read(val_x))
      output.set('val_y', self.read(val_y))
    if 'test' in keys:
      test_x = f["test/x"]
      output.set('test_x', self.read(test_x))
      if 'y' in f['test'].keys():
        test_y = f['test/y']
        output.set('test_y', self.read(test_y))
    return output

  def train(self):
    """Title

      Description
    """
    return None, None
  
  def val(self):
    """Title

      Description
    """
    return None, None

  def test(self):
    """Title

      Description
    """
    return None, None

