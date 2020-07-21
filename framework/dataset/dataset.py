# -*- coding: utf-8 -*-
"""Dataset
  ======

  Deep Go General Dataset Class
"""


from deepgo.core.abcs import Manager
from deepgo.core.api import get_floatx
from deepgo.core.api import np


class Dataset(Manager):
  """Dataset

    数据集对象的父类
  """
  def __init__(self, floatx=None):
    if floatx is None:
      floatx = get_floatx()
    self._floatx = floatx
    # Default Attributes
    self.num_train = 0
    self.num_val = 0
    self.num_test = 0
    self.input_shape = ()
    self.output_shape = ()
    self.classes = 0
    self.train_x = None
    self.train_y = None
    self.val_x = None
    self.val_y = None
    self.test_x = None
    self.test_y = None

  @property
  def dict(self):
    return self.__dict__

  def get(self, key):
    """Return the value of the specified key.
      
      If the key does not exist, return None.
    """
    return self.__dict__.get(key)

  def set(self, key, value):
    """Set the value of the specified key."""
    self.__dict__[key] = value

  def tonp(self, value):
    """Return a np.ndarray object with floatx dtype from value."""
    return np.array(value).astype(self._floatx)


if __name__ == "__main__":
  class Test(Dataset):
    def __init__(self):
      super().__init__()
      self.num_val = 1

  data = Test()
  print(data.get('num_train'))
  print(data.get('num_val'))
  print(data.dict)

