# -*- coding: utf-8 -*-
"""Dataset
  ======

  TensorFlow Example Datasets
"""


__all__ = [
    'Cifar10',
    'Cifar100',
    'Fashion_mnist',
    'Mnist',
    'get_dataset',
    'register_dataset']


from deepgo.core.api.keras import dataset as ds
from deepgo.core.api.utils import get_dataset
from deepgo.core.api.utils import register_dataset
from deepgo.framework.dataset import Dataset


class Cifar10(Dataset):
  """Cifar10

    Cifar10数据集

    数据集描述:
      训练集: 50000
      验证集: 10000
      图像尺寸: (32, 32, 3)
      分类: 10
  """
  def __init__(self):
    super().__init__()
    self.num_train = 50000
    self.num_val = 10000
    self.input_shape = (32, 32, 3)
    self.output_shape = (10,)
    self.classes = 10
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.cifar10.load_data()
    self.train_x = self.train_x / 255.0
    self.val_x = self.val_x / 255.0
    self.train_x = self.tonp(self.train_x)
    self.train_y = self.tonp(self.train_y)
    self.val_x = self.tonp(self.val_x)
    self.val_y = self.tonp(self.val_y)


class Cifar100(Dataset):
  """Cifar100

    Cifar100数据集

    数据集描述:
      训练集: 50000
      验证集: 10000
      图像尺寸: (32, 32, 3)
      分类: 100
  """
  def __init__(self):
    super().__init__()
    self.num_train = 50000
    self.num_val = 10000
    self.input_shape = (32, 32, 3)
    self.output_shape = (100,)
    self.classes = 100
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.cifar100.load_data()
    self.train_x = self.train_x / 255.0
    self.val_x = self.val_x / 255.0
    self.train_x = self.tonp(self.train_x)
    self.train_y = self.tonp(self.train_y)
    self.val_x = self.tonp(self.val_x)
    self.val_y = self.tonp(self.val_y)


class Fashion_mnist(Dataset):
  """Fashion_mnist

    十类服装黑白图片数据集，Mnist数据集的简易替换

    数据集描述:
      训练集: 60000
      验证集: 10000
      图像尺寸: (28, 28, 1)
      分类: 10
  """
  def __init__(self):
    super().__init__()
    self.num_train = 60000
    self.num_val = 10000
    self.input_shape = (28, 28, 1)
    self.output_shape = (10,)
    self.classes = 10
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.fashion_mnist.load_data()
    self.train_x = self.train_x / 255.0
    self.train_x = self.train_x.reshape((self.num_train, *self.input_shape))
    self.val_x = self.val_x / 255.0
    self.val_x = self.val_x.reshape((self.num_val, *self.input_shape))
    self.train_x = self.tonp(self.train_x)
    self.train_y = self.tonp(self.train_y)
    self.val_x = self.tonp(self.val_x)
    self.val_y = self.tonp(self.val_y)


class Mnist(Dataset):
  """Mnist

    手写数字数据集

    数据集描述:
      训练集: 60000
      验证集: 10000
      图像尺寸: (28, 28, 1)
      分类: 10
  """
  def __init__(self):
    super().__init__()
    self.num_train = 60000
    self.num_val = 10000
    self.input_shape = (28, 28, 1)
    self.output_shape = (10,)
    self.classes = 10
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.mnist.load_data()
    self.train_x = self.train_x / 255.0
    self.train_x = self.train_x.reshape((self.num_train, *self.input_shape))
    self.val_x = self.val_x / 255.0
    self.val_x = self.val_x.reshape((self.num_val, *self.input_shape))
    self.train_x = self.tonp(self.train_x)
    self.train_y = self.tonp(self.train_y)
    self.val_x = self.tonp(self.val_x)
    self.val_y = self.tonp(self.val_y)


# Register Datasets
register_dataset('cifar10', Cifar10)
register_dataset('cifar100', Cifar100)
register_dataset('fashion_mnist', Fashion_mnist)
register_dataset('mnist', Mnist)

