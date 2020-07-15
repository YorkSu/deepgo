# -*- coding: utf-8 -*-
"""Dataset
  ======

  TensorFlow Example Datasets
"""


__all__ = [
    'BostonHousing',
    'Cifar10',
    'Cifar100',
    'Fashion_mnist',
    'Imdb',
    'Mnist',
    'Reuters',
    'get_dataset',
    'register_dataset']


from deepgo.core.api import np
from deepgo.core.api.keras import dataset as ds
from deepgo.core.api.utils import get_dataset
from deepgo.core.api.utils import register_dataset
from deepgo.framework.dataset import Dataset


class BostonHousing(Dataset):
  """BostonHousing

    Boston Housing数据集

    数据集描述:
      训练集: 404
      验证集: 102
      输入: (13,)
      输出: (1,)
    
    输入特征:
      1.CRIM - per capita crime rate by town
      2.ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
      3.INDUS - proportion of non-retail business acres per town.
      4.CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
      5.NOX - nitric oxides concentration (parts per 10 million)
      6.RM - average number of rooms per dwelling
      7.AGE - proportion of owner-occupied units built prior to 1940
      8.DIS - weighted distances to five Boston employment centres
      9.RAD - index of accessibility to radial highways
      10.TAX - full-value property-tax rate per $10,000
      11.PTRATIO - pupil-teacher ratio by town
      12.B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
      13.LSTAT - % lower status of the population
  """
  def __init__(self):
    super().__init__()
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.boston_housing.load_data()
    self.num_train = self.train_x.shape[0]
    self.num_val = self.val_x.shape[0]
    self.input_shape = self.train_x.shape[1:]
    self.output_shape = (1,)
    self.train_x = self.tonp(self.train_x)
    self.train_y = self.tonp(self.train_y)
    self.val_x = self.tonp(self.val_x)
    self.val_y = self.tonp(self.val_y)


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
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.cifar10.load_data()
    self.num_train = self.train_x.shape[0]
    self.num_val = self.val_x.shape[0]
    self.input_shape = self.train_x.shape[1:]
    self.output_shape = (10,)
    self.classes = 10
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
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.cifar100.load_data()
    self.num_train = self.train_x.shape[0]
    self.num_val = self.val_x.shape[0]
    self.input_shape = self.train_x.shape[1:]
    self.output_shape = (100,)
    self.classes = 100
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
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.fashion_mnist.load_data()
    self.num_train = self.train_x.shape[0]
    self.num_val = self.val_x.shape[0]
    self.input_shape = (28, 28, 1)
    self.output_shape = (10,)
    self.classes = 10
    self.train_x = self.train_x / 255.0
    self.train_x = self.train_x.reshape((self.num_train, *self.input_shape))
    self.val_x = self.val_x / 255.0
    self.val_x = self.val_x.reshape((self.num_val, *self.input_shape))
    self.train_x = self.tonp(self.train_x)
    self.train_y = self.tonp(self.train_y)
    self.val_x = self.tonp(self.val_x)
    self.val_y = self.tonp(self.val_y)


class Imdb(Dataset):
  """Imdb

    IMDB影评数据集

    标签数据集包含5万条IMDB影评，专门用于情绪分析。评论的情绪是二元的，
    这意味着IMDB评级< 5导致情绪得分为0，而评级>=7的情绪得分为1。没有哪
    部电影的评论超过30条。标有training set的2.5万篇影评不包括与2.5万篇
    影评测试集相同的电影。此外，还有另外5万篇IMDB影评没有任何评级标签。
  
    数据集描述:
      训练集: 25000
      验证集: 25000
      输入尺寸: (maxlen,)
      输出尺寸: (1,)
  """
  def __init__(self, num_words=None, maxlen=80):
    super().__init__()
    self.num_words = num_words
    self.maxlen = maxlen
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.imdb.load_data(num_words=self.num_words)
    self.train_x = _vectorize(self.train_x, dimension=maxlen)
    self.val_x = _vectorize(self.val_x, dimension=maxlen)
    self.num_train = self.train_x.shape[0]
    self.num_val = self.val_x.shape[0]
    self.input_shape = self.train_x.shape[1:]
    self.output_shape = (1,)
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
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.mnist.load_data()
    self.num_train = self.train_x.shape[0]
    self.num_val = self.val_x.shape[0]
    self.input_shape = (28, 28, 1)
    self.output_shape = (10,)
    self.classes = 10
    self.train_x = self.train_x / 255.0
    self.train_x = self.train_x.reshape((self.num_train, *self.input_shape))
    self.val_x = self.val_x / 255.0
    self.val_x = self.val_x.reshape((self.num_val, *self.input_shape))
    self.train_x = self.tonp(self.train_x)
    self.train_y = self.tonp(self.train_y)
    self.val_x = self.tonp(self.val_x)
    self.val_y = self.tonp(self.val_y)


class Reuters(Dataset):
  """Reuters

    Reuters新闻分类数据集

    本数据库包含来自路透社的11,228条新闻，分为了46个主题。与IMDB库一样，
    每条新闻被编码为一个词下标的序列。

    数据集描述:
      训练集: 8982
      验证集: 2246
      输入尺寸: (maxlen,)
      输出尺寸: (1,)
  """
  def __init__(self, num_words=None, maxlen=1000):
    super().__init__()
    self.num_words = num_words
    self.maxlen = maxlen
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.reuters.load_data(num_words=self.num_words)
    self.train_x = _vectorize(self.train_x, dimension=maxlen)
    self.val_x = _vectorize(self.val_x, dimension=maxlen)
    self.num_train = self.train_x.shape[0]
    self.num_val = self.val_x.shape[0]
    self.input_shape = self.train_x.shape[1:]
    self.output_shape = (1,)
    self.train_x = self.tonp(self.train_x)
    self.train_y = self.tonp(self.train_y)
    self.val_x = self.tonp(self.val_x)
    self.val_y = self.tonp(self.val_y)


class ReutersWord(object):
  """ReutersWord

    将Reuters数据集向量转化为单词
  """
  def __init__(self):
    self.word_index = ds.reuters.get_word_index()
    self.index_word = dict([(value, key) for (key, value) in self.word_index.items()])

  def __call__(self, sample, **kwargs):
    return self.decode(sample, **kwargs)

  def decode(self, sample, ignore_zero=True):
    words = []
    for i in sample:
      if ignore_zero and i == 0:
        continue
      words.append(self.index_word.get(i, '?'))
    return ' '.join(words)


def _vectorize(sequences, length=None, dimension=None) -> np.ndarray:
  if length is None:
    length = len(sequences)
  if dimension is None:
    dimension = max([len(i) for i in sequences])
  result = np.zeros((length, dimension))
  for inx, vector in enumerate(sequences):
    if len(vector) > dimension:
      vector = vector[:dimension]
    result[inx, :len(vector)] += vector
  return result


# Register Datasets
register_dataset('bostonhousing', BostonHousing)
register_dataset('cifar10', Cifar10)
register_dataset('cifar100', Cifar100)
register_dataset('fashion_mnist', Fashion_mnist)
register_dataset('imdb', Imdb)
register_dataset('mnist', Mnist)
register_dataset('reuters', Reuters)

