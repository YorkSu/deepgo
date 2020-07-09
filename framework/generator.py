# -*- coding: utf-8 -*-
"""Generator
  ======

  Deep Go Data Generator
"""


import math

from deepgo.core.api import np


class Generator(object):
  """Generator

    数据生成器

    Args:
      x: np.array. 推荐使用`Dataset.train_x`.
      y: np.array. 推荐使用`Dataset.train_y`.
      batch_size: Int. 批量大小.
      shuffle: Boolean. 是否进行打乱
      aug: keras.ImageDataGenerator, default None. 数据增强器
      **kwargs: Any. 任意参数，自动忽略
  """
  def __init__(self,
        x: np.array,
        y: np.array,
        batch_size: int,
        shuffle=True,
        aug=None,
        **kwargs):
    self._x = x
    self._y = y
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._aug = aug
    self._len = math.ceil(len(x) / self._batch_size)
    self._iterator = self.__iterator__()
    del kwargs

  def __next__(self):
    return next(self._iterator)

  def __iter__(self):
    for i in range(self._len):
      yield next(self)
  
  def __del__(self):
    try:
      del self._iterator
    except Exception:
      pass

  def __iterator__(self):
    while True:
      epoch = self.shuffle([self._x, self._y])
      for inx in range(self._len):
        if inx + 1 == self._len:
          x = epoch[0][inx * self._batch_size:]
          y = epoch[1][inx * self._batch_size:]
        else:
          x = epoch[0][inx * self._batch_size:(inx + 1) * self._batch_size]
          y = epoch[1][inx * self._batch_size:(inx + 1) * self._batch_size]
        if self._aug is not None:
          x = next(self._aug.flow(x, batch_size=self._batch_size))
        yield x, y

  def shuffle(self, inputs):
    if not self._shuffle:
      return inputs
    r = np.random.permutation(len(inputs[-1]))
    return inputs[0][r], inputs[1][r]

