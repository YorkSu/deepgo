# -*- coding: utf-8 -*-
"""Train
  ======

  Deep Go Trainer
"""


import time

from deepgo.core.abcs import Manager
from deepgo.core.api import set_learning_phase
from deepgo.core.layer import Model
from deepgo.framework.config import Config
from deepgo.framework.generator import Generator


class Trainer(Manager):
  """Abstract Trainer Class
  
    Args:
      model: Model.
  """
  def __init__(self, model: Model, config: Config, *args, **kwargs):
    self._model = None
    self._config = Config()
    self.set_model(model)
    self.set_config(config)

  def __call__(self, *args, **kwargs):
    self.call(*args, **kwargs)

  def call(self, *args, **kwargs):
    """Function to start the entire training

      Subclasses should override for any actions to run.
    """

  @property
  def model(self):
    return self._model

  def set_model(self, model: Model):
    if isinstance(model, Model):
      self._model = model
    else:
      # self._model = None
      pass # TODO: log "model is not a Model"

  @property
  def config(self):
    return self._config

  def set_config(self, config: Config):
    if isinstance(config, Config):
      self._config = config
    elif isinstance(config, dict):
      self._config = Config(json=config)
    else:
      # self._config = Config(json={})
      pass # TODO: log "config is empty, use default config"

  def train(self, *args, **kwargs):
    """The entire training process
    
      Subclasses should override for any actions to run.

      Before training, the `on_train_begin` function should be called.

      After training, the `on_train_end` function should be called.
    """

  def on_train_begin(self, *args, **kwargs):
    """Called at the beginning of training
    
      Subclasses should override for any actions to run.
    """

  def on_train_end(self, *args, **kwargs):
    """Called at the end of training
    
      Subclasses should override for any actions to run.
    """

  def epoch(self, *args, **kwargs):
    """The training process for each epoch
    
      Subclasses should override for any actions to run.

      Before epoch, the `on_epoch_begin` function should be called.

      After epoch, the `on_epoch_end` function should be called.
    """

  def on_epoch_begin(self, *args, **kwargs):
    """Called at the beginning of an epoch
    
      Subclasses should override for any actions to run.
    """

  def on_epoch_end(self, *args, **kwargs):
    """Called at the end of an epoch
    
      Subclasses should override for any actions to run.
    """

  def batch(self, *args, **kwargs):
    """The training process for each batch
    
      Subclasses should override for any actions to run.

      Before batch, the `on_batch_begin` function should be called.

      After batch, the `on_batch_end` function should be called.
    """

  def on_batch_begin(self, *args, **kwargs):
    """Called at the beginning of a batch
    
      Subclasses should override for any actions to run.
    """

  def on_batch_end(self, *args, **kwargs):
    """Called at the end of a batch
    
      Subclasses should override for any actions to run.
    """


class StepTrainer(Trainer):
  def __init__(self, model: Model, config: Config, **kwargs):
    super().__init__(model, config)
    self._Generator = None
    self._record = {
        'cost': [],
        'loss': [],
        'accuracy': []}
    self._train_x = None
    self._train_y = None
    self._val_x = None
    self._val_y = None
    self._batch_size = self.config.get('batch_size')
    self._step = self.config.get('step')
    self._step_per_log = self.config.get('step_per_log')
    self._step_per_val = self.config.get('step_per_val')

  def call(self, train_x, trian_y, val_x, val_y):
    """Function to start the entire training"""
    self._train_x = train_x
    self._train_y = trian_y
    self._val_x = val_x
    self._val_y = val_y
    self._Generator = Generator(self._train_x, self._train_y, self._batch_size)
    self.train()

  def train(self):
    """The entire training process"""
    self.on_train_begin()
    for step in range(self._step):
      result = self.batch(step + 1)
    self.on_train_end()
    
  def batch(self, step):
    """The training process for each batch"""
    begin = self.on_batch_begin()
    x, y = next(self._Generator)
    result = self.model.train_on_batch(x, y)
    end = self.on_batch_end(step, begin, result)
    return end

  def on_train_begin(self):
    """Called at the beginning of training"""
    set_learning_phase(1)

  def on_train_end(self, *args, **kwargs):
    """Called at the end of training"""
    set_learning_phase(0)

  def on_batch_begin(self):
    """Called at the beginning of a batch"""
    return time.perf_counter()

  def on_batch_end(self, step, begin, result):
    """Called at the end of a batch"""
    end = time.perf_counter()
    cost = round(end - begin, 6)
    loss = result[0]
    accuracy = result[1]
    self._record['cost'].append(cost)
    self._record['loss'].append(loss)
    self._record['accuracy'].append(accuracy)
    if not (step % self._step_per_log):
      mean_cost = sum(self._record['cost']) / self._step_per_log
      mean_loss = sum(self._record['loss']) / self._step_per_log
      mean_accuracy = sum(self._record['accuracy']) / self._step_per_log
      self._record = {
        'cost': [],
        'loss': [],
        'accuracy': []}
      print(f"Step: {step}, \tCost: {mean_cost:6f}, Loss: {mean_loss:4f}, Accuracy: {mean_accuracy:4f}")
      # TODO: Log
    if not (step % self._step_per_val):
      val_cost, val_loss, val_accuracy = self.evaluate()
      print(f"Val:  {step}, \tCost: {val_cost:6f}, Loss: {val_loss:4f}, Accuracy: {val_accuracy:4f}")
      # TODO: Log

  def evaluate(self):
    """Evaluate Function"""
    set_learning_phase(0)
    begin = time.perf_counter()
    result = self.model.evaluate(
        self._val_x,
        self._val_y,
        self._batch_size,
        verbose=0)
    end = time.perf_counter()
    cost = round(end - begin, 6)
    set_learning_phase(1)
    return cost, result[0], result[1]


if __name__ == "__main__":
  # t1 = time.perf_counter_ns()
  # time.sleep(1)
  # t2 = time.perf_counter_ns()
  # cost = round((t2-t1) / 1e9, 6)
  # print(cost, type(cost))
  # t1 = time.perf_counter()
  # time.sleep(1)
  # t2 = time.perf_counter()
  # cost = round(t2 - t1, 6)
  # print(cost, type(cost), time.perf_counter())
  pass





