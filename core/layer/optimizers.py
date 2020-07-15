# -*- coding: utf-8 -*-
"""Keras Optimizer
  =====

  Keras Optimizers
"""


__all__ = [
    'Adadelta',
    'Adagrad',
    'Adam',
    'Adamax',
    'Nadam',
    'Optimizer',
    'RMSprop',
    'SGD',]


from tensorflow.keras import optimizers as _optimizers


class Optimizer(_optimizers.Optimizer):
  """Abstract optimizer base class.

  Note: this is the parent class of all optimizers, not an actual optimizer
  that can be used for training models.

  All Keras optimizers support the following keyword arguments:

      clipnorm: float >= 0. Gradients will be clipped
          when their L2 norm exceeds this value.
      clipvalue: float >= 0. Gradients will be clipped
          when their absolute value exceeds this value.
  """


class SGD(_optimizers.SGD):
  """Stochastic gradient descent optimizer.

  Includes support for momentum,
  learning rate decay, and Nesterov momentum.

  Arguments:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter that accelerates SGD in the relevant
        direction and dampens oscillations.
      decay: float >= 0. Learning rate decay over each update.
      nesterov: boolean. Whether to apply Nesterov momentum.
  """
  def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
    super().__init__(lr, momentum, decay, nesterov, **kwargs)


class RMSprop(_optimizers.RMSprop):
  """RMSProp optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values
  (except the learning rate, which can be freely tuned).

  Arguments:
      lr: float >= 0. Learning rate.
      rho: float >= 0.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
  """
  def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0., **kwargs):
    super().__init__(lr, rho, epsilon, decay, **kwargs)


class Adagrad(_optimizers.Adagrad):
  """Adagrad optimizer.

  Adagrad is an optimizer with parameter-specific learning rates,
  which are adapted relative to how frequently a parameter gets
  updated during training. The more updates a parameter receives,
  the smaller the updates.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  # Arguments
      lr: float >= 0. Initial learning rate.
      epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.

  # References
      - [Adaptive Subgradient Methods for Online Learning and Stochastic
      Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  """
  def __init__(self, lr=0.01, epsilon=None, decay=0., **kwargs):
    super().__init__(lr, epsilon, decay, **kwargs)


class Adadelta(_optimizers.Adadelta):
  """Adadelta optimizer.

  Adadelta is a more robust extension of Adagrad
  that adapts learning rates based on a moving window of gradient updates,
  instead of accumulating all past gradients. This way, Adadelta continues
  learning even when many updates have been done. Compared to Adagrad, in the
  original version of Adadelta you don't have to set an initial learning
  rate. In this version, initial learning rate and decay factor can
  be set, as in most other Keras optimizers.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  # Arguments
      lr: float >= 0. Initial learning rate, defaults to 1.
          It is recommended to leave it at the default value.
      rho: float >= 0. Adadelta decay factor, corresponding to fraction of
          gradient to keep at each time step.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Initial learning rate decay.

  # References
      - [Adadelta - an adaptive learning rate
      method](http://arxiv.org/abs/1212.5701)
  """
  def __init__(self, lr=1.0, rho=0.95, epsilon=None, decay=0., **kwargs):
    super().__init__(lr, rho, epsilon, decay, **kwargs)


class Adam(_optimizers.Adam):
  """Adam optimizer.

  Default parameters follow those provided in the original paper.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
      amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm
        from the paper "On the Convergence of Adam and Beyond".
  """
  def __init__(self,
      lr=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=None,
      decay=0.,
      amsgrad=False,
      **kwargs):
    super().__init__(lr, beta_1, beta_2, epsilon, decay, amsgrad, **kwargs)


class Adamax(_optimizers.Adamax):
  """Adamax optimizer from Adam paper's Section 7.

  It is a variant of Adam based on the infinity norm.
  Default parameters follow those provided in the paper.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
  """

  def __init__(self,
      lr=0.002,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=None,
      decay=0.,
      **kwargs):
    super().__init__(lr, beta_1, beta_2, epsilon, decay, **kwargs)


class Nadam(_optimizers.Nadam):
  """Nesterov Adam optimizer.

  Much like Adam is essentially RMSprop with momentum,
  Nadam is Adam RMSprop with Nesterov momentum.

  Default parameters follow those provided in the paper.
  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
  """

  def __init__(self,
      lr=0.002,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=None,
      schedule_decay=0.004,
      **kwargs):
    super().__init__(lr, beta_1, beta_2, epsilon, schedule_decay, **kwargs)

