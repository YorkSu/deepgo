# -*- coding: utf-8 -*-
"""Keras Regularizer
  =====

  Keras Regularizers
"""


__all__ = [
    'L1L2',
    'l1',
    'l2',
    'l1_l2']


from tensorflow.keras import regularizers as _regularizers


class L1L2(_regularizers.L1L2):
  r"""A regularizer that applies both L1 and L2 regularization penalties.

  The L1 regularization penalty is computed as:
  $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

  The L2 regularization penalty is computed as
  $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

  Attributes:
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.
  """
  def __init__(self, l1=0., l2=0.):
    super().__init__(l1, l2)


def l1(l=0.01):
  r"""Create a regularizer that applies an L1 regularization penalty.

  The L1 regularization penalty is computed as:
  $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

  Arguments:
      l: Float; L1 regularization factor.

  Returns:
    An L1 Regularizer with the given regularization factor.
  """
  return L1L2(l1=l)


def l2(l=0.01):
  r"""Create a regularizer that applies an L2 regularization penalty.

  The L2 regularization penalty is computed as:
  $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

  Arguments:
      l: Float; L2 regularization factor.

  Returns:
    An L2 Regularizer with the given regularization factor.
  """
  return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
  r"""Create a regularizer that applies both L1 and L2 penalties.

  The L1 regularization penalty is computed as:
  $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

  The L2 regularization penalty is computed as:
  $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

  Arguments:
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.

  Returns:
    An L1L2 Regularizer with the given regularization factors.
  """
  return L1L2(l1=l1, l2=l2)

