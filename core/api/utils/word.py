# -*- coding: utf-8 -*-
"""Word API
  ======

  Custom API, containing methods for handling String
"""


def del_tail_digit(text) -> str:
  """api.utils.del_tail_digit

    删除字符串尾部的数字

    Args:
      text: Str. 原字符串
  """
  return text.rstrip('0123456789')


def get_iuwhx(x: int, rounder=2) -> str:
  """api.utils.get_iuwhx
  
    获取某一数字的带国际单位制词头的表示形式

    Args:
      x: int. 目标数字
      rounder: int. 包留小数点后几位，默认2位
  """
  _Xlen = (len(str(int(x))) - 1) // 3
  _X = 'KMGTPEZY'[_Xlen - 1]
  _num = round(x / (10 ** (_Xlen * 3)), rounder)
  return f'{_num}{_X}'


def get_ex(x: float) -> str:
  """api.utils.get_ex

    获取某一数字的科学计数法的表示形式

    Args:
      x: float. 目标数字
  """
  return f"{x:.1e}"

