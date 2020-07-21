# -*- coding: utf-8 -*-
"""Log
  ======

  Deep Go Logging Module
"""


import os
import logging
from logging import handlers

from deepgo.core.abcs import Manager


class Logger(Manager):
  """Logger
  
    日志工具
  """
  def __init__(self, **kwargs):
    self._filename = ''
    self._fullname = ''
    self._detial = True
    self._suffix = '.log'
    self._filemode = 'a+'
    self._fmt = ''
    self._datefmt = ''
    self._built = False
    self._no_init_warning = False
    # Init StreamHandler
    logging.basicConfig(level=logging.INFO)
    
  def _check_built(self):
    if not self._built and not self._no_init_warning:
      print("[WARNING] Logger has not been built. This MSG will only log once.")
      self._no_init_warning = True

  def build(self, filename: str, detail=True, suffix='.log', filemode='a+',
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] >%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"):
    """Logger.build
    
      日志工具初始化函数。

      Args:
        filename: Str. 日志文件名，包含路径。如果文件名不带有后缀名，会自动补上
            若文件名为空，则不输出日志文件
        detail: Bool. 日志写文件粒度。默认最低为DEBUG。
            若值为False，则写文件的最低等级为INFO
        suffix: Str. 日志文件后缀名，默认为'.log'
        filemode: Str. 日志文件的打开方式，同open.mode，默认'a+'为追加写入
        fmt: Str. 日志输出的格式，详见python.logging.Formatter
        datefmt: Str. 日志输出格式中的时间格式，详见python.logging.Formatter
    """
    if self._built:
      return
    self._filename = filename
    self._detial = detail
    self._suffix = suffix
    self._filemode = filemode
    self._fmt = fmt
    self._datefmt = datefmt
    self._fullname = self._filename
    # Ensure the filename has suffix
    if self._fullname and self._fullname[len(suffix):] != suffix:
      self._fullname += suffix
    # Main Logger
    logger = logging.getLogger()
    if self._detial:
      logger.setLevel(logging.DEBUG)
    else:
      logger.setLevel(logging.INFO)
    # Remove Init StreamHandler
    logger.removeHandler(logger.handlers[0])
    # Formatter
    formatter = logging.Formatter(
        fmt=self._fmt,
        datefmt=self._datefmt)
    # File handler
    if self._fullname:
      handler = handlers.RotatingFileHandler(
          self._fullname,
          mode=self._filemode,
          maxBytes=2 ** 25,
          backupCount=1000)
      if self._detial:
        handler.setLevel(logging.DEBUG)
      else:
        handler.setLevel(logging.INFO)
      handler.setFormatter(formatter)
      logger.addHandler(handler)
    # Console Handler
    # TODO: use io.StreamIO
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    self._built = True

  def debug(self, msg, name=__name__, **kwargs):
    """Logger.debug

      DEBUG级别的日志

      Args:
        msg: Str. 日志内容
        name: Str, default __name__. 可指定名称
            传入`__name__`可用于定位日志的发送位置
        **kwargs: 参见logging.debug
      
      NOTE:
        若Logger未被初始化，则使用默认的Stream进行控制台打印
    """
    self._check_built()
    logging.getLogger(name).debug(msg, **kwargs)

  def info(self, msg, name=__name__, **kwargs):
    """Logger.info

      INFO级别的日志

      Args:
        msg: Str. 日志内容
        name: Str, default __name__. 可指定名称
            传入`__name__`可用于定位日志的发送位置
        **kwargs: 参见logging.info
      
      NOTE:
        若Logger未被初始化，则使用默认的Stream进行控制台打印
      
      Alias:
        log
        loghere
    """
    self._check_built()
    logging.getLogger(name).info(msg, **kwargs)

  log = info
  loghere = info

  def warning(self, msg, name=__name__, **kwargs):
    """Logger.warning

      WARNING级别的日志

      Args:
        msg: Str. 日志内容
        name: Str, default __name__. 可指定名称
            传入`__name__`可用于定位日志的发送位置
        **kwargs: 参见logging.warning
      
      NOTE:
        若Logger未被初始化，则使用默认的Stream进行控制台打印
      
      Alias:
        warn
    """
    self._check_built()
    logging.getLogger(name).warning(msg, **kwargs)

  warn = warning

  def error(self, msg, name=__name__, exit=False, **kwargs):
    """Logger.error

      ERROR级别的日志

      Args:
        msg: Str. 日志内容
        name: Str, default __name__. 可指定名称
            传入`__name__`可用于定位日志的发送位置
        exit: Bool, default False. 
            传入`True`可在记录日志后退出程序
        **kwargs: 参见logging.error
      
      NOTE:
        若Logger未被初始化，则使用默认的Stream进行控制台打印
    """
    self._check_built()
    logging.getLogger(name).error(msg, **kwargs)
    if exit:
      os._exit(0)

  def critical(self, msg, name=__name__, exit=False, **kwargs):
    """Logger.critical

      CRITICAL级别的日志

      Args:
        msg: Str. 日志内容
        name: Str, default __name__. 可指定名称
            传入`__name__`可用于定位日志的发送位置
        exit: Bool, default False. 
            传入`True`可在记录日志后退出程序
        **kwargs: 参见logging.critical
      
      NOTE:
        若Logger未被初始化，则使用默认的Stream进行控制台打印
      
      Alias:
        fatal
    """
    self._check_built()
    logging.getLogger(name).critical(msg, **kwargs)
    if exit:
      os._exit(0)

  fatal = critical

  def exception(self, msg='Exception Logged', name=__name__,
        exit=False, **kwargs):
    """Logger.exception

      记录异常日志，日志等级为ERROR

      Args:
        msg: Str, default 'Exception Logged'. 提示捕抓异常日志的语句
        name: Str, default __name__. 可指定名称
            传入`__name__`可用于定位日志的发送位置
        exit: Bool, default False. 
            传入`True`可在记录日志后退出程序
        **kwargs: 参见logging.exception
      
      NOTE:
        若Logger未被初始化，则使用默认的Stream进行控制台打印
    """
    self._check_built()
    logging.getLogger(name).exception(msg, **kwargs)
    if exit:
      os._exit(0)

  def batch(self, inputs, level='INFO', name=__name__, prefix='',
        suffix='', **kwargs):
    """Logger.batch

    批量输出日志

    Args:
      inputs: List of Str. 列表形式的日志内容
      level: Str, default 'INFO'. 日志等级，默认为INFO，大小写均可
          可选的等级有：
          DEBUG
          INFO
          WARNING
          ERROR
          CRITICAL
      name: Str, default __name__. 可指定名称
        传入`__name__`可用于定位日志的发送位置
      prefix: Str, default ''. 前缀日志内容
      suffix: Str, default ''. 后缀日志内容
      **kwargs: 参见logging.info

    NOTE:
      若Logger未被初始化，则使用默认的Stream进行控制台打印
    """    
    level_dict = {
        "DEBUG": self.debug,
        "INFO": self.info,
        "WARNING": self.warning,
        'ERROR': self.error,
        "CRITICAL": self.critical}
    # level = level.upper()
    try:
      log_func = level_dict[level.upper()]
    except KeyError:
      log_func = self.info
    for msg in inputs:
      log_func(prefix + msg + suffix, name=name, **kwargs)


logger = Logger()


def build(filename: str, detail=True, suffix='.log', filemode='a+',
      fmt="%(asctime)s.%(msecs)03d [%(levelname)s] >%(name)s: %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S"):
  """log.build
    
    日志工具初始化函数。

    Args:
      filename: Str. 日志文件名，包含路径。如果文件名不带有后缀名，会自动补上
          若文件名为空，则不输出日志文件
      detail: Bool. 日志写文件粒度。默认最低为DEBUG。
          若值为False，则写文件的最低等级为INFO
      suffix: Str. 日志文件后缀名，默认为'.log'
      filemode: Str. 日志文件的打开方式，同open.mode，默认'a+'为追加写入
      fmt: Str. 日志输出的格式，详见python.logging.Formatter
      datefmt: Str. 日志输出格式中的时间格式，详见python.logging.Formatter
  """
  logger.build(
      filename,
      detail=detail,
      suffix=suffix,
      filemode=filemode,
      fmt=fmt,
      datefmt=datefmt,)


def debug(msg, name=__name__, **kwargs):
  """log.debug

    DEBUG级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default __name__. 可指定名称
          传入`__name__`可用于定位日志的发送位置
      **kwargs: 参见logging.debug
      
    NOTE:
      若Logger未被初始化，则使用默认的Stream进行控制台打印
  """
  logger.debug(msg, name, **kwargs)


def info(msg, name=__name__, **kwargs):
  """log.info

    INFO级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default __name__. 可指定名称
          传入`__name__`可用于定位日志的发送位置
      **kwargs: 参见logging.info
      
    NOTE:
      若Logger未被初始化，则使用默认的Stream进行控制台打印
    
    Alias:
      log
      loghere
  """
  logger.info(msg, name, **kwargs)


log = info
loghere = info


def warning(msg, name=__name__, **kwargs):
  """log.warning

    WARNING级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default __name__. 可指定名称
          传入`__name__`可用于定位日志的发送位置
      **kwargs: 参见logging.warning
    
    NOTE:
      若Logger未被初始化，则使用默认的Stream进行控制台打印
    
    Alias:
      warn
  """
  logger.warning(msg, name, **kwargs)


warn = warning


def error(msg, name=__name__, **kwargs):
  """Logger.error

    ERROR级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default __name__. 可指定名称
          传入`__name__`可用于定位日志的发送位置
      exit: Bool, default False. 
          传入`True`可在记录日志后退出程序
      **kwargs: 参见logging.error
    
    NOTE:
      若Logger未被初始化，则使用默认的Stream进行控制台打印
  """
  logger.error(msg, name, **kwargs)


def critical(msg, name=__name__, **kwargs):
  """log.critical

    CRITICAL级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default __name__. 可指定名称
          传入`__name__`可用于定位日志的发送位置
      exit: Bool, default False. 
          传入`True`可在记录日志后退出程序
      **kwargs: 参见logging.critical
    
    NOTE:
      若Logger未被初始化，则使用默认的Stream进行控制台打印
    
    Alias:
      fatal
  """
  logging.critical(msg, name, **kwargs)


fatal = critical


def exception(msg='Exception Logged', name=__name__, exit=False, **kwargs):
  """log.exception

    记录异常日志，日志等级为ERROR

    Args:
      msg: Str, default 'Exception Logged'. 提示捕抓异常日志的语句
      name: Str, default __name__. 可指定名称
          传入`__name__`可用于定位日志的发送位置
      exit: Bool, default False. 
          传入`True`可在记录日志后退出程序
      **kwargs: 参见logging.exception
    
    NOTE:
      若Logger未被初始化，则使用默认的Stream进行控制台打印
  """
  logger.exception(msg, name, exit, **kwargs)


def batch(inputs, level='INFO', name=__name__, prefix='', suffix='',
      **kwargs):
  """Logger.batch

    批量输出日志

    Args:
      inputs: List of Str. 列表形式的日志内容
      level: Str, default 'INFO'. 日志等级，默认为INFO，大小写均可
          可选的等级有：
          DEBUG
          INFO
          WARNING
          ERROR
          CRITICAL
      name: Str, default __name__. 可指定名称
        传入`__name__`可用于定位日志的发送位置
      prefix: Str, default ''. 前缀日志内容
      suffix: Str, default ''. 后缀日志内容
      **kwargs: 参见logging.info

    NOTE:
      若Logger未被初始化，则使用默认的Stream进行控制台打印
  """ 
  logger.batch(inputs, level, name, prefix, suffix, **kwargs)   


if __name__ == "__main__":
  logger.loghere("test")
  logger.loghere("test")
  logger.build("unpush/test")
  logger.loghere("test1")
  logger.debug("test2")
  logger.batch(['test3', 'test4'], 'warning')
  logger.error('test5')
  logger.critical('test6', exit=True)
  logger.loghere("test7")

