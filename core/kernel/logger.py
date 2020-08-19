# -*- coding: utf-8 -*-
"""Logger
  ======

  Logger Class
"""


import os
import logging
from logging import handlers

from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod
from deepgo.core.kernel.flags import flags


class AbstractLogger(AbstractSingleton):
  """Abstract Logger Class

    This is a Abstract Singleton Class
  """
  @abstractmethod
  def build(self): ...

  @abstractmethod
  def base(self, level): ...

  @abstractmethod
  def batch(self, inputs, level): ...

  @abstractmethod
  def debug(self, msg): ...

  @abstractmethod
  def info(self, msg): ...

  @abstractmethod
  def warning(self, msg): ...

  @abstractmethod
  def error(self, msg): ...

  @abstractmethod
  def critical(self, msg): ...

  @abstractmethod
  def exception(self, msg): ...


class Logger(AbstractLogger):
  """Logger Class

    This is a Singleton Class
  """
  def __init__(self):
    # Init Logger
    logging.basicConfig(level=logging.WARNING)
    # Build Logger
    self.build()

  def build(self):
    if flags.logging is None:
      return

    conf = flags.logging
    filename = conf['filename']
    suffix = conf['suffix']
    file_level = conf['file_level']
    stream_level = conf['stream_level']
    mode = conf['mode']
    maxBytes = conf['maxBytes']
    backupCount = conf['backupCount']
    fmt = conf['fmt']
    datefmt = conf['datefmt']

    if filename:
      _path, _file = os.path.split(filename)
      _name, _ext = os.path.splitext(_file)
      if not os.path.exists(_path):
        os.makedirs(_path)
      if _ext != suffix:
        filename = os.path.join(_path, _name) + suffix
    
    root = logging.getLogger()
    for handler in root.handlers:
      root.removeHandler(handler)
    root.setLevel(logging.NOTSET)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if filename:
      handler = handlers.RotatingFileHandler(
          filename,
          mode=mode,
          maxBytes=maxBytes,
          backupCount=backupCount)
      handler.setLevel(file_level)
      handler.setFormatter(formatter)
      root.addHandler(handler)

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if filename and file_level > stream_level:
      self.warning(f"Expected file_level <= stream_level, "
                   f"got {file_level}, {stream_level}")

  def base(self, level: str, name=__name__):
    """Title

      Description
    """
    level = level.upper()
    _Logger = logging.getLogger(name)
    level2func = {
        "DEBUG": _Logger.debug,
        "INFO": _Logger.info,
        "LOG": _Logger.info,
        "WARNING": _Logger.warning,
        "WARN": _Logger.warning,
        "ERROR": _Logger.error,
        "CRITICAL": _Logger.critical,
        "FATAL": _Logger.critical,
        "EXCEPTION": _Logger.exception}
    level2exit = [
        "ERROR",
        "CRITICAL",
        "FATAL",
        "EXCEPTION"]
    def logfunc(msg: str, exit=False, **kwargs):
      """Title

        Description
      """
      level2func[level](msg, **kwargs)
      if level in level2exit and exit:
        os._exit(0)
    return logfunc

  def batch(self, inputs, level="INFO", name=__name__,
      perfix='', suffix='', **kwargs):
    """Title

      Description
    """
    level = level.upper()
    try:
      logfunc = self.base(level, name)
      for msg in inputs:
        logfunc(f"{perfix}{msg}{suffix}", **kwargs)
    except KeyError:
      self.base("WARNING", f"Unknown level {level}")

  def debug(self, msg, name=__name__, **kwargs):
    """Title

      Description
    """
    self.base("DEBUG", name=name)(msg, **kwargs)

  def info(self, msg, name=__name__, **kwargs):
    """Title

      Description
    """
    self.base("INFO", name=name)(msg, **kwargs)

  log = info

  def warning(self, msg, name=__name__, **kwargs):
    """Title

      Description
    """
    self.base("WARNING", name=name)(msg, **kwargs)

  warn = warning

  def error(self, msg, name=__name__, exit=False, **kwargs):
    """Title

      Description
    """
    self.base("ERROR", name=name)(msg, exit=exit, **kwargs)

  def critical(self, msg, name=__name__, exit=False, **kwargs):
    """Title

      Description
    """
    self.base("CRITICAL", name=name)(msg, exit=exit, **kwargs)

  fatal = critical

  def exception(self, msg="Exception Logged", name=__name__, exit=False,
      **kwargs):
    """Title

      Description
    """
    self.base("EXCEPTION", name=name)(msg, exit=exit, **kwargs)


logger = Logger()


if __name__ == "__main__":
  # flags.logging = {
  #     'filename': 'unpush/',
  #     'suffix': '.log',
  #     'file_level': 10,
  #     'stream_level': 20,
  #     'mode': 'a+',
  #     'maxBytes': 2** 24,
  #     'backupCount': 100,
  #     'fmt': "%(asctime)s.%(msecs)03d [%(levelname)s] >%(name)s: %(message)s",
  #     'datefmt': "%Y-%m-%d %H:%M:%S",}
  logger.base('debug')('hello')
  logger.base('info')('hello')
  logger.build()
  logger.debug("Log with DEBUG")
  logger.info("Log with INFO")
  logger.info("Log with INFO", name='deepgo')
  logger.warning("Log with WARNING")
  logger.error("Log with ERROR")
  logger.critical("Log with CRITICAL")
  logger.batch(["Log with BATCH 1", "Log with BATCH 2"])
  try:
    x = 1 / 0
  except Exception as e:
    logger.error(e)
    logger.exception()
  

