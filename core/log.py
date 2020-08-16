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
  """Log Manager"""
  built = False
  init_warnings = False
  _store = []

  @staticmethod
  def assert_built():
    """Assert Logger is built"""
    if Logger.built:
      return True
    if not Logger.init_warnings:
      logging.basicConfig(level=logging.DEBUG)
      print(", ".join([
          "[WARNING] Logger has not been built",
          "this message will only log once."]))
      Logger.init_warnings = True
    return False

  @staticmethod
  def build(
        filename="",
        file_level=logging.DEBUG,
        stream_level=logging.INFO,
        suffix='.log',
        mode='a+',
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] >%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        maxBytes=2 ** 24,
        backupCount=1000):
    """Initialize the log tool

      Args:
        filename: Str. The log filename
            If not have a suffix, will auto replenished
            If "", will not output the log file
        file_level: Integer. The log file level
            file_level must less or equal to stream_level
        stream_level: Integer. The log stream level
        suffix: Str. the log file suffix, default '.log'
        mode: Str. the log file mode
        fmt: Str. the log format, see python.logging.Formatter
        datefmt: Str. the log date format, see python.logging.Formatter
        maxBytes: Integer. The maximum size of the log file
        backupCount: Integer. The maximun backup file count.
    """
    if file_level > stream_level:
      print(", ".join([
        f"[WARNING]",
        f"excepted file_level <= stream_level"
        f"got {file_level}, {stream_level}"]))
      return
    if filename:
      _path, _file = os.path.split(filename)
      _name, _ext = os.path.splitext(_file)
      if not os.path.exists(_path):
        os.makedirs(_path)
      if _ext != suffix:
        filename = os.path.join(_path, _name) + suffix
    logger = logging.getLogger()
    for handler in logger.handlers:
      logger.removeHandler(handler)
    logger.setLevel(file_level)
    formatter = logging.Formatter(
        fmt=fmt,
        datefmt=datefmt)
    if filename:
      handler = handlers.RotatingFileHandler(
          filename,
          mode=mode,
          maxBytes=maxBytes,
          backupCount=backupCount)
      handler.setLevel(file_level)
      handler.setFormatter(formatter)
      logger.addHandler(handler)
    # Stream Handler
    stream = logging.StreamHandler()
    stream.setLevel(stream_level)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    Logger.built = True

  @staticmethod
  def base_function(level: str, name=__name__):
    """Base Log Function

      Get the log function based on the level and name

      Args:
        level: Str. The log level
        name: Str. The log name
            Default __name__, the log location

      Returns:
        python.function: log_func
    """
    Logger.assert_built()
    level = level.upper()
    _Logger = logging.getLogger(name)
    level_to_function = {
        "DEBUG": _Logger.debug,
        "INFO": _Logger.info,
        "LOG": _Logger.info,
        "WARNING": _Logger.warning,
        "WARN": _Logger.warning,
        "ERROR": _Logger.error,
        "CRITICAL": _Logger.critical,
        "FATAL": _Logger.critical,
        "EXCEPTION": _Logger.exception}
    level_to_exit = [
        "ERROR",
        "CRITICAL",
        "FATAL",
        "EXCEPTION"]
    def log_func(
        msg: str,
        exit=False,
        **kwargs):
      """The actual logging function

        Args:
          msg: Str. The log message
          exit: Boolean. If True, exit the program after log
          **kwargs: other keyword arguments
      """
      level_to_function[level](msg, **kwargs)
      if level in level_to_function and exit:
        os._exit(0)
    return log_func

  @staticmethod
  def batch(
      inputs,
      level="INFO",
      name=__name__,
      perfix="",
      suffix="",
      **kwargs):
    """Log with a batch of messages

      Args:
        inputs: a List of Str. The log messages
        level: Str. The log level
        name: Str. The log name
            Default __name__, the log location
        perfix: Str. The prefix for the log messages
        suffix: Str. The suffix for the log messages
        **kwargs: other keyword arguments
    """
    level = level.upper()
    try:
      func = Logger.base_function(level, name)
      for msg in inputs:
        func(f"{perfix}{msg}{suffix}", **kwargs)
    except KeyError:
      Logger.base_function("WARNING", ", ".join([
          f"WARNING",
          f"excepted level {except_level}",
          f"got {level}"]))


def debug(msg, name=__name__, **kwargs):
  """Log with DEBUG level

    Args:
      msg: Str. The log message
      name: Str. The log name
          Default __name__, the log location
      **kwargs: other keyword arguments
  """
  Logger.base_function("DEBUG", name=name)(msg, **kwargs)


def info(msg, name=__name__, **kwargs):
  """Log with INFO level

    Args:
      msg: Str. The log message
      name: Str. The log name
          Default __name__, the log location
      **kwargs: other keyword arguments
  """
  Logger.base_function("INFO", name=name)(msg, **kwargs)


log = info


def warning(msg, name=__name__, **kwargs):
  """Log with WARNING level

    Args:
      msg: Str. The log message
      name: Str. The log name
          Default __name__, the log location
      **kwargs: other keyword arguments
  """
  Logger.base_function("WARNING", name=name)(msg, **kwargs)


warn = warning


def error(msg, name=__name__, exit=False, **kwargs):
  """Log with ERROR level

    Args:
      msg: Str. The log message
      name: Str. The log name
          Default __name__, the log location
      exit: Boolean. If True, exit the program after log
      **kwargs: other keyword arguments
  """
  Logger.base_function("ERROR", name=name)(msg, exit=exit, **kwargs)


def critical(msg, name=__name__, exit=False, **kwargs):
  """Log with CRITICAL level

    Args:
      msg: Str. The log message
      name: Str. The log name
          Default __name__, the log location
      exit: Boolean. If True, exit the program after log
      **kwargs: other keyword arguments
  """
  Logger.base_function("CRITICAL", name=name)(msg, exit=exit, **kwargs)
  

fatal = critical


def exception(msg='Exception Logged', name=__name__, exit=False, **kwargs):
  """Log with Exception

    Args:
      msg: Str. The log message
      name: Str. The log name
          Default __name__, the log location
      exit: Boolean. If True, exit the program after log
      **kwargs: other keyword arguments
  """
  Logger.base_function("EXCEPTION", name=name)(msg, exit=exit, **kwargs)


base_function = Logger.base_function
build = Logger.build
batch = Logger.batch


# Test Part
if __name__ == "__main__":
  base_function("debug")("hello")
  base_function("info")("hello")
  build('./unpush/')
  debug("Log with DEBUG")
  info("Log with INFO")
  warning("Log with WARNING")
  error("Log with ERROR")
  critical("Log with CRITICAL")
  batch(["Log batch 1", "Log batch 2"], level="INFO", name='app')
  error("Log with exit, you can see the next log", exit=True)
  info("You can not see this log")
  # Stream Out
  # ================
  # [WARNING] Logger has not been built, this message will only log once.
  # DEBUG:__main__:hello
  # INFO:__main__:hello
  # 2020-07-27 22:05:10.300 [INFO] >__main__: Log with INFO
  # 2020-07-27 22:05:10.301 [WARNING] >__main__: Log with WARNING
  # 2020-07-27 22:05:10.302 [ERROR] >__main__: Log with ERROR
  # 2020-07-27 22:05:10.303 [CRITICAL] >__main__: Log with CRITICAL
  # 2020-07-27 22:05:10.304 [INFO] >app: Log batch 1
  # 2020-07-27 22:05:10.304 [INFO] >app: Log batch 2
  # 2020-07-27 22:05:10.305 [ERROR] >__main__: Log with exit, you can see the next log
  # File Out
  # =================
  # 2020-07-27 22:09:07.792 [DEBUG] >__main__: Log with DEBUG
  # 2020-07-27 22:09:07.792 [INFO] >__main__: Log with INFO
  # 2020-07-27 22:09:07.793 [WARNING] >__main__: Log with WARNING
  # 2020-07-27 22:09:07.794 [ERROR] >__main__: Log with ERROR
  # 2020-07-27 22:09:07.795 [CRITICAL] >__main__: Log with CRITICAL
  # 2020-07-27 22:09:07.796 [INFO] >app: Log batch 1
  # 2020-07-27 22:09:07.809 [INFO] >app: Log batch 2
  # 2020-07-27 22:09:07.810 [ERROR] >__main__: Log with exit, you can see the next log

