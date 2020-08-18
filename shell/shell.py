# -*- coding: utf-8 -*-
"""Shell
  ======

  Deep Go Shell
"""


import argparse
from deepgo import __version__, __codename__, __release_date__
from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod
from deepgo.shell.request import VersionRequest


class AbstractShell(AbstractSingleton):
  """Abstract Shell Class

    This is a Abstract Singleton Class
  """
  @abstractmethod
  def parse(self): ...


class Shell(AbstractShell):
  """Deep Go Shell Class

    Handle with arguments

    This is a Singleton Class
  """
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument(
        "-V", "--version",
        help="print the Deep Go version number and exit (also --version)\n"
             "when given twice, print more information about the build",
        action="count")
    self.args = self.parser.parse_args()

  def parse(self):
    """Parse the command line arguments"""
    # print("Argument Parser")  # DEBUG
    if self.args.version:
      # self.parse_version()
      request = VersionRequest()
      # print("Get Request")  # DEBUG
      response = request.get(self.args.version)
      # print("Get Response")  # DEBUG
      # print("Parse Response")  # DEBUG
      print(response.message)
      # print("Exit")  # DEBUG
      return  # Exit

