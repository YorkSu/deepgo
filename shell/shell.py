# -*- coding: utf-8 -*-
"""Shell
  ======

  Deep Go Shell
"""


import argparse

from deepgo.shell.interpreter import Parser, interpreter


class Shell(Parser):
  """Deep Go Shell Class

    Handle with sys.args

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
    if self.args.version:
      interpreter.parse(f"version count={self.args.version} & exit")

