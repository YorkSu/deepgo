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
    self.parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False)
    self.group = self.parser.add_mutually_exclusive_group()
    self.parser.add_argument(
        "-h", "--help",
        help="打印帮助信息并退出",
        action="help")
    self.group.add_argument(
        "-V", "--version",
        help="输出版本号并退出\n如果输入两次则打印更多信息，包括版本代号",
        action="count")
    self.group.add_argument(
        "-P", "--project",
        help="生成一个完整的空白工程到指定路径")
    self.args = self.parser.parse_args()

  def parse(self):
    """Parse the command line arguments"""
    if self.args.version:
      interpreter.parse(f"version count={self.args.version} & exit")
    if self.args.project:
      interpreter.parse(f"project path={self.args.project} & exit")

