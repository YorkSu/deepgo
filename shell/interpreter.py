# -*- coding: utf-8 -*-
"""Interpreter
  ======

  Interpreter Class
"""


from typing import Tuple

from deepgo.core.pattern.singleton import AbstractSingleton, abstractmethod
from deepgo.shell.command import *


class Parser(AbstractSingleton):
  """Parser Interface

    This is an Abstract Singleton Class
  """
  @abstractmethod
  def parse(self, expression: str): ...


class AndParser(Parser):
  """And Parser Class

    Parse `&` expression to an expression list

    This is a Singleton Class
  """
  def parse(self, expression: str) -> Tuple[str, ...]:
    expression = expression.strip()
    expressions = expression.split('&')
    return expressions


class ElementParser(Parser):
  """Element Parser Class

    Split Expression by ` `(Space)

    This is a Singleton Class
  """
  def parse(self, expression: str) -> Tuple[str, ...]:
    expression = expression.strip()
    elements = expression.split(' ')
    return elements


class CommandParser(Parser):
  """Command Parser Class

    Parse Expression to Command

    This is a Singleton Class
  """
  def __init__(self):
    self.command_map = {
        "VERSION": VersionCommand(),
        "EXIT": ExitCommand(),
        "PROJECT": ProjectCommand(),}

  def parse(self, expression: str) -> Command:
    expression = expression.strip().upper()
    if expression in self.command_map:
      return self.command_map[expression]
    return None


class ArgumentParser(Parser):
  """Argument Parser Class

    Parse Elements to *args, **kwargs

    This is a Singleton Class
  """
  def parse(self, expression: Tuple[str, ...]) -> Tuple[list, dict, list]:
    args = []
    kwargs = {}
    error = []
    for element in expression:
      if element in ['', ' ']:
        continue
      if '=' in element:
        kw = element.split('=')
        if len(kw) == 2:
          kwargs[kw[0]] = kw[1]
        else:
          error.append(element)
      else:
        args.append(element)
    return args, kwargs, error


and_parser = AndParser()
element_parser = ElementParser()
command_parser = CommandParser()
argument_parser = ArgumentParser()


class Interpreter(Parser):
  """Interpreter Class

    This is a Singleton Class

    1) Parse Expression 
    2) Execute Command
    3) Parse the Response
  """
  def parse(self, expression: str):
    expressions = and_parser.parse(expression)
    for expression in expressions:
      elements = element_parser.parse(expression)
      cmd = elements[0]
      argument = elements[1:]
      command = command_parser.parse(cmd)
      args, kwargs, error = argument_parser.parse(argument)
      if command is None:
        print(f"Invalid Command: {elements[0]}")
        continue
      if error:
        for e in error:
          print(f"Unknown argument: {e}")
      response = command.execute(*args, **kwargs)
      print(response.message)


interpreter = Interpreter()


if __name__ == "__main__":
  interpreter.parse("version abc     def  ")
  interpreter.parse("version more")
  interpreter.parse("version count=0")
  interpreter.parse("version count=1")
  interpreter.parse("version count=2 & exit & test")

