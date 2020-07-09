# -*- coding: utf-8 -*-
"""IO
  ======

  Deep Go IO
"""


import os
import sys
import json


class Console(object):
  """Console IO"""
  def __init__(self, *args, **kwargs):
    self._output_list = []
    self._stdout = sys.stdout
    del args, kwargs
  
  def print(self, *args, **kwargs):
    """Console.print"""
    print(*args, **kwargs)

  def input(self, *args, **kwargs):
    """Console.input"""
    return input(*args, **kwargs)

  def clean(self):
    """Console.clean"""
    os.system("cls")

  def progress(self, value, end=False):
    """Console.progress"""
    self.print('\r'+value, end="\n" if end else "", flush=True)

  def hook(self, value: bool):
    """Console.hook"""
    sys.stdout = self if value else self._stdout

  def write(self, str):
    self._output_list.append(str)
    self._stdout.write(str)

  def show(self):
    print(self._output_list)
    self._stdout.write(str(self._output_list) + '\n')

  def set_control(self, value: bool):
    """Console.set_stdout"""
    sys.stdout = self if value else self._stdout

  def flush(self):
    self._stdout.flush()


class File(object):
  """File IO"""
  def __init__(self, filename):
    self._filename = filename
    self._content = []
  
  def add(self, text):
    if text[-1] != '\n':
      text += '\n'
    self._content.append(text)
    with open(self._filename, 'a+') as f:
      f.write(text)

  def load(self):
    with open(self._filename, 'r') as f:
      self._content = f.readlines()

  def save(self):
    with open(self._filename, 'w') as f:
      f.writelines(self._content)


class Json(object):
  """Json IO"""
  def __init__(self, filename, *args, **kwargs):
    self._filename = filename
    self._dict = {}
    self._created = False
    self.load()
    del args, kwargs

  @property
  def created(self):
    """Json.created

      Return a boolean indicating if the json is created.
    """
    return self._created

  @property
  def dict(self):
    """Json.dict

      Return the dict of the json.
    """
    return self._dict

  def get(self, key):
    """Json.get

      Return the value of the specified key.
      If the key does not exist, return None.
    """
    return self._dict.get(key)

  def load(self):
    """Json.load
    
      Load the filename.json.
    """
    if os.path.exists(self._filename):
      try:
        self._dict = json.load(open(self._filename))
        self._created = True
      except Exception:
        pass

  def merge(self, other):
    """Json.merge

      Merge this json with the other json.
    """
    if not isinstance(other, Json):
      return
    self._dict.update(other.dict)

  def save(self):
    """Json.save
    
      Save the filename.json.
    """
    json.dump(self._dict, open(self._filename, 'w'), indent=2)

  def set(self, key, value):
    """Json.set

      Set the value of the specified key.
    """
    self._dict[key] = value
  

console = Console()


if __name__ == "__main__":
  # ========
  # File
  print("123\n"[-1] == '\n')
  # ========
  
  
  # ========
  # Json
  # js = Json("unpush/test.json")
  # print(js.dict)
  # print(js.get("data"))
  # js.set("data2", "TensorFlow")
  # print(js.dict)
  # js2 = Json('')
  # print(js2.dict)
  # js2.set("json2", "Keras")
  # js.merge(js2)
  # js.merge({"wrong_dict": "1"})
  # print(js.dict)
  # js.save()
  # ========

