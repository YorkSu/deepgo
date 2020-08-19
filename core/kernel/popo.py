# -*- coding: utf-8 -*-
"""POPO
  ======

  Plain Ordinary Python Object
"""


from deepgo.core.pattern.singleton import Singleton


class VO(object):
  """Value Object"""
  def get(self, key):
    """Get the value of an argument"""
    return self.__getattribute__(key)

  def set(self, key, value):
    """Set the value of an argument"""
    self.__setattr__(key, value)

  def keys(self):
    """Get the list of argument keys"""
    return list(self.__dict__.keys())


class FVO(object):
  """Parent Class for a Field Value Object

    Define Value by define the `class.fields`
      key: default value
      key must be a string

    Example:
    ```python
    class StudentFVO(FVO):
      fields = {'name': ''}, 'age': 0}
    ```

    SetValue:
    ```python
    # Way 1
    s = StudentFVO()
    s.name = 'John'
    # Way 2
    s.set('name', 'John')
    # Way 3
    s = StudentFVO(name='John')
    ```

    GetValue:
    ```python
    # Way 1
    name = s.name
    # Way 2
    name = s.get('name')
    ```
  """
  fields = {}
  def __init__(self, *args, **kwargs):
    for key in self.__class__.fields:
      if key in kwargs:
        value = kwargs[key]
      else:
        value = self.__class__.fields[key]
      self.__setattr__(key, value)
    del args, kwargs

  def __setattr__(self, name, value):
    if name in self.__class__.fields:
      super(FVO, self).__setattr__(name, value)
    
  def get(self, name):
    return self.__getattribute__(name)

  def set(self, name, value):
    self.__setattr__(name, value)


class Converter(Singleton):
  """Title

    This is a Singleton Class
  """
  def dict2vo(self, inputs) -> VO:
    output = VO()
    if inputs is None:
      return output
    if not isinstance(inputs, dict):
      return inputs
    for key in inputs:
      if isinstance(inputs[key], dict):
        output.set(key, self.dict2vo(inputs[key]))
      elif isinstance(inputs[key], list):
        output.set(key, self.list2vo(inputs[key]))
      else:
        output.set(key, inputs[key])
    return output

  def list2vo(self, inputs) -> list:
    if not isinstance(inputs, list):
      return inputs
    output = []
    for item in inputs:
      if isinstance(item, list):
        output.append(self.list2vo(item))
      elif isinstance(item, dict):
        output.append(self.dict2vo(item))
      else:
        output.append(item)
    return output


converter = Converter()


if __name__ == "__main__":
  class StudentFVO(FVO):
    fields = {'name': '', 'age': 0}
  s1 = StudentFVO()
  print(s1.name, s1.age)
  s2 = StudentFVO(name='John', age=31)
  print(s2.name, s2.age)
  a1 = VO()
  a1.name = 'Lennon'
  print(a1.name)
  print(a1.keys())
  # print(a1._dict)
  print(a1.__dict__)
  d = {
    'name': 'John',
    'score': [99, 100, 70, 40],
    'location': {
      'x': 100,
      'y': 50
    },
    'li': [
      {'x1': 33},
      {"y1": 98}
    ]
  }
  # print(d)
  c = converter.dict2vo(d)
  print(c)
  print(c.keys())
  print(c.name)
  print(c.score)
  print(c.location)
  print(c.location.x, c.location.y)
  print(c.li)
  print(c.li[0].x1, c.li[1].y1)

