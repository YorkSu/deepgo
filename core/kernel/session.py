# -*- coding: utf-8 -*-
"""Session
  ======

  Core Session
"""


import abc
import threading


class SessionMetaclass(type):
  """Metaclass for defining Session Classes"""
  __instance_lock = threading.RLock()
  __instance = {}
  def __call__(cls, *args, session='', **kwargs):
    if not session:
      session = 'root'
    if session not in cls.__instance:
      with cls.__instance_lock:
        if session not in cls.__instance:
          cls.__instance[session] = super(SessionMetaclass, cls).__call__(session=session, *args, **kwargs)
    return cls.__instance[session]


class AbstractSessionMetaclass(abc.ABCMeta, SessionMetaclass): ...


class AbstractSession(metaclass=AbstractSessionMetaclass):
  """Abstract Session Class"""
  @property
  @abc.abstractmethod
  def session(self): ...

  @staticmethod
  @abc.abstractmethod
  def get(session): ...


class Session(AbstractSession):
  """Session"""
  def __init__(self, session=''):
    self._lock = threading.RLock()
    self.__session = session

  @property
  def session(self):
    return self.__session

  @staticmethod
  def get(session=''):
    return Session(session=session)


if __name__ == "__main__":
  # sess = SessionCache.get_session()
  # sess.name = 'John'
  # print(sess.name)
  # sess2 = SessionCache.get_session()
  # print(sess2.name)
  # sess3 = SessionCache.get_session('test')
  # print(sess3.name)
  # sess.name = 'Lennon'
  # print(sess.name, sess2.name, sess3.name)

  # sess1 = Session()
  # sess1.name = 'John'
  # sess2 = Session()
  # sess3 = Session(session='test')
  # print(sess1.name, sess2.name, sess3.name)
  # sess1.name = 'Paul'
  # print(sess1.name, sess2.name, sess3.name)

  sess1 = Session()
  sess1.name = 'John'
  sess2 = Session()
  sess3 = Session.get(session='test')
  sess3.name = 'Test'
  print(sess1.name, sess2.name, sess3.name)
  print(sess1.session, sess2.session, sess3.session)
  sess1.name = 'Paul'
  print(sess1.name, sess2.name, sess3.name)
  pass

