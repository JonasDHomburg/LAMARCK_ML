import numpy as np
from typing import Dict
from datetime import datetime
from random import randint, seed

from LAMARCK_ML.data_util import ProtoSerializable
from LAMARCK_ML.data_util.attribute import pb2attr, attr2pb
from LAMARCK_ML.individuals.Individual_pb2 import IndividualProto

seed()


class IndividualInterface(ProtoSerializable):
  arg_METRICS = 'metrics'
  arg_LOSSES = 'losses'
  arg_NAME = 'name'

  def __init__(self, **kwargs):
    super(IndividualInterface, self).__init__()
    self.metrics = dict()
    self._fitness = None
    self._id_name = kwargs.get(self.arg_NAME)
    if self._id_name is None:
      self._id_name = self.getNewName()

    self.attr = dict()

  @classmethod
  def getNewName(cls):
    """
    Generates a new individual id based on datetime and a random value to avoid collisions.
    :return: new individual id (str)
    """
    name = cls.__name__ + '_' + str(datetime.now().timestamp()) + '_%09i' % randint(0, 1e9 - 1)
    return name

  @property
  def id_name(self):
    return self._id_name

  @property
  def fitness(self):
    """
    A scalar quality value.
    :return: The fitness value f if assigned otherwise the calculated value f = \frac{sum_{m\in M}(1-m)^2}{|M|}
    """
    if hasattr(self, '_fitness') and self._fitness:
      return self._fitness
    elif self.metrics:
      try:
        self._fitness = np.sum((np.ones(len(self.metrics)) - np.asarray(self.metrics.values())) ** 2) / len(
          self.metrics)
      except:
        return -.1
      return self._fitness
    else:
      return -.1

  def __gt__(self, other):
    """
    Determines whether this individual is better than the 'other' individual in terms of higher fitness. To allow multi object optimization this should be
      overwritten and a respective multi dimensional comparison be used.
    :param other: The other individual this individual is compared with
    :return: True if other is instance of IndividualInterface and self is better than other else False
    """
    if isinstance(other, IndividualInterface):
      return self.fitness > other.fitness
    return False

  def __sub__(self, other):
    """
    A directed distance between individuals. Currently unused (will be used for Particle Swarm in the future).
    :param other: The other individual.
    :return: Directed distance from other to self.
    """
    raise NotImplementedError()

  def get_pb(self, result=None):
    """
    Convert the in individual into a protobuf object.
    :param result: optional result protobuf object to modify
    :return: protobuf object (IndividualProto)
    """
    if not isinstance(result, IndividualProto):
      result = IndividualProto()
    result.cls_name = self.__class__.__name__
    result.id_name = self._id_name
    result.metrics.extend(
      [IndividualProto.MetricProto(id_name=key, value=value) for key, value in self.metrics.items()])
    result.attr.extend([attr2pb(key, value) for key, value in self.attr.items()])
    return result

  @staticmethod
  def getClassByName(cls_name: str):
    """
    .. _getClassByName:
    Searcing for class reference by given string based on class inheritance.
    :param cls_name: class inheritance separated by '-'
    :return: class reference
    :raises: Exception if the class is not found
    """
    stack = [IndividualInterface]

    while stack:
      cls = stack.pop()
      if cls.__name__ == cls_name:
        return cls
      stack.extend(cls.__subclasses__())
    raise Exception("Couldn't find class with name: " + cls_name)

  def __getstate__(self):
    return self.get_pb().SerializeToString()

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _individual = IndividualProto()
      _individual.ParseFromString(state)
    elif isinstance(state, IndividualProto):
      _individual = state
    else:
      try:
        self.__setstate__(state.SerializeToString())
      except Exception:
        pass
      return

    self.__class__ = self.getClassByName(_individual.cls_name)
    self._cls_setstate(state)

  def __eq__(self, other):
    if (isinstance(other, self.__class__)
        and self._id_name == other._id_name
        and self.metrics == other.metrics
        and len(self.attr) == len(other.attr) == len(
          [value == other.attr.get(key) for key, value in self.attr.items()])
    ):
      return True
    return False

  def _cls_setstate(self, state):
    """
    Individual specific reconstruction from ProtobufIndividual object or serialised ProtobufIndividual object.
    For class dependent attribute retrieval override this method!
    :param state: optional result ProtobufIndividual object to modify; if None, one is created
    :return: ProtobufIndividual object
    """
    if isinstance(state, str) or isinstance(state, bytes):
      _individual = IndividualProto()
      _individual.ParseFromString(state)
    elif isinstance(state, IndividualProto):
      _individual = state
    else:
      return
    self._id_name = _individual.id_name
    self.metrics = dict([(m.id_name, m.value) for m in _individual.metrics])
    self.attr = dict([pb2attr(attr) for attr in _individual.attr if attr.name != self.arg_METRICS])

  def norm(self, other):
    """
    A norm d(self, other) for individuals.
    :param other: The other individual.
    :return: Distance between self and other.
    """
    raise NotImplementedError()

  @staticmethod
  def get_instance(state):
    """
    Convert a (serialised) protobuf object into a individual object.
    :param state: (serialised) protobuf object
    :return: individual object
    """
    if isinstance(state, str) or isinstance(state, bytes):
      _function = IndividualProto()
      _function.ParseFromString(state)
    elif isinstance(state, IndividualProto):
      _function = state
    else:
      return
    result = IndividualInterface.__new__(IndividualInterface)
    result.__setstate__(state)
    return result

  def update_state(self, *args, **kwargs):
    raise NotImplementedError()

  def build_instance(self, nn_framework):
    raise NotImplementedError()

  def train_instance(self, nn_framework) -> Dict:
    raise NotImplementedError()

  pass
