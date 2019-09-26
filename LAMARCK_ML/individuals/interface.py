import numpy as np
from datetime import datetime
from random import randint, seed

from LAMARCK_ML.architectures import NeuralNetwork
from LAMARCK_ML.architectures.losses import LossInterface
from LAMARCK_ML.data_util import ProtoSerializable
from LAMARCK_ML.data_util.attribute import pb2attr, attr2pb
from LAMARCK_ML.data_util.typeShape import TypeShape
from LAMARCK_ML.individuals.Individual_pb2 import IndividualProto

seed()

class IndividualInterface(ProtoSerializable):
  arg_NEURAL_NETWORKS = 'neuralNetworks'  # TODO: change to more general Architecture
  arg_DATA_NTS = 'data_nts'
  arg_METRICS = 'metrics'
  arg_LOSSES = 'losses'
  arg_NAME = 'name'

  # _nameIdx = 0

  def __init__(self, **kwargs):
    self.metrics = dict()
    self._fitness = None
    self._networks = kwargs.get(self.arg_NEURAL_NETWORKS, [])
    for idx, network in enumerate(self._networks):
      if not isinstance(network, NeuralNetwork):
        raise Exception(
          'False network provided! Expected instance of NeuralNetwork, got: ' + str(
            type(network)) + ' at idx: ' + str(idx))
    self._data_nts = kwargs.get(self.arg_DATA_NTS, dict())

    if not isinstance(self._data_nts, dict):
      raise Exception('Expected dict for ' + self.arg_DATA_NTS + ' but got ' + str(type(self._data_nts)))
    for label, (nts, data_set_id) in self._data_nts.items():
      if not (isinstance(data_set_id, str)
              and isinstance(nts, TypeShape)
              and isinstance(label, str)
              and label):
        raise Exception('Expected list of (TypeShape,id_name) but got: (' + str(type(nts)) + ', ' + str(
          type(data_set_id)) + ')')
    self._losses = kwargs.get(self.arg_LOSSES, [])
    for idx, loss in enumerate(self._losses):
      if not isinstance(loss, LossInterface):
        raise Exception(
          'False loss provided! Expected instance of LossInterface, got: ' + str(type(loss)) + ' at idx: ' + str(idx))
    self._id_name = kwargs.get(self.arg_NAME)
    if self._id_name is None:
      self._id_name = self.getNewName()

    self.attr = dict()

  @classmethod
  def getNewName(cls):
    # name = cls.__name__ + ':%09i' % (cls._nameIdx)
    # cls._nameIdx += 1
    name = cls.__name__ + '_' + str(datetime.now().timestamp()) + '_%09i'%randint(0, 1e9-1)
    return name

  @property
  def id_name(self):
    return self._id_name

  @property
  def fitness(self):
    if hasattr(self, '_fitness') and self._fitness:
      return self._fitness
    elif self.metrics:
      try:
        self._fitness = (np.ones(len(self.metrics)) - np.asarray(self.metrics.values())) ** 2
      except:
        return -.1
      return self._fitness
    else:
      return -.1

  def __gt__(self, other):
    if isinstance(other, IndividualInterface):
      return self.fitness > other.fitness
    return False

  def __sub__(self, other):
    raise NotImplementedError()

  def get_pb(self, result=None):
    if not isinstance(result, IndividualProto):
      result = IndividualProto()
    result.cls_name = self.__class__.__name__
    result.id_name = self._id_name

    result.metrics.extend(
      [IndividualProto.MetricProto(id_name=key, value=value) for key, value in self.metrics.items()])
    result.networks.extend([network.get_pb() for network in self._networks])
    result.data_sources.extend(
      [IndividualProto.DataSourceProto(id_name=id_name, label=label, tsp=nts.get_pb())
       for label, (nts, id_name) in self._data_nts.items()])
    result.losses.extend([loss.get_pb() for loss in self._losses])

    result.attr.extend([attr2pb(key, value) for key, value in self.attr.items()])
    return result

  @staticmethod
  def getClassByName(cls_name: str):
    stack = [IndividualInterface]
    while stack:
      cls = stack.pop(0)
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
      return

    self.__class__ = self.getClassByName(_individual.cls_name)
    self._id_name = _individual.id_name

    self.metrics = dict([(m.id_name, m.value) for m in _individual.metrics])
    self._networks = list()
    for network in _individual.networks:
      _obj = NeuralNetwork.__new__(NeuralNetwork)
      _obj.__setstate__(network)
      self._networks.append(_obj)
    self._data_nts = dict([(d.label, (TypeShape.from_pb(d.tsp), d.id_name)) for d in _individual.data_sources])
    self._losses = list()
    for loss in _individual.losses:
      _obj = LossInterface.__new__(LossInterface)
      _obj.__setstate__(loss)
      self._losses.append(_obj)

    self._cls_setstate(_individual)

  def _cls_setstate(self, _individual):
    self.attr = dict([pb2attr(attr) for attr in _individual.attr if attr.name != self.arg_METRICS])

  def norm(self, other):
    raise NotImplementedError()

  @staticmethod
  def get_instance(state):
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

  pass
