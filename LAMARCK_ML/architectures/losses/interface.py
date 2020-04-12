from enum import Enum
from typing import Dict, Tuple, Set, List

from LAMARCK_ML.architectures.dataFlow import DataFlow
from LAMARCK_ML.architectures.losses.Loss_pb2 import LossProto
from LAMARCK_ML.data_util import IOLabel, TypeShape


class Reduce(Enum):
  MEAN = 0
  MININUM = 1
  MAXIMUM = 2
  SUM = 3
  PRODUCT = 4
  VARIANCE = 5
  STD = 6


class LossInterface(DataFlow):
  # TODO: is this class needed as DataFlow child class?
  IOLabel.PREDICTION = 'PREDICTION'
  arg_REDUCE = 'reduce'

  _DF_INPUTS = [IOLabel.PREDICTION, IOLabel.TARGET]
  _usedNames = dict()

  def __init__(self, **kwargs):
    super(LossInterface, self).__init__()
    self.reduction = kwargs.get(self.arg_REDUCE, Reduce.MEAN)
    if not isinstance(self.reduction, Reduce):
      self.reduction = Reduce(self.reduction)
    self._id_name = self.__getNewName(self)

  @classmethod
  def __getNewName(cls, obj=None) -> str:
    assert obj is not None and isinstance(obj, LossInterface)
    cls_name = cls.__name__
    idx = LossInterface._usedNames.get(cls_name, 0)
    name = cls_name + ':%09i' % (idx)
    idx += 1
    LossInterface._usedNames[cls_name] = idx
    LossInterface._usedNames[name] = obj
    return name

  def get_pb(self, result=None):
    if not isinstance(result, LossProto):
      result = LossProto()
    result.id_name = self._id_name
    result.cls_name = self.__class__.__name__
    result.reduction = self.reduction.value
    return result

  @staticmethod
  def getClassByName(cls_name: str):
    stack = [LossInterface]
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
      _loss = LossProto()
      _loss.ParseFromString(state)
    elif isinstance(state, LossProto):
      _loss = state
    else:
      return

    self._id_name = _loss.id_name
    self.__class__ = self.getClassByName(_loss.cls_name)
    self.reduction = Reduce(_loss.reduction)

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return True
    return False

  @property
  def outputs(self) -> Set[TypeShape]:
    return set()

  @property
  def inputs(self) -> Dict[IOLabel, Tuple[IOLabel, str]]:
    pass

  @property
  def id_name(self) -> str:
    return self._id_name

  @property
  def inputLabels(self) -> List[str]:
    return self._DF_INPUTS
