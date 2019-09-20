from LAMARCK_ML.architectures.variables.Variable_pb2 import VariableProto
from LAMARCK_ML.data_util import ProtoSerializable

RegularisationProto = VariableProto.RegularisationProto
from LAMARCK_ML.data_util.attribute import pb2attr, attr2pb


class Regularisation(ProtoSerializable):
  def __init__(self, **kwargs):
    self.attr = dict()

  def get_pb(self, result=None):
    if not isinstance(result, RegularisationProto):
      result = RegularisationProto()
    result.cls_name = self.__class__.__name__
    result.attr.extend([attr2pb(key, value) for key, value in self.attr.items()])
    return result

  def __eq__(self, other):
    if isinstance(other, self.__class__) and \
        len({k: self.attr.get(k) for k in self.attr if self.attr.get(k) == other.attr.get(k)}) \
        == len(self.attr) == len(other.attr):
      return True
    return False

  def __getstate__(self):
    return self.get_pb().SerializeToString()

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _regularisation = RegularisationProto()
      _regularisation.ParseFromString(state)
    elif isinstance(state, RegularisationProto):
      _regularisation = state
    else:
      return
    for sub_cls in Regularisation.__subclasses__():
      if sub_cls.__name__ == _regularisation.cls_name:
        self.__class__ = sub_cls
        break
    self.attr = dict([pb2attr(attr) for attr in _regularisation.attr])

  pass
