from typing import NamedTuple, List

from LAMARCK_ML.data_util.protoInterface import ProtoSerializable
from LAMARCK_ML.reproduction.Ancestry_pb2 import AncestryProto


class AncestryEntity(NamedTuple, ProtoSerializable):
  method: str
  descendant: str
  ancestors: List[str]

  def get_pb(self, result=None):
    if not isinstance(result, AncestryProto):
      result = AncestryProto()
    result.method = self.method
    result.descendant = self.descendant
    result.ancestors.extend(self.ancestors)
    return result

  def __getstate__(self):
    return self.get_pb().SerializeToString()

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _anc = AncestryProto()
      _anc.ParseFromString(state)
    elif isinstance(state, AncestryProto):
      _anc = state
    else:
      return

    self.method = _anc.method
    self.descendant = _anc.descendant
    self.ancestors = [id_name for id_name in _anc.ancestors]

  @classmethod
  def from_pb(cls, anc_proto):
    return AncestryEntity(anc_proto.method,
                          anc_proto.descendant,
                          [id_name for id_name in anc_proto.ancestors])
