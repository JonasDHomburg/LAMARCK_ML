from typing import NamedTuple, Type

from LAMARCK_ML.data_util import Shape, BaseType
from LAMARCK_ML.data_util.TypeShape_pb2 import TypeShapeProto
from LAMARCK_ML.data_util.protoInterface import ProtoSerializable


class IOLabel():
  DEFAULT = 'DEFAULT'


class TypeShape(NamedTuple, ProtoSerializable):
  dtype: Type[BaseType]
  shape: Shape

  def get_pb(self, result=None):
    if not isinstance(result, TypeShapeProto):
      result = TypeShapeProto()
    self.dtype.get_pb(result.dtype_val)
    self.shape.get_pb(result.shape_val)
    return result

  def __getstate__(self):
    return self.get_pb().SerializeToString()

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _proto_ob = TypeShapeProto()
      _proto_ob.ParseFromString(state)
    elif isinstance(state, TypeShapeProto):
      _proto_ob = state
    else:
      return
    self.dtype = BaseType.pb2cls(_proto_ob.dtype_val)
    self.shape = Shape.__new__(Shape)
    self.shape.__setstate__(_proto_ob.shape_val)

  @classmethod
  def from_pb(cls, nts_proto):
    dtype = BaseType.pb2cls(nts_proto.dtype_val)[0]
    shape = Shape.__new__(Shape)
    shape.__setstate__(nts_proto.shape_val)
    result = TypeShape(dtype, shape)
    return result

  def __eq__(self, other):
    if isinstance(other, self.__class__) and \
        self.dtype == other.dtype and \
        self.shape == other.shape:
      return True
    else:
      return False

  def __hash__(self):
    result = hash(self.dtype)
    result = result * 13 + hash(self.shape)
    return result

  def __str__(self):
    return '(dtype: ' + self.dtype.__str__() + ', shape: ' + self.shape.__str__() + ')'

  def __copy__(self):
    return TypeShape(self.dtype, self.shape.__copy__())

  def __cmp__(self, other):
    if self.dtype != other.dtype:
      return 0
    return self.shape.__cmp__(other.shape)
