from enum import Enum

import numpy as np

from LAMARCK_ML.data_util import Shape, BaseType, TypeShape
from LAMARCK_ML.data_util.Attribute_pb2 import AttributeProto


def attr2pb(name, value):
  def value2pb(value, v=None):
    if v is None:
      v = AttributeProto.Value()
    if isinstance(value, int):
      v.int_val = value
    elif isinstance(value, float):
      v.double_val = value
    elif isinstance(value, bool):
      v.bool_val = value
    elif isinstance(value, str):
      v.string_val = value
    elif isinstance(value, bytes):
      v.bytes_val = value
    elif isinstance(value, Shape):
      value.get_pb(v.shape_val)
    elif isinstance(value, TypeShape):
      value.get_pb(v.nts_val)
    elif isinstance(value, set):
      v.set_val.v.extend([value2pb(_v) for _v in value])
    elif isinstance(value, list):
      v.list_val.v.extend([value2pb(_v) for _v in value])
    elif isinstance(value, tuple):
      v.tuple_val.v.extend([value2pb(_v) for _v in value])
    elif isinstance(value, dict):
      v.dict_val.vs.extend([attr2pb(name=_k, value=_v) for _k, _v in value.items()])
    elif isinstance(value, Enum):
      value2pb(value.value, v=v)
    elif isinstance(value, np.ndarray):
      value2pb(value.tolist(), v=v)
      v.list_val.numpy = True
    # elif inspect.isclass(value) and issubclass(value, BaseType):
    elif isinstance(value, type) and issubclass(value, BaseType):
      value.get_pb(v.type_val)
    else:
      v.bytes_val = bytes(value)
    return v

  attr = AttributeProto()
  attr.name = name
  value2pb(value, attr.v)
  return attr


def pb2attr(attr):
  def pb2val(pb):
    whichone = pb.WhichOneof("v")
    if whichone == 'shape_val':
      shape_ = Shape.__new__(Shape)
      shape_.__setstate__(getattr(pb, whichone))
      return shape_
    elif whichone == 'type_val':
      return BaseType.pb2cls(getattr(pb, whichone))[0]
    elif whichone == 'list_val':
      _list = [pb2val(_pb) for _pb in pb.list_val.v]
      return np.asarray(_list) if getattr(pb.list_val, 'numpy', False) else _list
    elif whichone == 'set_val':
      return set([pb2val(_pb) for _pb in pb.set_val.v])
    elif whichone == 'tuple_val':
      return tuple([pb2val(_pb) for _pb in pb.tuple_val.v])
    elif whichone == 'nts_val':
      return TypeShape.from_pb(pb.nts_val)
    elif whichone == 'dict_val':
      return dict([(elem.name, pb2val(elem.v)) for elem in pb.dict_val.vs])
    else:
      return getattr(pb, str(whichone))

  return attr.name, pb2val(attr.v)
