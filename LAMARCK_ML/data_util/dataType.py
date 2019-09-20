from LAMARCK_ML.data_util.DType_pb2 import \
  DInvalid as DInvalidProto, \
  DBinary as DBinaryProto, \
  DUInt as DUIntProto, \
  DInt as DIntProto, \
  DFloat as DFloatProto, \
  DComplex as DComplexProto, \
  DBool as DBoolProto, \
  DString as DStringProto, \
  DTypeProto


class InvalidDatatype(Exception):
  pass


class BaseType(type):
  attr = 'bytes_val'
  pb = DInvalidProto
  bits = None

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self.__class__)

  @staticmethod
  def pb2cls(pb_type, _class=None):
    if _class is None:
      _class = BaseType
    for _subclass in _class.__subclasses__(_class):
      if _subclass.pb == pb_type.type_val and \
          _subclass.bits == getattr(pb_type, 'bits_val', None):
        return _subclass, True
      else:
        _sc, found = BaseType.pb2cls(pb_type, _subclass)
        if found:
          return _sc, found
    return None, False

  @staticmethod
  def str2cls(str_type, _class=None):
    if _class is None:
      _class = BaseType
    for _subclass in _class.__subclasses__(_class):
      if _subclass.__name__ == str_type:
        return _subclass, True
      else:
        _sc, found = BaseType.str2cls(str_type, _subclass)
        if found:
          return _sc, found
    return None, False

  @classmethod
  def get_pb(cls, result=None):
    if not isinstance(result, DTypeProto):
      result = DTypeProto()
    result.type_val = cls.pb
    if cls.bits is not None:
      result.bits_val = cls.bits
    return result

  @classmethod
  def __str__(cls):
    return cls.__name__ + ':' + str(cls.bits)

  pass


class DHalf(BaseType):
  attr = 'half_val'
  pb = DFloatProto
  bits = 16


class DFloat(BaseType):
  attr = 'float_val'
  pb = DFloatProto
  bits = 32


class DDouble(BaseType):
  attr = 'double_val'
  pb = DFloatProto
  bits = 64


class DInt8(BaseType):
  attr = 'int_val'
  pb = DIntProto
  bits = 8


class DInt16(BaseType):
  attr = 'int_val'
  pb = DIntProto
  bits = 16


class DInt32(BaseType):
  attr = 'int_val'
  pb = DIntProto
  bits = 32


class DInt64(BaseType):
  attr = 'int64_val'
  pb = DIntProto
  bits = 64


class DString(BaseType):
  attr = 'string_val'
  pb = DStringProto


class DBool(BaseType):
  attr = 'bool_val'
  pb = DBoolProto
  bits = 1


class DComplex64(BaseType):
  attr = 'scomplex_val'
  pb = DComplexProto
  bits = 64


class DComplex128(BaseType):
  attr = 'dcomplex_val'
  pb = DComplexProto
  bits = 128


class DUInt8(BaseType):
  attr = 'uint32_val'
  pb = DUIntProto
  bits = 8


class DUInt16(BaseType):
  attr = 'uint32_val'
  pb = DUIntProto
  bits = 16


class DUInt32(BaseType):
  attr = 'uint32_val'
  pb = DUIntProto
  bits = 32


class DUInt64(BaseType):
  attr = 'uint64_val'
  pb = DUIntProto
  bits = 64


class DBinary(BaseType):
  attr = 'bytes_val'
  pb = DBinaryProto
  bits = None
