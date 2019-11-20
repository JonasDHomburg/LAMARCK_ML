import numpy as np

from LAMARCK_ML.architectures.variables.Variable_pb2 import VariableProto
from LAMARCK_ML.architectures.variables.initializer import Initializer
from LAMARCK_ML.architectures.variables.regularisation import Regularisation
from LAMARCK_ML.data_util import BaseType, ProtoSerializable
from LAMARCK_ML.data_util.Shape_pb2 import ShapeProto


class Variable(ProtoSerializable):
  arg_DTYPE = 'dtype'
  arg_TRAINABLE = 'trainable'
  arg_NAME = 'name'
  arg_SHAPE = 'shape'
  arg_INITIALIZER = 'initializer'
  arg_REGULARISATION = 'regularisation'
  arg_VALUE = 'value'

  def __init__(self, **kwargs):
    self.dtype = kwargs.get(self.arg_DTYPE)
    assert isinstance(self.dtype, BaseType) or (isinstance(self.dtype, type) and issubclass(self.dtype, BaseType))
    self.trainable = kwargs.get(self.arg_TRAINABLE, True)
    assert isinstance(self.trainable, bool)
    self.name = kwargs.get(self.arg_NAME, '')
    assert isinstance(self.name, str)
    self.value = kwargs.get(self.arg_VALUE)
    self.initializer = kwargs.get(self.arg_INITIALIZER)
    self._shape = tuple(kwargs.get(self.arg_SHAPE)) if not isinstance(self.value, np.ndarray) else tuple(self.value.shape)
    assert isinstance(self.value, np.ndarray) or (self.initializer is not None and self._shape is not None)
    self.regularisation = kwargs.get(self.arg_REGULARISATION)

  def __eq__(self, other):
    if not isinstance(other, Variable):
      return False
    if not (
        self.dtype == other.dtype and
        self.trainable == other.trainable and
        self.name == other.name and
        self._shape == other._shape and
        self.initializer == other.initializer and
        self.regularisation == other.regularisation and
        ((self.value is None and other.value is None) or
         np.array_equal(self.value, other.value))
    ):
      return False
    return True

  @property
  def shape(self):
    return self._shape

  def get_pb(self, result=None):
    if not isinstance(result, VariableProto):
      result = VariableProto()
    self.dtype.get_pb(result.dtype)
    result.trainable = self.trainable
    result.name = self.name
    for dim in self.shape:
      _dim = ShapeProto.Dim()
      _dim.size = dim
      result.shape.dim.append(_dim)
    if self.value is not None:
      result.__getattribute__(self.dtype.attr).extend(self.value.reshape(-1).tolist())
    if self.initializer is not None:
      self.initializer.get_pb(result.initializer)
    if self.regularisation is not None:
      self.regularisation.get_pb(result.regularisation)
    return result

  def __getstate__(self):
    return self.get_pb().SerializeToString()

  def __copy__(self):
    result = Variable.__new__(Variable)
    result.dtype = self.dtype
    result.trainable = self.trainable
    result.name = self.name
    result.value = np.copy(self.value) if self.value is not None else None
    result.initializer = self.initializer
    result._shape = tuple(self._shape)
    result.regularisation = self.regularisation
    return result

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _variable = VariableProto()
      _variable.ParseFromString(state)
    elif isinstance(state, VariableProto):
      _variable = state
    else:
      return
    self.dtype, _ = BaseType.pb2cls(_variable.dtype)
    _shape = []
    for s in _variable.shape.dim:
      _shape.append(s.size)
    _shape = tuple(_shape)
    self._shape = _shape
    valueAsList = _variable.__getattribute__(self.dtype.attr)
    if len(valueAsList) > 0:
      values = np.array(valueAsList)
      self.value = values.reshape(_shape)
      self._shape = tuple(self.value.shape)
    else:
      self.value = None
    self.trainable = _variable.trainable
    self.name = _variable.name
    if _variable.HasField('initializer'):
      init_ = Initializer.__new__(Initializer)
      init_.__setstate__(_variable.initializer)
      self.initializer = init_
    else:
      self.initializer = None
    if _variable.HasField('regularisation'):
      regu_ = Regularisation.__new__(Regularisation)
      regu_.__setstate__(_variable.regularisation)
      self.regularisation = regu_
    else:
      self.regularisation = None

  pass
