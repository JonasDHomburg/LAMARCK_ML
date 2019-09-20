from typing import Tuple, List, Dict, Set

from LAMARCK_ML.architectures import DataFlow
from LAMARCK_ML.architectures.IOMapping_pb2 import IOMappingProto
from LAMARCK_ML.architectures.functions.Function_pb2 import FunctionProto
from LAMARCK_ML.architectures.variables import Variable
from LAMARCK_ML.data_util import TypeShape, IOLabel, ProtoSerializable
from LAMARCK_ML.data_util.attribute import pb2attr, attr2pb
from LAMARCK_ML.reproduction.methods import Mutation, Recombination


class InvalidFunctionClass(Exception):
  pass


class InvalidFunctionType(Exception):
  pass


class Function(DataFlow):
  usedNames = dict()

  allowedTypes = []

  arg_ATTRIBUTES = 'attributes'
  arg_VARIABLES = 'variables'
  arg_INPUT_MAPPING = 'input_mapping'

  @classmethod
  def possible_output_shapes(cls,
                             input_ntss: Dict[str, TypeShape],
                             target_output: TypeShape,
                             is_reachable,
                             max_possibilities: int = 10) -> \
      List[Tuple[Dict[str, TypeShape], Dict[str, TypeShape], Dict[str, str]]]:
    raise NotImplementedError()

  @classmethod
  def generateParameters(cls,
                         input_dict: Dict[str, Tuple[str, Dict[str, TypeShape], str]],
                         expected_outputs: Dict[str, TypeShape],
                         variable_pool: dict = None) -> \
      Tuple[List[Dict[str, object]], List[float]]:
    raise NotImplementedError()

  @staticmethod
  def resetNames():
    Function.usedNames.clear()

  @staticmethod
  def getClassByName(class_name: str):
    names = class_name.split('-')
    if Mutation.Interface.__name__ in names:
      names.remove(Mutation.Interface.__name__)
    if Recombination.Interface.__name__ in names:
      names.remove(Recombination.Interface.__name__)
    if ProtoSerializable.__name__ in names:
      names.remove(ProtoSerializable.__name__)
    if not (len(names) > 2 and names[-2] == Function.__name__ and names[-1] == DataFlow.__name__):
      raise InvalidFunctionClass()
    current_cls = Function
    classes = names[::-1][2:]
    while len(classes) > 0:
      sub_classes = dict([(f.__name__, f) for f in current_cls.__subclasses__()])
      cls = classes[0]
      new_class = sub_classes.get(cls)
      if new_class is None:
        break
      current_cls = new_class
      classes.pop(0)

    if len(classes) == 0:
      return current_cls
    raise InvalidFunctionClass('No class found for: ' + class_name)

  @classmethod
  def getNewName(cls, obj=None) -> str:
    # assert obj is not None and isinstance(obj, Function)
    cls_name = cls.__name__
    idx = Function.usedNames.get(cls_name, 0)
    name = cls_name + ":%09i" % (idx)
    idx += 1
    Function.usedNames[cls_name] = idx
    if obj is not None and isinstance(obj, Function):
      Function.usedNames[name] = obj
    return name

  def __init__(self, *args, **kwargs):
    super(Function, self).__init__(*args, **kwargs)
    self._name = self.__class__.getNewName(self)
    self.variables = kwargs.get(self.arg_VARIABLES, list())
    assert isinstance(self.variables, list) and not any([not isinstance(v, Variable) for v in self.variables])
    self.input_mapping = kwargs.get(self.arg_INPUT_MAPPING)
    if not (isinstance(self.input_mapping, dict) and
            all([isinstance(key, str) and isinstance(value, tuple) and
                 isinstance(value[0], str) and (isinstance(value[1], str) or isinstance(value[1], DataFlow))
                 for key, value in self.input_mapping.items()])):
      raise Exception('Wrong input mapping!!')
    for key, (label, obj) in self.input_mapping.items():
      if isinstance(obj, DataFlow):
        self.input_mapping[key] = (label, obj.id_name)
    self.attr = kwargs.get(self.arg_ATTRIBUTES, dict())
    assert isinstance(self.attr, dict)

  def __getstate__(self):
    return self.get_pb().SerializeToString()

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _function = FunctionProto()
      _function.ParseFromString(state)
    elif isinstance(state, FunctionProto):
      _function = state
    else:
      return
    self.__class__ = Function.getClassByName(_function.class_name)
    self._name = _function.id_name
    self.input_mapping = dict()
    for inProto in _function.input_mapping:
      self.input_mapping[inProto.in_label] = (inProto.out_label, inProto.df_id_name)
    self.variables = []
    for _variable in _function.variables:
      _v = Variable.__new__(Variable)
      _v.__setstate__(_variable)
      self.variables.append(_v)
    self._cls_setstate(_function)
    Function.usedNames[self._name] = self

  def __copy__(self):
    result = self.__class__.__new__(self.__class__)
    result._name = self._name
    result.input_mapping = dict(self.input_mapping)
    result.variables = list(self.variables)
    result.attr = dict([pb2attr(attr2pb(key, value)) for key, value in self.attr.items()])
    return result

  def _cls_setstate(self, _function):
    """
    For class dependent attribute retrieval override this method!
    :param state: FunctionProto
    :return: None
    """
    self.attr = dict([pb2attr(attr) for attr in _function.attr])

  def __eq__(self, other):
    if (isinstance(other, self.__class__) and
        self._name == other._name and
        self.variables == other.variables and
        len({k: self.attr.get(k) for k in self.attr if self.attr.get(k) == other.attr.get(k)})
        == len(self.attr) == len(other.attr) and
        len({i: self.input_mapping.get(i) for i in self.input_mapping if
             self.input_mapping.get(i) == other.input_mapping.get(i)})
        == len(self.input_mapping) == len(other.input_mapping)
    ):
      return True
    return False

  def __ne__(self, other):
    return not self.__eq__(other)

  @classmethod
  def get_cls_name(cls):
    return '-'.join([c.__name__ for c in cls.mro()[:-1]])

  def get_pb(self, result=None) -> FunctionProto:
    if not isinstance(result, FunctionProto):
      result = FunctionProto()
    result.class_name = self.get_cls_name()
    result.id_name = self._name
    for input_label, (output_label, df_obj) in self.input_mapping.items():
      input_proto = IOMappingProto()
      input_proto.in_label = input_label
      input_proto.out_label = output_label
      input_proto.df_id_name = df_obj
      result.input_mapping.append(input_proto)
    result.variables.extend([v.get_pb() for v in self.variables])
    result.attr.extend([attr2pb(key, value) for key, value in self.attr.items()])
    return result

  @staticmethod
  def get_instance(state):
    if isinstance(state, str) or isinstance(state, bytes):
      _function = FunctionProto()
      _function.ParseFromString(state)
    elif isinstance(state, FunctionProto):
      _function = state
    else:
      return
    result = Function.usedNames.get(_function.id_name)
    if result is None:
      result = Function.__new__(Function)
      result.__setstate__(state)
    return result

  @property
  def id_name(self) -> str:
    return self._name

  @property
  def inputs(self) -> Dict[str, Tuple[str, str]]:
    return self.input_mapping

  @classmethod
  def min_transform(cls, nts):
    raise NotImplementedError()

  @classmethod
  def max_transform(cls, nts):
    raise NotImplementedError()
