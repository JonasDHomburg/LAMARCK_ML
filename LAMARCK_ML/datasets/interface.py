from LAMARCK_ML.architectures import DataFlow
from LAMARCK_ML.data_util.attribute import pb2attr
from LAMARCK_ML.datasets.Dataset_pb2 import DatasetProto


class ResetState(Exception):
  pass


class InvalidBatchSize(Exception):
  pass


class DatasetInterface(DataFlow):
  arg_NAME = 'name'
  arg_CLSNAME = 'cls_name'

  def __init__(self, **kwargs):
    super(DatasetInterface, self).__init__(**kwargs)
    self._id_name = kwargs.get(self.arg_NAME, 'None')

  def get_pb(self, result=None):
    if not isinstance(result, DatasetProto):
      result = DatasetProto()
    result.name_val = self._id_name
    result.cls_name = self.__class__.__name__
    return result

  def restore_attributes(self, attr: dict):
    raise NotImplementedError()

  def __getstate__(self):
    self.get_pb().SerializeToString()

  @staticmethod
  def getClassByName(cls_name: str):
    stack = [DatasetInterface]
    while stack:
      cls = stack.pop(0)
      if cls.__name__ == cls_name:
        return cls
      stack.extend(cls.__subclasses__())
    raise Exception("Couldn't find class with name: " + cls_name)

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _dataset = DatasetProto()
      _dataset.ParseFromString(state)
    elif isinstance(state, DatasetProto):
      _dataset = state
    else:
      return
    cls_name = _dataset.cls_name

    try:
      self.__class__ = DatasetInterface.getClassByName(cls_name)
    except:
      pass
    self._id_name = _dataset.name_val
    attr_d = dict([pb2attr(attr) for attr in _dataset.attr_val])
    self.restore_attributes(attr_d)

  def __next__(self):
    """
    :return: Dictionary (label, data)
    """
    raise NotImplementedError()

  def __iter__(self):
    return self

  def __eq__(self, other):
    if isinstance(other, self.__class__) and \
        self._id_name == other._id_name:
      return True
    return False

  def __hash__(self):
    return hash(int.from_bytes(self._id_name.encode('utf-8'), byteorder='big'))

  @property
  def id_name(self) -> str:
    return self._id_name

  @property
  def inputs(self):
    return {}

  # @property
  # def inputLabels(self):
  #   return {}

  pass
