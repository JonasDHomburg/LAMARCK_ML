from typing import Set, Dict, Tuple

from LAMARCK_ML.data_util import TypeShape, IOLabel
from LAMARCK_ML.data_util import ProtoSerializable


class DataFlow(ProtoSerializable):
  def __init__(self, *args, **kwargs):
    super(DataFlow, self).__init__(**kwargs)
    self._DF_INPUTS = []

  @property
  def outputs(self) -> Dict[str, TypeShape]:
    """
    Set of named data output shape and type of DataFlow object.
    :return: Set of TypeShape
    """
    raise NotImplementedError()

  @property
  def inputLabels(self) -> Set[str]:
    """
    Labels for one or more inputs.
    :return: Set of labels for inputs
    """
    return set(self._DF_INPUTS)

  @property
  def inputs(self) -> Dict[str, Tuple[str, str]]:
    """
    DataFlow connections: Dict[obj_inputLabel, Tuple[other_outputLabel, DataFlow object/id_name]]
    :return: Dict[IOLabel, Tuple[IOLabel, DataFlow/str]]
    """
    raise NotImplementedError()

  # def connect(self, dataFlow_obj: 'DataFlow', connection: Dict[IOLabel, IOLabel]):
  #   """
  #   Add a DataFlow object as input to another DataFlow object.
  #   :param dataFlow_obj: data providing DataFlow object
  #   :param connection: inputLabel -> outputLabel
  #   """
  #   raise NotImplementedError()

  @property
  def id_name(self) -> str:
    """
    :return: unique object name, typically composition of class name and class wide unique identifier
    """
    raise NotImplementedError()

  pass
