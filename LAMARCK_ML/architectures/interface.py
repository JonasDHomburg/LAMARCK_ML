from typing import Set

from LAMARCK_ML.data_util import TypeShape


class ArchitectureInterface():
  def __init__(self, **kwargs):
    super(ArchitectureInterface, self).__init__()

  @property
  def outputs(self) -> Set[TypeShape]:
    """
    Set of named data output type and shape.
    :return: Set of TypeShape
    """
    raise NotImplementedError()

  def norm(self, other):
    raise NotImplementedError()

  pass
