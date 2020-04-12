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
    """
    A norm d(self, other) for the architecture.
    :param other: The other architecture.
    :return: Distance between self and other.
    """
    raise NotImplementedError()

  def update_state(self, *args, **kwargs):
    """
    Update the state of the architecture like trainable weights.
    :param args: Currently unused.
    :param kwargs: Dictionary used to pass i.e. trained weights or estimated metrics.
    :return: None
    """
    raise NotImplementedError()

  pass
