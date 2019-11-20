from LAMARCK_ML.data_util import ProtoSerializable
from LAMARCK_ML.metrics import Accuracy

from LAMARCK_ML.utils.CompareClass_pb2 import CompareClassProto
from LAMARCK_ML.data_util.attribute import attr2pb, pb2attr


class CompareClass(ProtoSerializable):
  arg_PRIMARY_OBJECTIVE = 'primary_objective'
  arg_PRIMARY_ALPHA = 'primary_alpha'
  arg_PRIMARY_THRESHOLD = 'primary_threshold'
  arg_SECONDARY_OBJECTIVES = 'secondary_objectives'

  def __init__(self, **kwargs):
    super(CompareClass, self).__init__(**kwargs)
    self.primary_objective = kwargs.get(self.arg_PRIMARY_OBJECTIVE, Accuracy.ID)
    self.primary_alpha = kwargs.get(self.arg_PRIMARY_ALPHA, 1)
    self.primary_threshold = kwargs.get(self.arg_PRIMARY_THRESHOLD, .6)
    self.secondary_objectives = kwargs.get(self.arg_SECONDARY_OBJECTIVES, {})

  def greaterThan(self, one, other):
    if one is not None and other is None:
      return True
    elif one is None:
      return False
    PO_0 = one.get(self.primary_objective) * self.primary_alpha
    PO_1 = other.get(self.primary_objective) * self.primary_alpha
    if PO_0 < self.primary_threshold and PO_1 < self.primary_threshold:
      return PO_0 > PO_1
    f = (PO_0 - PO_1) * self.primary_alpha
    if any([one[so] != other[so] and 0 < f * self.secondary_objectives[so] / (one[so] - other[so]) < 1
            for so in self.secondary_objectives if so in one and so in other]):
      return False if PO_0 > PO_1 else True
    else:
      return True if PO_0 > PO_1 else False

  def get_pb(self, result=None):
    if not isinstance(result, CompareClassProto):
      result = CompareClassProto()

    result.attr.append(attr2pb(self.arg_PRIMARY_OBJECTIVE, self.primary_objective))
    result.attr.append(attr2pb(self.arg_PRIMARY_ALPHA, self.primary_alpha))
    result.attr.append(attr2pb(self.arg_PRIMARY_THRESHOLD, self.primary_threshold))
    result.attr.append(attr2pb(self.secondary_objectives, self.secondary_objectives))
    return result

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _cmp = CompareClassProto()
      _cmp.ParseFromString(state)
    elif isinstance(state, CompareClassProto):
      _cmp = state
    else:
      return
    attr = dict([pb2attr(pb) for pb in state.attr])
    self.primary_objective = attr.get(self.arg_PRIMARY_OBJECTIVE, Accuracy.ID)
    self.primary_alpha = attr.get(self.arg_PRIMARY_ALPHA, 1)
    self.primary_threshold = attr.get(self.arg_PRIMARY_THRESHOLD, .6)
    self.secondary_objectives = attr.get(self.arg_SECONDARY_OBJECTIVES, {})