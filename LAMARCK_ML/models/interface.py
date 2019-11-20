from LAMARCK_ML.data_util import ProtoSerializable


class NEADone(Exception):
  pass


class NoSelectionMethod(Exception):
  pass


class NoMetric(Exception):
  pass


class NoReproductionMethod(Exception):
  pass


class NoReplaceMethod(Exception):
  pass


class ModelInterface(ProtoSerializable):
  def reset(self):
    raise NotImplementedError()
    pass

  def run(self):
    raise NotImplementedError()

  def stop(self):
    raise NotImplementedError()

  @property
  def abstract_timestamp(self):
    raise NotImplementedError()

  def state_stream(self):
    raise NotImplementedError()

  def from_state_stream(self, stream):
    raise NotImplementedError()

  pass


class ModellUtil(object):
  def __init__(self, **kwargs):
    super(ModellUtil, self).__init__()
