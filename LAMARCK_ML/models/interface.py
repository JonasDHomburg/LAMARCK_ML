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

  def setstate_from_pb(self, _model):
    raise NotImplementedError()

  pass


class ModellUtil(object):
  pass
