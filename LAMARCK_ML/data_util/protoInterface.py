class ProtoSerializable(object):
  def __init__(self, **kwargs):
    super(ProtoSerializable, self).__init__()

  def get_pb(self, result=None):
    raise NotImplementedError()
