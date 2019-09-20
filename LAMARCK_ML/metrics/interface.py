class MetricInterface(object):
  ID = 'NONE'

  def __init__(self, **kwargs):
    pass

  def evaluate(self, individual, framework):
    raise NotImplementedError("Function evaluateIndividual has to be inplemented!")

  @staticmethod
  def getMetricByName(name='NONE'):
    stack = [MetricInterface]
    while stack:
      cls = stack.pop(0)
      if cls.__name__ == name:
        return cls
      stack.extend(cls.__subclasses__())
    raise Exception("Couldn't find class with name: " + name)
