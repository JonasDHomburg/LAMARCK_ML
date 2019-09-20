from LAMARCK_ML.metrics.interface import MetricInterface


class Accuracy(MetricInterface):
  """
  Accuracy on test data set (this should not be the data set to evaluate the overall algorithm).
  """
  ID = 'ACC'

  class Interface:
    def accuracy(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(framework, self.Interface):
      return framework.accuracy()
    return -.1


class FlOps(MetricInterface):
  """
  Floating Operations for one sample.
  """
  ID = 'FLOPS'

  class Interface:
    def flops_per_sample(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(framework, self.Interface):
      return framework.flops_per_sample()
    return -.1


class Parameters(MetricInterface):
  ID = 'WEIGHTS'

  class Interface:
    def parameters(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(framework, self.Interface):
      return framework.parameters()
    if isinstance(individual, self.Interface):
      return individual.parameters()
    return -.1


class MemoryMetric(MetricInterface):
  ID = 'MEMORY'

  class Interface:
    def memory(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(framework, self.Interface):
      return framework.memory()
    return -.1


class TimeMetric(MetricInterface):
  ID = 'TIME'

  class Interface:
    def time(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(framework, self.Interface):
      return framework.time()
    return -.1


class NodesInNetwork(MetricInterface):
  ID = 'NODES'

  class Interface:
    def nodes_in_network(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.nodes_in_network()
    return -.1
