from LAMARCK_ML.metrics.interface import MetricInterface


class Accuracy(MetricInterface):
  """
  Accuracy on test data set (this should not be the data set to evaluate the overall algorithm).
  """
  ID = 'ACC'

  class Interface:
    def accuracy(self, obj):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.accuracy(framework)
    if isinstance(framework, self.Interface):
      return framework.accuracy(individual)
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
    if isinstance(individual, self.Interface):
      return individual.flops_per_sample()
    if isinstance(framework, self.Interface):
      return framework.flops_per_sample()
    return -.1


class Parameters(MetricInterface):
  ID = 'WEIGHTS'

  class Interface:
    def parameters(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.parameters()
    if isinstance(framework, self.Interface):
      return framework.parameters()
    return -.1


class MemoryMetric(MetricInterface):
  ID = 'MEMORY'

  class Interface:
    def memory(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.memory()
    if isinstance(framework, self.Interface):
      return framework.memory()
    return -.1


class TimeMetric(MetricInterface):
  ID = 'TIME'

  class Interface:
    def time(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.time()
    if isinstance(framework, self.Interface):
      return framework.time()
    return -.1


class Nodes(MetricInterface):
  ID = 'NODES'

  class Interface:
    def nodes(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.nodes()
    if isinstance(framework, self.Interface):
      return framework.nodes()
    return -.1


class LayoutCrossingEdges(MetricInterface):
  ID = 'LCE'

  class Interface:
    def layoutCrossingEdges(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.layoutCrossingEdges()
    return -.1


class LayoutDistanceX(MetricInterface):
  ID = 'LDX'

  class Interface:
    def layoutDistanceX(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.layoutDistanceX()
    return -.1


class LayoutDistanceY(MetricInterface):
  ID = 'LDY'

  class Interface:
    def layoutDistanceY(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.layoutDistanceY()
    return -.1


class CartesianFitness(MetricInterface):
  ID = 'CF'

  class Interface:
    def cartesianFitness(self):
      raise NotImplementedError()

  def evaluate(self, individual, framework):
    if isinstance(individual, self.Interface):
      return individual.cartesianFitness()
    if isinstance(framework, self.Interface):
      return framework.cartesianFitness()
    return -.1
