from LAMARCK_ML.models.interface import ModellUtil
from LAMARCK_ML.individuals import ClassifierIndividualOPACDG, GraphLayoutIndividual


class InitializationStrategyInterface(ModellUtil):
  def __init__(self, **kwargs):
    pass

  def seed_generation(self, func):
    def wrapper(model):
      raise NotImplementedError()

    return wrapper

  pass


class SimpleRandomClassifierInitializer(InitializationStrategyInterface):
  arg_MIN_DEPTH = 'min_depth'
  arg_MAX_DEPTH = 'max_depth'
  arg_MAX_BRANCH = 'max_branch'
  arg_FUNCTIONS = 'functions'
  arg_DATA_SHAPES = 'data_shapes'
  arg_GEN_SIZE = 'gen_size'

  def __init__(self, **kwargs):
    super(SimpleRandomClassifierInitializer, self).__init__(**kwargs)
    self.min_depth = kwargs.get(self.arg_MIN_DEPTH, 1)
    self.max_depth = kwargs.get(self.arg_MAX_DEPTH, 10)
    self.max_branch = kwargs.get(self.arg_MAX_BRANCH, 2)
    self.functions = kwargs.get(self.arg_FUNCTIONS)
    self.data_shapes = kwargs.get(self.arg_DATA_SHAPES)
    self.gen_size = kwargs.get(self.arg_GEN_SIZE)

  def seed_generation(self, func):
    def wrapper(model):
      model.generation = [ClassifierIndividualOPACDG(**{
        ClassifierIndividualOPACDG.arg_DATA_NTS: self.data_shapes,
        ClassifierIndividualOPACDG.arg_MAX_NN_DEPTH: self.max_depth,
        ClassifierIndividualOPACDG.arg_MIN_NN_DEPTH: self.min_depth,
        ClassifierIndividualOPACDG.arg_MAX_NN_BRANCH: self.max_branch,
        ClassifierIndividualOPACDG.arg_NN_FUNCTIONS: self.functions,
      }) for _ in range(self.gen_size)]
      func()

    return wrapper


class RandomGraphLayoutInitializer(InitializationStrategyInterface):
  arg_GEN_SIZE = 'gen_size'
  arg_EDGES = 'edges'
  arg_DISTANCE = 'distance'
  arg_METRIC_WEIGHTS = 'metric_weights'

  def __init__(self, **kwargs):
    super(RandomGraphLayoutInitializer, self).__init__(**kwargs)
    self.gen_size = kwargs.get(self.arg_GEN_SIZE)
    self.edges = kwargs.get(self.arg_EDGES)
    self.distance = kwargs.get(self.arg_DISTANCE)
    self.metric_weights = kwargs.get(self.arg_METRIC_WEIGHTS, dict())

  def seed_generation(self, func):
    def wrapper(model):
      model.generation = [GraphLayoutIndividual(**{
        GraphLayoutIndividual.arg_EDGES: self.edges,
        GraphLayoutIndividual.arg_DISTANCE: self.distance,
        GraphLayoutIndividual.arg_METRIC_WEIGHTS: self.metric_weights,
      }) for _ in range(self.gen_size)]
      func()

    return wrapper


class RandomInitializer(InitializationStrategyInterface):
  arg_CLASS = 'individual_class'
  arg_PARAM = 'individual_parameter'
  arg_GEN_SIZE = 'gen_size'

  def __init__(self, **kwargs):
    super(RandomInitializer, self).__init__(**kwargs)
    self._class = kwargs.get(self.arg_CLASS, ClassifierIndividualOPACDG)
    self._param = kwargs.get(self.arg_PARAM, {})
    self.gen_size = kwargs.get(self.arg_GEN_SIZE)

  def seed_generation(self, func):
    def wrapper(model):
      model.generation = [self._class(**self._param) for _ in range(self.gen_size)]
      func()

    return wrapper
