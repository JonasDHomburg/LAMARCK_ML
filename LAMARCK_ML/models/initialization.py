from LAMARCK_ML.models.interface import ModellUtil
from LAMARCK_ML.individuals import ClassifierIndividual


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
      model.generation = [ClassifierIndividual(**{
        ClassifierIndividual.arg_DATA_NTS: self.data_shapes,
        ClassifierIndividual.arg_MAX_NN_DEPTH: self.max_depth,
        ClassifierIndividual.arg_MIN_NN_DEPTH: self.min_depth,
        ClassifierIndividual.arg_MAX_NN_BRANCH: self.max_branch,
        ClassifierIndividual.arg_NN_FUNCTIONS: self.functions,
      }) for _ in range(self.gen_size)]
      func()

    return wrapper
