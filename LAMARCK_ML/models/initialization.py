from LAMARCK_ML.models.interface import ModellUtil


class InitializationStrategyInterface(ModellUtil):
  def __init__(self, **kwargs):
    pass

  def seed_generation(self, func):
    def wrapper(model):
      raise NotImplementedError()
      # func()

    return wrapper

  pass
