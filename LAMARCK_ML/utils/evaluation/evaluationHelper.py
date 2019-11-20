from LAMARCK_ML.nn_framework import NeuralNetworkFrameworkInterface
from LAMARCK_ML.utils.evaluation.interface import EvaluationHelperInterface


class LocalEH(EvaluationHelperInterface):
  arg_NN_FRAMEWORK = 'framework'

  def __init__(self, **kwargs):
    super(LocalEH, self).__init__()
    self._framework = kwargs.get(self.arg_NN_FRAMEWORK)
    #   TODO: setup framework
    if not isinstance(self._framework, NeuralNetworkFrameworkInterface):
      raise Exception()

  def evaluate(self, generation, metrics):
    for individual in generation:
      state = self._framework.setup_individual(individual)
      individual.metrics = dict([(m.ID, m.evaluate(individual=individual, framework=self._framework))
                                 for m in metrics])
      individual.update_state(**state)
      self._framework.reset_framework()


class GraphLayoutEH(EvaluationHelperInterface):

  def __init__(self, **kwargs):
    super(GraphLayoutEH, self).__init__(**kwargs)

  def evaluate(self, generation, metrics):
    for individual in generation:
      individual.metrics = dict([(m.ID, m.evaluate(individual=individual, framework=None))
                                 for m in metrics])
