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
      self._framework.setup_individual(individual)
      individual.metrics = dict([(m.ID, m.evaluate(self=m, individual=individual, framework=self._framework))
                                 for m in metrics])
      self._framework.teardown_individual()
