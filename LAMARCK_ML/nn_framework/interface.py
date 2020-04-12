from LAMARCK_ML.architectures.functions.interface import Function
from typing import Dict, List, Tuple, Set
from deprecated import deprecated

class NeuralNetworkFrameworkInterface():
  arg_DATA_SETS = 'data_sets'
  arg_CMP = 'cmp'

  arg_OPTIMIZER = 'optimizer'
  arg_LOSS = 'loss'
  arg_METRICS = 'metrics'

  def __init__(self, **kwargs):
    self.data_sets = kwargs.get(self.arg_DATA_SETS)
    self.cmp = kwargs.get(self.arg_CMP)
    if self.data_sets is None:
      raise Exception('Data set is None!')
    self._prepare_data_set()

  def _prepare_data_set(self):
    pass

  @deprecated(version='0.2', reason='Each individual class would need its own implementation => interface rework')
  def setup_individual(self, individual):
    raise NotImplementedError()

  @deprecated(version='0.2', reason='Interface rework for better modularity.')
  def reset_framework(self):
    raise NotImplementedError()

  # =================================
  # New Interface!

  def init_model(self, dataset_input_data: Set[str], dataset_target_data: Set[str]):
    raise NotImplementedError()

  def finalize_model(self, output_ids: List[Tuple[str, str]]):
    raise NotImplementedError()

  def set_weights(self, weights: Dict):
    raise NotImplementedError()

  def set_train_parameters(self, **kwargs):
    raise NotImplementedError

  def add_function(self, function: Function):
    raise NotImplementedError()

  def train(self) -> Dict:
    raise NotImplementedError()

  def reset(self):
    raise NotImplementedError()

  pass
