class NeuralNetworkFrameworkInterface():
  arg_DATA_SETS = 'data_sets'
  arg_CMP = 'cmp'

  def __init__(self, **kwargs):
    self.data_sets = kwargs.get(self.arg_DATA_SETS)
    self.cmp = kwargs.get(self.arg_CMP)
    if self.data_sets is None:
      raise Exception('Data set is None!')
    self._prepare_data_set()

  def _prepare_data_set(self):
    pass

  def setup_individual(self, individual):
    raise NotImplementedError()

  def reset_framework(self):
    raise NotImplementedError()

  pass
