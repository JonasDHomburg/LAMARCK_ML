class NeuralNetworkFrameworkInterface():
  arg_DATA_SETS = 'data_sets'

  def __init__(self, **kwargs):
    self.data_sets = kwargs.get(self.arg_DATA_SETS)
    if self.data_sets is None:
      raise Exception('Data set is None!')
    self._prepare_data_set()

  def _prepare_data_set(self):
    pass

  def setup_individual(self, individual):
    raise NotImplementedError()

  def teardown_individual(self):
    raise NotImplementedError()

  pass
