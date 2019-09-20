class SelectionStrategyInterface():
  arg_LIMIT = 'limit'

  def __init__(self, **kwargs):
    self._limit = kwargs.get(self.arg_LIMIT, 3)
    pass

  def select(self, pool):
    raise NotImplementedError()
    pass

  pass
