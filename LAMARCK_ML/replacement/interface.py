class ReplacementSchemeInterface():
  """
  Base Class For Replacement Schemes
  """

  def __init__(self, **kwargs):
    pass

  def new_generation(self, prev_gen, descendants):
    raise NotImplementedError()

  pass
