from LAMARCK_ML.models.interface import ModellUtil


class DataSaverInterface(ModellUtil):
  def __init__(self, **kwargs):
    pass

  def get_individual_by_name(self, name):
    raise NotImplementedError()

  def get_ancestry_for_ind(self, ind_name):
    raise NotImplementedError()