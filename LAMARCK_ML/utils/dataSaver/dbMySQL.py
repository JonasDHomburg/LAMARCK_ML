from LAMARCK_ML.utils.dataSaver.interface import DataSaverInterface


class DSMySQL(DataSaverInterface):
  arg_DATABASE = 'database'

  def __init__(self, **kwargs):
    super(DSMySQL, self).__init__(**kwargs)
    self._database = kwargs.get(self.arg_DATABASE, 'log')

    self.setup_db()

  def setup_db(self):
    # cursor.execute("CREATE DATABASE IF NOT EXISTS %s DEFAULT CHARACTER SET 'utf8';" % (self._database,))
    # cursor.execute("USE %s;" % (self._database,))
    pass
