import sqlite3 as db
import time
import os

from LAMARCK_ML.individuals import IndividualInterface
from LAMARCK_ML.models.models import GenerationalModel
from LAMARCK_ML.reproduction import AncestryEntity
from LAMARCK_ML.reproduction.Ancestry_pb2 import AncestryProto
from LAMARCK_ML.utils.dataSaver.dbConnection import DBConstants
from LAMARCK_ML.utils.dataSaver.interface import DataSaverInterface
from LAMARCK_ML.individuals.implementations.NetworkIndividual_pb2 import NetworkIndividualProto
from LAMARCK_ML.individuals.Individual_pb2 import IndividualProto


class DSSqlite3(DataSaverInterface):
  arg_FILE = 'file'
  arg_SAVE_ALL = 'save_all'

  def __init__(self, **kwargs):
    super(DSSqlite3, self).__init__(**kwargs)
    self.save_all = kwargs.get(self.arg_SAVE_ALL, False)

    self._file = kwargs.get(self.arg_FILE, 'default.db3')
    self._path = os.path.dirname(self._file)
    if self._path == '':
      self._path = './'
    if (not os.path.exists(self._path)) and (self._path != ''):
      os.makedirs(self._path)
    self.conn = db.connect(self._file)

    self.setup_db()

  def __del__(self):
    self.conn.close()

  def setup_db(self):
    cursor = self.conn.cursor()

    while True:
      try:
        cursor.execute(
          "CREATE TABLE IF NOT EXISTS {} (rowid INTEGER PRIMARY KEY autoincrement, real_timestamp INTEGER, "
          "abstract_timestamp INTEGER, id_name TEXT, serialized_file TEXT);".format(
            DBConstants.table_individual.value[0]))
      except db.OperationalError:
        continue
      break
    while True:
      try:
        cursor.execute(
          "CREATE TABLE IF NOT EXISTS {} (rowid INTEGER PRIMARY KEY autoincrement, real_timestamp INTEGER, "
          "abstract_timestamp INTEGER, operation VARCHAR(8), descendant TEXT, serialized BLOB)".format(
            DBConstants.table_ancestry.value[0]))
      except db.OperationalError:
        continue
      break
    self.conn.commit()
    cursor.close()

  def get_individual_by_name(self, name):
    # cursor = self.conn.cursor()
    # cursor.execute("SELECT serialized FROM {} WHERE id_name=?".format(DBConstants.table_individual.value[0]), [name])
    # fetched = cursor.fetchone()
    # last = None
    # while fetched:
    #   last = fetched[0]
    #   fetched = cursor.fetchone()
    # cursor.close()
    with open(self._path + '/' + name + '.pb', 'rb') as f:
      last = f.read()
    ind = IndividualInterface.__new__(IndividualInterface)
    ind.__setstate__(last)
    return ind

  def end_evaluate(self, func):
    def end_evaluate_wrapper(model: GenerationalModel):
      real_timestamp = int(time.time())
      abstract_timestamp = model.abstract_timestamp
      statement = "INSERT INTO {} (real_timestamp, abstract_timestamp, id_name, serialized_file) " \
                  "VALUES (?, ?, ?, ?);".format(DBConstants.table_individual.value[0])
      for individual in model.generation:
        cursor = self.conn.cursor()
        cursor.execute(statement, [real_timestamp,
                                   abstract_timestamp,
                                   individual.id_name,
                                   individual.id_name + '.pb'])
        self.conn.commit()
        cursor.close()
        with open(self._path + '/' + individual.id_name + '.pb', 'wb') as f:
          _bytes = individual.__getstate__()
          f.write(_bytes)
        del _bytes
      func()

    return end_evaluate_wrapper

  def end_reproduce(self, func):
    def end_reproduce_wrapper(model):
      real_timestamp = int(time.time())
      abstract_timestamp = model.abstract_timestamp
      statement_rep = "INSERT INTO {} (real_timestamp, abstract_timestamp, operation, descendant, serialized) " \
                      "VALUES (?, ?, ?, ?, ?);".format(
        DBConstants.table_ancestry.value[0])
      if self.save_all:
        statement_ind = "INSERT INTO {} (real_timestamp, abstract_timestamp, id_name, serialized_file) " \
                        "VALUES (?, ?, ?, ?);".format(DBConstants.table_individual.value[0])
        for pool in model._REPRODUCTION_POOLS:
          for individual in pool:
            cursor = self.conn.cursor()
            cursor.execute(statement_ind, [real_timestamp,
                                           abstract_timestamp,
                                           individual.id_name,
                                           individual.id_name + '.pb'])
            with open(self._path + '/' + individual.id_name + '.pb', 'wb') as f:
              _bytes = individual.__getstate__()
              f.write(_bytes)
              f.close()
            self.conn.commit()
            cursor.close()
            del _bytes
      for _, ancestry in model.reproduction:
        for anc in ancestry:
          cursor = self.conn.cursor()
          _state = anc.__getstate__()
          cursor.execute(statement_rep, [real_timestamp,
                                         abstract_timestamp,
                                         anc.method,
                                         anc.descendant,
                                         db.Binary(_state)])
          self.conn.commit()
          cursor.close()
          del _state
      func()

    return end_reproduce_wrapper

  def get_ancestry_for_ind(self, ind_name):
    cursor = self.conn.cursor()
    cursor.execute("SELECT abstract_timestamp, serialized FROM {} WHERE descendant=?;".format(
      DBConstants.table_ancestry.value[0]), [ind_name])
    fetched = cursor.fetchone()
    last = None
    while fetched:
      last = fetched
      fetched = cursor.fetchone()
    cursor.close()
    if last is None:
      return None, None
    pb = AncestryProto()
    pb.ParseFromString(last[1])
    return last[0], AncestryEntity.from_pb(pb)

  def get_ancestries(self):
    cursor = self.conn.cursor()
    while True:
      try:
        cursor.execute("SELECT abstract_timestamp, serialized FROM {};".format(
          DBConstants.table_ancestry.value[0]))
      except db.OperationalError:
        continue
      break
    result = []
    for abstract_time, pb_bytes in cursor.fetchall():
      pb = AncestryProto()
      pb.ParseFromString(pb_bytes)
      result.append((abstract_time, AncestryEntity.from_pb(pb)))
    cursor.close()
    return result

  def get_individual_names(self):
    cursor = self.conn.cursor()
    while True:
      try:
        cursor.execute("SELECT id_name FROM {};".format(DBConstants.table_individual.value[0]))
      except db.OperationalError:
        continue
      break
    result = set([id_[0] for id_ in cursor.fetchall()])
    cursor.close()
    return result

  def time_stamps_by_individual_name(self, individual_name):
    cursor = self.conn.cursor()
    while True:
      try:
        cursor.execute("SELECT real_timestamp, abstract_timestamp FROM {} WHERE id_name=?;".format(
          DBConstants.table_individual.value[0]), [individual_name])
      except db.OperationalError:
        continue
      break
    result = cursor.fetchall()
    cursor.close()
    return result

  def get_abstract_time_stamps(self):
    cursor = self.conn.cursor()
    while True:
      try:
        cursor.execute("SELECT DISTINCT abstract_timestamp FROM {};".format(
          DBConstants.table_individual.value[0]))
      except db.OperationalError:
        continue
      break
    result = [ts[0] for ts in cursor.fetchall()]
    cursor.close()
    return result

  def get_individual_names_by_abstract_time_stamp(self, time_stamp):
    cursor = self.conn.cursor()
    cursor.execute("SELECT id_name FROM {} WHERE abstract_timestamp=?;".format(
      DBConstants.table_individual.value[0]), [time_stamp])
    result = [name[0] for name in cursor.fetchall()]
    cursor.close()
    return result

  def get_individual_metrics(self, name):
    with open(self._path + '/' + name + '.pb', 'rb') as f:
      last = f.read()
    proto = NetworkIndividualProto()
    # proto = IndividualProto()
    proto.ParseFromString(last)
    return dict([(m.id_name, m.value) for m in proto.baseIndividual.metrics])
    # return dict([(m.id_name, m.value) for m in proto.metrics])

  def get_individual_functions(self, name):
    with open(self._path + '/' + name + '.pb', 'rb') as f:
      last = f.read()
    proto = NetworkIndividualProto()
    proto.ParseFromString(last)
    return [f.id_name for f in proto.networks[0].functions]
