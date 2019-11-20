import os
import types
import unittest

from LAMARCK_ML.data_util import TypeShape, IOLabel, DFloat, Shape, DimNames
from LAMARCK_ML.individuals import ClassifierIndividualOPACDG, NetworkIndividualInterface
from LAMARCK_ML.models.models import GenerationalModel
from LAMARCK_ML.reproduction import Mutation, Recombination, AncestryEntity
from LAMARCK_ML.utils.dataSaver.dbSqlite3 import DSSqlite3


@unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
class TestDBSqlite3(unittest.TestCase):
  class dummyModel(GenerationalModel):
    def __init__(self, **kwargs):
      super(TestDBSqlite3.dummyModel, self).__init__(**kwargs)
      _data_nts = TypeShape(DFloat, Shape((DimNames.BATCH, 1), (DimNames.UNITS, 20)))
      _target_nts = TypeShape(DFloat, Shape((DimNames.BATCH, 1), (DimNames.UNITS, 10)))

      self.ci = ClassifierIndividualOPACDG(**{
        NetworkIndividualInterface.arg_DATA_NTS: {IOLabel.DATA: (_data_nts, 'Dataset'),
                                                  IOLabel.TARGET: (_target_nts, 'Dataset')},
      })
      self.anc1 = ClassifierIndividualOPACDG(**{
        NetworkIndividualInterface.arg_DATA_NTS: {IOLabel.DATA: (_data_nts, 'Dataset'),
                                                  IOLabel.TARGET: (_target_nts, 'Dataset')},
      })
      self.anc2 = ClassifierIndividualOPACDG(**{
        NetworkIndividualInterface.arg_DATA_NTS: {IOLabel.DATA: (_data_nts, 'Dataset'),
                                                  IOLabel.TARGET: (_target_nts, 'Dataset')},
      })

      self._GENERATION = [self.ci]
      self._GENERATION_IDX = 1

    def mut(self):
      mut = Mutation()
      self._REPRODUCTION = [(mut, [AncestryEntity(mut.ID, self.anc1.id_name, [self.ci.id_name]),
                                   AncestryEntity(mut.ID, self.anc2.id_name, [self.ci.id_name])])]

    def rec(self):
      rec = Recombination()
      self._REPRODUCTION = [(rec, [AncestryEntity(rec.ID, self.ci.id_name, [self.anc1.id_name, self.anc2.id_name])])]

  def test_db_generation(self):
    db_file = './test_db_gen.db3'
    ds = DSSqlite3(**{
      DSSqlite3.arg_FILE: db_file,
    })

    dummyM = TestDBSqlite3.dummyModel()
    setattr(dummyM, '_end_evaluate', types.MethodType(ds.end_evaluate(
      getattr(dummyM, '_end_evaluate')), dummyM))

    dummyM._end_evaluate()
    origin = dummyM.generation[0]
    ind = ds.get_individual_by_name(origin.id_name)
    self.assertEqual(origin, ind)
    self.assertIsNot(origin, ind)
    os.remove(db_file)

  def test_db_ancestry_mut(self):
    db_file = './test_db_anc_mut.db3'
    ds = DSSqlite3(**{
      DSSqlite3.arg_FILE: db_file,
    })
    dummyM = TestDBSqlite3.dummyModel()
    setattr(dummyM, '_end_reproduce', types.MethodType(ds.end_reproduce(
      getattr(dummyM, '_end_reproduce')), dummyM))

    dummyM.mut()
    dummyM._end_reproduce()
    _, anc_ent = ds.get_ancestry_for_ind(dummyM.anc1.id_name)
    self.assertEqual(anc_ent.method, Mutation.ID)
    self.assertEqual(anc_ent.descendant, dummyM.anc1.id_name)
    self.assertListEqual(anc_ent.ancestors, [dummyM.ci.id_name])

    _, anc_ent = ds.get_ancestry_for_ind(dummyM.anc2.id_name)
    self.assertEqual(anc_ent.method, Mutation.ID)
    self.assertEqual(anc_ent.descendant, dummyM.anc2.id_name)
    self.assertListEqual(anc_ent.ancestors, [dummyM.ci.id_name])

    self.assertEqual(ds.get_ancestry_for_ind(dummyM.ci.id_name), (None, None))
    os.remove(db_file)

  def test_db_ancestry_rec(self):
    db_file = './test_db_anc_rec.db3'
    ds = DSSqlite3(**{
      DSSqlite3.arg_FILE: db_file,
    })
    dummyM = TestDBSqlite3.dummyModel()
    setattr(dummyM, '_end_reproduce', types.MethodType(ds.end_reproduce(
      getattr(dummyM, '_end_reproduce')), dummyM))

    dummyM.rec()
    dummyM._end_reproduce()
    self.assertEqual(ds.get_ancestry_for_ind(dummyM.anc1.id_name), (None, None))
    self.assertEqual(ds.get_ancestry_for_ind(dummyM.anc2.id_name), (None, None))
    _, anc_ent = ds.get_ancestry_for_ind(dummyM.ci.id_name)
    self.assertIsNotNone(anc_ent)
    self.assertEqual(anc_ent.method, Recombination.ID)
    self.assertEqual(anc_ent.descendant, dummyM.ci.id_name)
    self.assertListEqual(anc_ent.ancestors, [dummyM.anc1.id_name, dummyM.anc2.id_name])

    os.remove(db_file)
