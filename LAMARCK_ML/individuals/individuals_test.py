import unittest

import numpy as np

from LAMARCK_ML.data_util import TypeShape, IOLabel, DFloat, Shape, DimNames
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.individuals import \
  ClassifierIndividualACDG, \
  GraphLayoutIndividual, \
  ClassifierIndividualOPACDG


class TestIndividuals(unittest.TestCase):
  def test_classifier_individualACDG(self):
    train_samples = 1
    train_X = [np.random.rand(20) for _ in range(train_samples)]
    train_Y = [np.random.rand(10) for _ in range(train_samples)]

    batch = 1
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.UNITS, 20))),
                                                 IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 10)))},
                                     name='Dataset')
    ci = ClassifierIndividualACDG(**{
      ClassifierIndividualACDG.arg_DATA_NTS: dict([(ts_label, (ts, dataset.id_name))
                                                     for ts_label, ts in dataset.outputs.items()])
    })
    self.assertIsNotNone(ci)
    ci.metrics['debug'] = .3

    pb = ci.get_pb()
    self.assertIsNotNone(pb)
    state = ci.__getstate__()
    self.assertIsNotNone(state)

    state_obj = ClassifierIndividualACDG.__new__(ClassifierIndividualACDG)
    state_obj.__setstate__(state)
    self.assertIsNotNone(state_obj)
    self.assertIsNot(ci, state_obj)
    self.assertEqual(ci, state_obj)

    pb_obj = ClassifierIndividualACDG.__new__(ClassifierIndividualACDG)
    pb_obj.__setstate__(pb)
    self.assertIsNotNone(pb_obj)
    self.assertIsNot(ci, pb_obj)
    self.assertEqual(ci, pb_obj)

  def test_classifier_individualOPACDG(self):
    train_samples = 1
    train_X = [np.random.rand(20) for _ in range(train_samples)]
    train_Y = [np.random.rand(10) for _ in range(train_samples)]

    batch = 1
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.UNITS, 20))),
                                                 IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 10)))},
                                     name='Dataset')
    ci = ClassifierIndividualOPACDG(**{
      ClassifierIndividualOPACDG.arg_DATA_NTS: dict([(ts_label, (ts, dataset.id_name))
                                                     for ts_label, ts in dataset.outputs.items()])
    })
    self.assertIsNotNone(ci)
    ci.metrics['debug'] = .3

    pb = ci.get_pb()
    self.assertIsNotNone(pb)
    state = ci.__getstate__()
    self.assertIsNotNone(state)

    state_obj = ClassifierIndividualOPACDG.__new__(ClassifierIndividualOPACDG)
    state_obj.__setstate__(state)
    self.assertIsNotNone(state_obj)
    self.assertIsNot(ci, state_obj)
    self.assertEqual(ci, state_obj)

    pb_obj = ClassifierIndividualOPACDG.__new__(ClassifierIndividualOPACDG)
    pb_obj.__setstate__(pb)
    self.assertIsNotNone(pb_obj)
    self.assertIsNot(ci, pb_obj)
    self.assertEqual(ci, pb_obj)

  def test_graphLayoutIndividual(self):
    edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (4, 7), (5, 7), (2, 7)]
    g0 = GraphLayoutIndividual(**{
      GraphLayoutIndividual.arg_EDGES: edges
    })
    self.assertTrue(isinstance(g0.layoutCrossingEdges(), float))
    self.assertTrue(isinstance(g0.layoutDistanceX(), float))
    self.assertTrue(isinstance(g0.layoutDistanceY(), float))
    g1 = GraphLayoutIndividual(**{
      GraphLayoutIndividual.arg_EDGES: edges
    })
    mut = g0.mutate(1)
    self.assertTrue(isinstance(mut, list))
    self.assertNotEqual(g0, mut[0])

    rec = g0.recombine(g1)
    self.assertTrue(isinstance(rec, list))
    self.assertNotEqual(g0, rec[0])

    pb_obj = GraphLayoutIndividual.__new__(GraphLayoutIndividual)
    pb_obj.__setstate__(g0.get_pb())
    self.assertEqual(g0, pb_obj)

    state_obj = GraphLayoutIndividual.__new__(GraphLayoutIndividual)
    state_obj.__setstate__(g1.__getstate__())
    self.assertEqual(g1, state_obj)
