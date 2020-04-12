import unittest

import numpy as np
from random import random, randint

from LAMARCK_ML.data_util import TypeShape, IOLabel, DFloat, Shape, DimNames
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.individuals import \
  ClassifierIndividualACDG, \
  GraphLayoutIndividual, \
  ClassifierIndividualOPACDG, \
  CartesianIndividual, \
  WeightAgnosticIndividual


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

  def test_cartesianIndividual_default(self):
    for _ in range(10):
      ind = CartesianIndividual()
      f = ind.cartesianFitness()
      self.assertIsInstance(f, float)
      self.assertLessEqual(f, 1)
      self.assertGreaterEqual(f, 0)
      self.assertTrue(hasattr(ind, '_id_name'))
      self.assertTrue(hasattr(ind, 'id_name'))

  def test_cartesianIndividual(self):
    for d in range(1, 10):
      _ = CartesianIndividual(**{
        CartesianIndividual.arg_Dimensions: d,
      })
      with self.assertRaises(Exception):
        _ = CartesianIndividual(**{
          CartesianIndividual.arg_Dimensions: d,
          CartesianIndividual.arg_State: d,
        })
      t = CartesianIndividual(**{
        CartesianIndividual.arg_FitnessFunction: d,
      })
      self.assertNotEqual(t.attr[CartesianIndividual.arg_FitnessFunction], d)
      ind0 = CartesianIndividual(**{
        CartesianIndividual.arg_Dimensions: d,
      })
      ind1 = CartesianIndividual(**{
        CartesianIndividual.arg_Dimensions: d,
      })
      ind_c0, ind_c1 = ind0.recombine(ind1)
      self.assertEqual(ind_c0.state.shape, ind0.state.shape)
      self.assertEqual(ind_c1.state.shape, ind0.state.shape)

      ind_m = ind_c0.mutate(.5)[0]
      self.assertEqual(ind_m.state.shape, ind0.state.shape)

      for _ in range(5):
        ind_m = ind_m.mutate(.5)[0]
        self.assertTrue(hasattr(ind_m, '_id_name'))
        self.assertTrue(hasattr(ind_m, 'id_name'))

      self.assertTrue(np.all(ind0 - ind1 == -(ind1 - ind0)))
      self.assertEqual(ind0.norm(ind1), ind1.norm(ind0))
      ind0.metrics = {'test': .5}
      pb = ind0.get_pb()
      self.assertIsNotNone(pb)
      state = ind0.__getstate__()
      self.assertIsNotNone(state)

      state_obj = CartesianIndividual.__new__(CartesianIndividual)
      state_obj.__setstate__(state)
      self.assertIsNotNone(state_obj)
      self.assertIsNot(state_obj, ind0)
      self.assertEqual(state_obj, ind0)

      pb_obj = CartesianIndividual.__new__(CartesianIndividual)
      pb_obj.__setstate__(pb)
      self.assertIsNot(pb_obj, ind0)
      self.assertEqual(pb_obj, ind0)

  def test_weightAgnosticIndividual(self):
    train_samples = 1
    train_X = [np.random.rand(20) for _ in range(train_samples)]
    train_Y = [np.random.rand(10) for _ in range(train_samples)]

    batch = 1
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA: TypeShape(DFloat,
                                                                         Shape((DimNames.UNITS, 5),
                                                                               (DimNames.UNITS, 5))),
                                                 IOLabel.TARGET: TypeShape(DFloat,
                                                                           Shape((DimNames.UNITS, 10)))},
                                     name='Dataset')
    wann = WeightAgnosticIndividual(**{
      WeightAgnosticIndividual.arg_DATA_NTS: {ts_label: (ts, dataset.id_name)
                                              for ts_label, ts in dataset.outputs.items()},
      WeightAgnosticIndividual.arg_INITIAL_DEPTH: 5,
    })
    self.assertIsNotNone(wann)
    wann.metrics['debug'] = .3

    pb = wann.get_pb()
    self.assertIsNotNone(pb)
    state = wann.__getstate__()
    self.assertIsNotNone(state)

    state_obj = WeightAgnosticIndividual.__new__(WeightAgnosticIndividual)
    state_obj.__setstate__(state)
    self.assertIsNotNone(state_obj)
    self.assertIsNot(wann, state_obj)
    self.assertEqual(wann, state_obj)

    pb_obj = WeightAgnosticIndividual.__new__(WeightAgnosticIndividual)
    pb_obj.__setstate__(pb)
    self.assertIsNotNone(pb_obj)
    self.assertIsNot(wann, pb_obj)
    self.assertEqual(wann, pb_obj)

    m_obj = wann.mutate(1)
    self.assertIsInstance(m_obj, list)
    self.assertEqual(len(m_obj), 1)
    m_obj = m_obj[0]
    self.assertIsInstance(m_obj, WeightAgnosticIndividual)
    for _ in range(20):
      m_obj = m_obj.mutate(random())
      self.assertIsInstance(m_obj, list)
      self.assertEqual(len(m_obj), 1)
      m_obj = m_obj[0]
      self.assertIsInstance(m_obj, WeightAgnosticIndividual)

    wann_other = WeightAgnosticIndividual(**{
      WeightAgnosticIndividual.arg_DATA_NTS: {ts_label: (ts, dataset.id_name)
                                              for ts_label, ts in dataset.outputs.items()},
      WeightAgnosticIndividual.arg_INITIAL_DEPTH: 5,
    })

    rec_obj = wann.recombine(wann_other)
    self.assertIsInstance(rec_obj, list)
    self.assertEqual(len(rec_obj), 1)
    rec_A, rec_B = rec_obj[0], wann_other.recombine(wann)[0]
    self.assertIsInstance(rec_A, WeightAgnosticIndividual)
    self.assertIsInstance(rec_B, WeightAgnosticIndividual)
    for _ in range(20):
      rec_obj = rec_A.recombine(rec_B)
      self.assertIsInstance(rec_obj, list)
      self.assertEqual(len(rec_obj), 1)
      rec_A, rec_B = rec_obj[0], rec_B.recombine(rec_A)[0]
      self.assertIsInstance(rec_A, WeightAgnosticIndividual)
      self.assertIsInstance(rec_B, WeightAgnosticIndividual)

    m_step = wann.step(2)
    self.assertIsInstance(m_step, list)
    self.assertEqual(len(m_step), 1)
    m_step = m_step[0]
    self.assertIsInstance(m_step, WeightAgnosticIndividual)
    for _ in range(20):
      m_step = m_step.step(randint(1, 5))
      self.assertIsInstance(m_step, list)
      self.assertEqual(len(m_step), 1)
      m_step = m_step[0]
      self.assertIsInstance(m_step, WeightAgnosticIndividual)
