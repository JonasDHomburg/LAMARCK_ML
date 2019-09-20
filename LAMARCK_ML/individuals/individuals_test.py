import unittest

import numpy as np

from LAMARCK_ML.data_util import TypeShape, IOLabel, DFloat, Shape, DimNames
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.individuals import ClassifierIndividual, IndividualInterface


class TestIndividuals(unittest.TestCase):
  def test_classifier_individual(self):
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
    ci = ClassifierIndividual(**{
      IndividualInterface.arg_DATA_NTS: dict([(ts_label, (ts, dataset.id_name))
                                              for ts_label, ts in dataset.outputs.items()])
    })
    self.assertIsNotNone(ci)
    ci.metrics['debug'] = .3

    pb = ci.get_pb()
    self.assertIsNotNone(pb)
    state = ci.__getstate__()
    self.assertIsNotNone(state)

    pb_obj = ClassifierIndividual.__new__(ClassifierIndividual)
    pb_obj.__setstate__(pb)
    self.assertIsNotNone(pb_obj)
    self.assertIsNot(ci, pb_obj)
    self.assertEqual(ci, pb_obj)

    state_obj = ClassifierIndividual.__new__(ClassifierIndividual)
    state_obj.__setstate__(state)
    self.assertIsNotNone(state_obj)
    self.assertIsNot(ci, state_obj)
    self.assertEqual(ci, pb_obj)

  pass
