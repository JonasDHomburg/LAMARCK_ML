import unittest

import numpy as np

from LAMARCK_ML.data_util import Shape, DimNames, TypeShape, IOLabel, DFloat
from LAMARCK_ML.datasets import UncorrelatedSupervised


class TestUncorrelatedSupervised(unittest.TestCase):
  def test_dataSamplingTrain(self):
    train_samples = 10
    train_X = [np.random.rand(3, 4, 5) for _ in range(train_samples)]
    train_Y = [np.random.rand(2) for _ in range(train_samples)]

    batch = 3
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DEFAULT: TypeShape(DFloat, Shape(
                                       (DimNames.CHANNEL, 3), (DimNames.HEIGHT, 4), (DimNames.WIDTH, 5)))})
    for i in ['train', 'Train', 'TRAIN', 1, True]:
      for idx, d_set in enumerate(dataset(i)):
        self.assertEqual(len(d_set), 2)
        self.assertEqual(len(d_set[IOLabel.DATA]), batch)
        self.assertEqual(len(d_set[IOLabel.TARGET]), batch)
        self.assertTupleEqual(d_set[IOLabel.DATA][0].shape, train_X[0].shape)
        self.assertTupleEqual(d_set[IOLabel.TARGET][0].shape, train_Y[0].shape)
        if idx > 20:
          break
    for i in [{'train': 1}, {'Train': 1}, {'TRAIN': 1},
              {'train': True}, {'Train': True}, {'TRAIN': True}]:
      for idx, d_set in enumerate(dataset(**i)):
        self.assertEqual(len(d_set), 2)
        self.assertEqual(len(d_set[IOLabel.DATA]), batch)
        self.assertEqual(len(d_set[IOLabel.TARGET]), batch)
        self.assertTupleEqual(d_set[IOLabel.DATA][0].shape, train_X[0].shape)
        self.assertTupleEqual(d_set[IOLabel.TARGET][0].shape, train_Y[0].shape)
        if idx > 20:
          break
      pass

  def test_dataSamplingTest(self):
    test_samples = 7
    test_X = [np.random.rand(3, 4, 5) for _ in range(test_samples)]
    test_Y = [np.random.rand(2) for _ in range(test_samples)]

    batch = 3
    dataset = UncorrelatedSupervised(test_X=test_X,
                                     test_Y=test_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DEFAULT: TypeShape(DFloat, Shape(
                                       (DimNames.CHANNEL, 3), (DimNames.HEIGHT, 4), (DimNames.WIDTH, 5)))})
    for i in ['test', 'Test', 'TEST', 0, False]:
      for idx, d_set in enumerate(dataset(i)):
        self.assertEqual(len(d_set), 2)
        self.assertEqual(len(d_set[IOLabel.DATA]), batch)
        self.assertEqual(len(d_set[IOLabel.TARGET]), batch)
        self.assertTupleEqual(d_set[IOLabel.DATA][0].shape, test_X[0].shape)
        self.assertTupleEqual(d_set[IOLabel.TARGET][0].shape, test_Y[0].shape)
        if idx > 20:
          break
    for i in [{'test': 1}, {'Test': 1}, {'TEST': 1},
              {'test': True}, {'Test': True}, {'TEST': True}]:
      for idx, d_set in enumerate(dataset(**i)):
        self.assertEqual(len(d_set), 2)
        self.assertEqual(len(d_set[IOLabel.DATA]), batch)
        self.assertEqual(len(d_set[IOLabel.TARGET]), batch)
        self.assertTupleEqual(d_set[IOLabel.DATA][0].shape, test_X[0].shape)
        self.assertTupleEqual(d_set[IOLabel.TARGET][0].shape, test_Y[0].shape)
        if idx > 20:
          break
    for idx, d_set in enumerate(dataset()):
      self.assertEqual(len(d_set), 2)
      self.assertEqual(len(d_set[IOLabel.DATA]), batch)
      self.assertEqual(len(d_set[IOLabel.TARGET]), batch)
      self.assertTupleEqual(d_set[IOLabel.DATA][0].shape, test_X[0].shape)
      self.assertTupleEqual(d_set[IOLabel.TARGET][0].shape, test_Y[0].shape)
      if idx > 20:
        break

  def test_pb(self):
    train_samples = 10
    train_X = [np.random.rand(3, 4, 5) for _ in range(train_samples)]
    train_Y = [np.random.rand(2) for _ in range(train_samples)]

    batch = 3
    ds0 = UncorrelatedSupervised(train_X=train_X,
                                 train_Y=train_Y,
                                 batch=batch,
                                 typeShapes={IOLabel.DEFAULT: TypeShape(DFloat, Shape(
                                   (DimNames.CHANNEL, 3), (DimNames.HEIGHT, 4), (DimNames.WIDTH, 5)))})
    pb = ds0.get_pb()
    self.assertIsNotNone(pb)
    ds1 = UncorrelatedSupervised.__new__(UncorrelatedSupervised)
    ds1.__setstate__(pb)

    for data0, data1 in [(ds0.train_X, ds1.train_X),
                         (ds0.train_Y, ds1.train_Y),
                         (ds0.test_X, ds1.test_X),
                         (ds0.test_Y, ds1.test_Y)]:
      self.assertTrue(data0 is None and data1 is None or
                      len(data0) == len(data1) and
                      not any([not np.array_equal(_d0, _d1) for _d0, _d1 in zip(data0, data1)]))
    self.assertEqual(ds0.idx, ds1.idx)
    self.assertEqual(ds0.len, ds1.len)
    self.assertEqual(ds0._id_name, ds1._id_name)
    self.assertIsNotNone(ds0, ds1)


#   TODO: test DataFlow overrides and errors

class TestCorrelatedSupervised(unittest.TestCase):
  def test_dataSamplingTrain(self):
    pass

  def test_dataSamplingTest(self):
    pass

  pass
