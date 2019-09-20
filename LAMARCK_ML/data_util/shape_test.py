import unittest

from LAMARCK_ML.data_util.shape import Shape

DimNames = Shape.Dim.Names


class TestShape(unittest.TestCase):
  def test_pb(self):
    shape0 = Shape((DimNames.BATCH, 16), (DimNames.CHANNEL, 3),
                   (DimNames.HEIGHT, 48), (DimNames.WIDTH, 32),
                   (DimNames.UNITS, 10), (DimNames.TIME, 13))
    self.assertIsNotNone(shape0.get_pb())
    self.assertIsInstance(shape0.__getstate__(), bytes)
    shape1 = Shape.__new__(Shape)
    shape1.__setstate__(shape0.get_pb())
    self.assertEqual(shape0, shape1)
    self.assertEqual(shape1, shape0)

  def test_state(self):
    shape0 = Shape((DimNames.BATCH, 16), (DimNames.CHANNEL, 3),
                   (DimNames.HEIGHT, 48), (DimNames.WIDTH, 32),
                   (DimNames.UNITS, 10), (DimNames.TIME, 13))
    state = shape0.__getstate__()
    self.assertIsNotNone(state)
    shape1 = Shape.__new__(Shape)
    shape1.__setstate__(state)
    self.assertEqual(shape0, shape1)
    self.assertEqual(shape1, shape0)

  def test_copy(self):
    shape0 = Shape((DimNames.BATCH, 16), (DimNames.CHANNEL, 3),
                   (DimNames.HEIGHT, 48), (DimNames.WIDTH, 32),
                   (DimNames.UNITS, 10), (DimNames.TIME, 13))
    shape1 = shape0.__copy__()
    self.assertEqual(shape0, shape1)
    self.assertEqual(shape1, shape0)
    self.assertIsNot(shape0, shape1)

  def test_hash(self):
    shape0 = Shape((DimNames.BATCH, 16), (DimNames.CHANNEL, 3),
                   (DimNames.HEIGHT, 48), (DimNames.WIDTH, 32),
                   (DimNames.UNITS, 10), (DimNames.TIME, 13))
    hash0 = shape0.__hash__()
    shape1 = shape0.__copy__()
    hash1 = shape1.__hash__()
    self.assertIsInstance(hash0, int)
    self.assertIsInstance(hash1, int)
    self.assertEqual(hash0, hash1)
    self.assertEqual(hash1, hash0)

  def test_as_dict(self):
    shape0 = Shape((DimNames.BATCH, 16), (DimNames.CHANNEL, 3),
                   (DimNames.HEIGHT, 48), (DimNames.WIDTH, 32),
                   (DimNames.UNITS, 10), (DimNames.TIME, 13))
    ref = {DimNames.BATCH: 16, DimNames.CHANNEL: 3,
           DimNames.HEIGHT: 48, DimNames.WIDTH: 32,
           DimNames.UNITS: 10, DimNames.TIME: 13}
    self.assertDictEqual(shape0.as_dict, ref)
    self.assertDictEqual(ref, shape0.as_dict)
