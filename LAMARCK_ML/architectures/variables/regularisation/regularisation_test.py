import random
import unittest

from LAMARCK_ML.architectures.variables.regularisation import Regularisation
from LAMARCK_ML.data_util import Shape, DimNames as DN, DDouble


class TestRegularisation(unittest.TestCase):
  def subclassTest(self, cls):
    obj = cls()
    pb = obj.get_pb()
    self.assertIsNotNone(pb)
    new_obj = Regularisation.__new__(Regularisation)
    new_obj.__setstate__(pb)
    self.assertEqual(obj, new_obj)
    self.assertIsNot(obj, new_obj)
    state = obj.__getstate__()
    new_obj = Regularisation.__new__(Regularisation)
    new_obj.__setstate__(state)
    self.assertEqual(obj, new_obj)
    self.assertIsNot(obj, new_obj)

    obj.attr = {'int_val': random.randint(0, 100),
                'float_val': random.random(),
                'string_val': "asdfasdf",
                'bool_val': True,
                'bool_val2': False,
                'bytes_val': random.randint(0, 100).to_bytes(10, byteorder='big'),
                'shape_val': Shape((DN.BATCH, -1),
                                   (DN.WIDTH, 16),
                                   (DN.HEIGHT, 16),
                                   (DN.WIDTH, 3)),
                'dtype_val': DDouble,
                'list_val': [1, 2, 3, 4],
                }
    pb = obj.get_pb()
    self.assertIsNotNone(pb)
    new_obj = Regularisation.__new__(Regularisation)
    new_obj.__setstate__(pb)
    self.assertEqual(obj, new_obj)
    self.assertIsNot(obj, new_obj)
    state = obj.__getstate__()
    new_obj = Regularisation.__new__(Regularisation)
    new_obj.__setstate__(state)
    self.assertEqual(obj, new_obj)
    self.assertIsNot(obj, new_obj)

    del obj.attr[obj.attr.keys().__iter__().__next__()]
    self.assertNotEqual(obj, new_obj)

  def test_classes(self):
    for c in Regularisation.__subclasses__():
      self.subclassTest(c)
