import unittest

import numpy as np

from LAMARCK_ML.architectures.variables import Variable
from LAMARCK_ML.architectures.variables.initializer import GlorotUniform, Constant
from LAMARCK_ML.architectures.variables.regularisation import L1, L2
from LAMARCK_ML.data_util.dataType import DDouble, DFloat


class TestVariable(unittest.TestCase):

  def test_serialization_str(self):
    for trainable in [True, False]:
      a = Variable(dtype=DDouble, value=np.random.random((3, 7, 4, 9)), trainable=trainable,
                   initializer=GlorotUniform(), regularisation=L1())
      b = Variable.__new__(Variable)
      b.__setstate__(a.__getstate__())
      self.assertEqual(a, b)
      self.assertEqual(b, a)
      b.trainable = not b.trainable
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.trainable = a.trainable
      self.assertEqual(b, a)
      b.dtype = DFloat
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.dtype = a.dtype
      self.assertEqual(b, a)
      b.value = np.random.random((3, 7, 4, 9))
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.value = a.value
      self.assertEqual(b, a)
      b.regularisation = L2()
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.regularisation = L1()
      self.assertEqual(b, a)
      b.initializer = Constant()
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.initializer = GlorotUniform()
      self.assertEqual(b, a)

  def test_serialization_pb(self):
    for trainable in [True, False]:
      a = Variable(dtype=DDouble, value=np.random.random((3, 7, 4, 9)), trainable=trainable,
                   initializer=GlorotUniform(), regularisation=L1())
      b = Variable.__new__(Variable)
      b.__setstate__(a.get_pb())
      self.assertEqual(a, b)
      self.assertEqual(b, a)
      b.trainable = not b.trainable
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.trainable = a.trainable
      self.assertEqual(b, a)
      b.dtype = DFloat
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.dtype = a.dtype
      self.assertEqual(b, a)
      b.value = np.random.random((3, 7, 4, 9))
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.value = a.value
      self.assertEqual(b, a)
      b.regularisation = L2()
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.regularisation = L1()
      self.assertEqual(b, a)
      b.initializer = Constant()
      self.assertNotEqual(a, b)
      self.assertNotEqual(b, a)
      b.initializer = GlorotUniform()
      self.assertEqual(b, a)

  pass
