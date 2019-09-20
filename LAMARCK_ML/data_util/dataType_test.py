import unittest

from LAMARCK_ML.data_util.dataType import BaseType


class TestBaseType(unittest.TestCase):
  def test_eq(self):
    self.assertEqual(BaseType, BaseType)

  def test_hashable(self):
    self.assertTrue(hash(BaseType))

  def test_hash_equal(self):
    self.assertEqual(hash(BaseType), hash(BaseType))


class TestAllTypes(unittest.TestCase):

  def get_all_subTypes(self, _class=BaseType):
    _types = []
    for _subclass in _class.__subclasses__(_class):
      _types.extend(self.get_all_subTypes(_subclass))
      _types.append(_subclass)
    return _types

  def _testType(self, _type):
    self.assertEqual(_type, _type)
    self.assertTrue(issubclass(_type, BaseType))
    self.assertTrue(hash(_type))
    self.assertEqual(hash(_type), hash(_type))
    self.assertIsNotNone(_type.pb)
    self.assertIsNotNone(_type.attr)

  def test_dummpy(self):
    for _type in self.get_all_subTypes():
      self._testType(_type)
