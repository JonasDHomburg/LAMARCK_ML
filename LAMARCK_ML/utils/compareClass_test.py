import unittest

from LAMARCK_ML.utils.compareClass import CompareClass

class TestCompareClass(unittest.TestCase):
  def test_comparefunction(self):
    prim = 'a'
    sec0 = 'b'
    sec1 = 'c'
    cmp = CompareClass(**{
      CompareClass.arg_PRIMARY_ALPHA: -1,
      CompareClass.arg_PRIMARY_OBJECTIVE: prim,
      CompareClass.arg_PRIMARY_THRESHOLD: 0.5,
      CompareClass.arg_SECONDARY_OBJECTIVES: {sec0: -10/0.1,
                                              sec1: -20/0.1}
    })

    one = {prim: -0.7,
           sec0: 10000,
           sec1: 40}
    other = {prim: -0.6,
             sec0: 10000,
             sec1: 30}
    self.assertTrue(cmp.greaterThan(one, other))
    self.assertFalse(cmp.greaterThan(other, one))