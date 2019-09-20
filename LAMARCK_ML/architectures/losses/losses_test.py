import unittest

from LAMARCK_ML.architectures.losses import LossInterface, SoftmaxCrossEntropyWithLogits, \
  SparseSoftmaxCrossEntropyWithLogits, Reduce


class TestLoss(unittest.TestCase):
  def test_pb_scewl(self):
    for reduce in Reduce:
      obj = SoftmaxCrossEntropyWithLogits(**{
        LossInterface.arg_REDUCE: reduce
      })
      pb = obj.get_pb()
      self.assertIsNotNone(pb)
      state = obj.__getstate__()
      self.assertIsNotNone(state)

      pb_obj = SoftmaxCrossEntropyWithLogits.__new__(SoftmaxCrossEntropyWithLogits)
      pb_obj.__setstate__(pb)
      self.assertIsNotNone(pb_obj)
      self.assertIsNot(obj, pb_obj)

      state_obj = SoftmaxCrossEntropyWithLogits.__new__(SoftmaxCrossEntropyWithLogits)
      state_obj.__setstate__(state)
      self.assertIsNotNone(state_obj)
      self.assertIsNot(obj, state_obj)
    pass

  def test_pb_sscewl(self):
    for reduce in Reduce:
      obj = SparseSoftmaxCrossEntropyWithLogits(**{
        LossInterface.arg_REDUCE: reduce
      })
      pb = obj.get_pb()
      self.assertIsNotNone(pb)
      state = obj.__getstate__()
      self.assertIsNotNone(state)

      pb_obj = SparseSoftmaxCrossEntropyWithLogits.__new__(SparseSoftmaxCrossEntropyWithLogits)
      pb_obj.__setstate__(pb)
      self.assertIsNotNone(pb_obj)
      self.assertIsNot(obj, pb_obj)

      state_obj = SparseSoftmaxCrossEntropyWithLogits.__new__(SparseSoftmaxCrossEntropyWithLogits)
      state_obj.__setstate__(state)
      self.assertIsNotNone(state_obj)
      self.assertIsNot(obj, state_obj)
    pass

  pass
