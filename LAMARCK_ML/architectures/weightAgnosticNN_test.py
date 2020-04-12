import unittest
import os
from random import randint, random

from LAMARCK_ML.architectures.weightAgnosticNN import WeightAgnosticNeuralNetwork
from LAMARCK_ML.data_util import DimNames, Shape, TypeShape, DFloat
from LAMARCK_ML.architectures.functions import InvalidFunctionType
from LAMARCK_ML.architectures.functions import Perceptron


@unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
class TestWeightAgnosticNN(unittest.TestCase):
  def test_instantiate(self):
    node_ts = TypeShape(DFloat, Shape((DimNames.BATCH, None),
                                      (DimNames.UNITS, 1)))
    inputs = {'%i:%i:%i' % (w, h, c): ('node', node_ts.__copy__(), 'data_src_%i:%i:%i' % (w, h, c))
              for w in range(5)
              for h in range(7)
              for c in range(3)}
    output_target = {'%02i' % (c): node_ts.__copy__() for c in range(11)}
    self.assertRaises(InvalidFunctionType,
                      WeightAgnosticNeuralNetwork,
                      **{
                        WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
                        WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
                      })

    obj = WeightAgnosticNeuralNetwork(**{
      WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
      WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
      WeightAgnosticNeuralNetwork.arg_FUNCTIONS: [Perceptron],
    })
    self.assertIsNotNone(obj)

    _pb = obj.get_pb()
    self.assertIsNotNone(_pb)
    pb_obj = object.__new__(WeightAgnosticNeuralNetwork)
    pb_obj.__setstate__(_pb)

    self.assertEqual(obj, pb_obj)
    self.assertIsNot(obj, pb_obj)

    _state = obj.__getstate__()
    self.assertIsNotNone(_state)
    state_obj = object.__new__(WeightAgnosticNeuralNetwork)
    state_obj.__setstate__(_state)

    self.assertEqual(obj, state_obj)
    self.assertIsNot(obj, state_obj)

  def test_mutate(self):
    node_ts = TypeShape(DFloat, Shape((DimNames.BATCH, None),
                                      (DimNames.UNITS, 1)))
    inputs = {'%i:%i:%i' % (w, h, c): ('node', node_ts.__copy__(), 'data_src_%i:%i:%i' % (w, h, c))
              for w in range(5)
              for h in range(7)
              for c in range(3)}
    output_target = {'%02i' % (c): node_ts.__copy__() for c in range(10)}
    self.assertRaises(InvalidFunctionType,
                      WeightAgnosticNeuralNetwork,
                      **{
                        WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
                        WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
                      })

    objA = WeightAgnosticNeuralNetwork(**{
      WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
      WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
      WeightAgnosticNeuralNetwork.arg_FUNCTIONS: [Perceptron],
    })
    self.assertIsNotNone(objA)

    mut_obj = objA.mutate(.75)
    self.assertIsInstance(mut_obj, list)
    self.assertGreaterEqual(len(mut_obj), 1)
    mut_obj = mut_obj[0]
    self.assertIsInstance(mut_obj, WeightAgnosticNeuralNetwork)
    for _ in range(20):
      mut_obj = mut_obj.mutate(random())
      self.assertIsInstance(mut_obj, list)
      self.assertGreaterEqual(len(mut_obj), 1)
      mut_obj = mut_obj[0]
      self.assertIsInstance(mut_obj, WeightAgnosticNeuralNetwork)

  def test_step(self):
    node_ts = TypeShape(DFloat, Shape((DimNames.BATCH, None),
                                      (DimNames.UNITS, 1)))
    inputs = {'%i:%i:%i' % (w, h, c): ('node', node_ts.__copy__(), 'data_src_%i:%i:%i' % (w, h, c))
              for w in range(5)
              for h in range(7)
              for c in range(3)}
    output_target = {'%02i' % (c): node_ts.__copy__() for c in range(11)}
    self.assertRaises(InvalidFunctionType,
                      WeightAgnosticNeuralNetwork,
                      **{
                        WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
                        WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
                      })

    objA = WeightAgnosticNeuralNetwork(**{
      WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
      WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
      WeightAgnosticNeuralNetwork.arg_FUNCTIONS: [Perceptron],
    })
    self.assertIsNotNone(objA)

    mut_obj = objA.step(3)
    self.assertIsInstance(mut_obj, list)
    self.assertGreaterEqual(len(mut_obj), 1)
    mut_obj = mut_obj[0]
    self.assertIsInstance(mut_obj, WeightAgnosticNeuralNetwork)
    for _ in range(20):
      mut_obj = mut_obj.step(randint(1, 10))
      self.assertIsInstance(mut_obj, list)
      self.assertGreaterEqual(len(mut_obj), 1)
      mut_obj = mut_obj[0]
      self.assertIsInstance(mut_obj, WeightAgnosticNeuralNetwork)

  def test_recombine(self):
    node_ts = TypeShape(DFloat, Shape((DimNames.BATCH, None),
                                      (DimNames.UNITS, 1)))
    inputs = {'%i:%i:%i' % (w, h, c): ('node', node_ts.__copy__(), 'data_src_%i:%i:%i' % (w, h, c))
              for w in range(5)
              for h in range(7)
              for c in range(3)}
    output_target = {'%02i' % (c): node_ts.__copy__() for c in range(10)}
    self.assertRaises(InvalidFunctionType,
                      WeightAgnosticNeuralNetwork,
                      **{
                        WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
                        WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
                      })

    objA = WeightAgnosticNeuralNetwork(**{
      WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
      WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
      WeightAgnosticNeuralNetwork.arg_FUNCTIONS: [Perceptron],
    })
    self.assertIsNotNone(objA)
    objB = WeightAgnosticNeuralNetwork(**{
      WeightAgnosticNeuralNetwork.arg_INPUTS: inputs,
      WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: output_target,
      WeightAgnosticNeuralNetwork.arg_FUNCTIONS: [Perceptron],
    })
    self.assertIsNotNone(objB)

    rec_objA = objA.recombine(objB)
    self.assertIsInstance(rec_objA, list)
    self.assertGreaterEqual(len(rec_objA), 2)
    rec_objA = rec_objA[0]
    self.assertIsInstance(rec_objA, WeightAgnosticNeuralNetwork)
    rec_objB = objB.recombine(objA)
    self.assertIsInstance(rec_objB, list)
    self.assertGreaterEqual(len(rec_objB), 2)
    rec_objB = rec_objB[0]
    self.assertIsInstance(rec_objB, WeightAgnosticNeuralNetwork)

    for _ in range(20):
      rec_objA = rec_objA.recombine(rec_objB)
      self.assertIsInstance(rec_objA, list)
      self.assertGreaterEqual(len(rec_objA), 2)
      rec_objA = rec_objA[0]
      self.assertIsInstance(rec_objA, WeightAgnosticNeuralNetwork)
      rec_objB = rec_objB.recombine(rec_objA)
      self.assertIsInstance(rec_objB, list)
      self.assertGreaterEqual(len(rec_objB), 2)
      rec_objB = rec_objB[0]
      self.assertIsInstance(rec_objB, WeightAgnosticNeuralNetwork)
