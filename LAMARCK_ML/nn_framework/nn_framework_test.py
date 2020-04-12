import unittest
import os

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification

from LAMARCK_ML.data_util import TypeShape, IOLabel, DFloat, Shape, DimNames
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.individuals import ClassifierIndividualACDG, ClassifierIndividualOPACDG
if tf.__version__ == '1.12.0':
  from LAMARCK_ML.nn_framework.nvidia_tensorflow_1_12_0 import NVIDIATensorFlow
else:
  from LAMARCK_ML.nn_framework.nvidia_tensorflow import NVIDIATensorFlow
from LAMARCK_ML.architectures.functions import *


@unittest.skipIf((os.environ.get('test_fast', False) in {'True','true', '1'}), 'time consuming')
class TestNVIDIATensorFlowFramework(unittest.TestCase):
  @unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
  def test_MLP_Dense_Merge(self):
    train_samples = 1000
    data_X, data_Y = make_classification(n_samples=train_samples,
                                         n_features=20,
                                         n_classes=5,
                                         n_informative=4,
                                         )
    data_Y = tf.keras.utils.to_categorical(data_Y)
    data_X, data_Y = np.asarray(data_X), np.asarray(data_Y)
    train_X, test_X = data_X[:int(train_samples * .9), :], data_X[int(train_samples * .9):, :]
    train_Y, test_Y = data_Y[:int(train_samples * .9), :], data_Y[int(train_samples * .9):, :]

    batch = None
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     test_X=test_X,
                                     test_Y=test_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.UNITS, 20))),
                                                 IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 5)))},
                                     name='Dataset')

    ci = ClassifierIndividualACDG(**{
      ClassifierIndividualACDG.arg_DATA_NTS: dict(
        [(label, (nts, dataset.id_name)) for label, nts in dataset.outputs.items()]),
      ClassifierIndividualACDG.arg_NN_FUNCTIONS: [Dense, Merge],
    })
    NN = ci.network
    f_ids = dict([(_id, None) for _, _id in NN.inputs.values()])
    for _f in NN.functions:
      f_ids[_f.id_name] = _f

    for _f in NN.functions:
      for _f_input, (other_output, other_id) in _f.inputs.items():
        if other_id not in f_ids:
          self.assertTrue(False)

    stack = [f_id for _, f_id in NN.output_mapping.values()]
    required_ids = set()
    while stack:
      f_id = stack.pop()
      required_ids.add(f_id)
      f_ = f_ids.get(f_id)
      if f_ is not None:
        stack.extend([f_id for _, f_id in f_.inputs.values()])
    self.assertSetEqual(required_ids, set(f_ids.keys()))

    framework = NVIDIATensorFlow(**{
      NVIDIATensorFlow.arg_DATA_SETS: [dataset],
    })

    ci.build_instance(framework)
    framework.accuracy(ci)
    framework.time()
    framework.memory()
    # framework.flops_per_sample()
    # framework.parameters()
    framework.reset()

  @unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
  def test_MLP_Dense_Merge_mutate(self):
    train_samples = 1000
    data_X, data_Y = make_classification(n_samples=train_samples,
                                         n_features=20,
                                         n_classes=5,
                                         n_informative=4,
                                         )
    data_Y = tf.keras.utils.to_categorical(data_Y)
    data_X, data_Y = np.asarray(data_X), np.asarray(data_Y)
    train_X, test_X = data_X[:int(train_samples * .9), :], data_X[int(train_samples * .9):, :]
    train_Y, test_Y = data_Y[:int(train_samples * .9), :], data_Y[int(train_samples * .9):, :]

    batch = None
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     test_X=test_X,
                                     test_Y=test_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.UNITS, 20))),
                                                 IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 5)))},
                                     name='Dataset')

    ci = ClassifierIndividualACDG(**{
      ClassifierIndividualACDG.arg_DATA_NTS: dict(
        [(label, (nts, dataset.id_name)) for label, nts in dataset.outputs.items()]),
      ClassifierIndividualACDG.arg_NN_FUNCTIONS: [Dense, Merge],
    })
    ci = ci.mutate(1)[0]
    NN = ci.network
    f_ids = dict([(_id, None) for _, _id in NN.inputs.values()])
    for _f in NN.functions:
      f_ids[_f.id_name] = _f

    for _f in NN.functions:
      for _f_input, (other_output, other_id) in _f.inputs.items():
        if other_id not in f_ids:
          self.assertTrue(False)

    stack = [f_id for _, f_id in NN.output_mapping.values()]
    required_ids = set()
    while stack:
      f_id = stack.pop()
      required_ids.add(f_id)
      f_ = f_ids.get(f_id)
      if f_ is not None:
        stack.extend([f_id for _, f_id in f_.inputs.values()])
    self.assertSetEqual(required_ids, set(f_ids.keys()))

    framework = NVIDIATensorFlow(**{
      NVIDIATensorFlow.arg_DATA_SETS: [dataset],
    })


    ci.build_instance(framework)
    framework.accuracy(ci)
    framework.time()
    framework.memory()
    # framework.flops_per_sample()
    # framework.parameters()
    framework.reset()

  @unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
  def test_Conv_Flatten_Pool_Dense_Merge(self):
    train_samples = 1000
    data_X, data_Y = make_classification(n_samples=train_samples,
                                         n_features=3072,
                                         n_classes=5,
                                         n_informative=4,
                                         )
    data_X = data_X.reshape((train_samples, 32, 32, 3))
    data_Y = tf.keras.utils.to_categorical(data_Y)
    data_X, data_Y = np.asarray(data_X), np.asarray(data_Y)
    train_X, test_X = data_X[:int(train_samples * .9), :], data_X[int(train_samples * .9):, :]
    train_Y, test_Y = data_Y[:int(train_samples * .9), :], data_Y[int(train_samples * .9):, :]

    batch = None
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     test_X=test_X,
                                     test_Y=test_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.HEIGHT, 32),
                                                                                       (DimNames.WIDTH, 32),
                                                                                       (DimNames.CHANNEL, 3))),
                                                 IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 5)))},
                                     name='Dataset')

    ci = ClassifierIndividualACDG(**{
      ClassifierIndividualACDG.arg_DATA_NTS: dict(
        [(label, (nts, dataset.id_name)) for label, nts in dataset.outputs.items()]),
      ClassifierIndividualACDG.arg_NN_FUNCTIONS: [Conv2D, Flatten, Dense, Merge],
      ClassifierIndividualACDG.arg_MAX_NN_DEPTH: 10,
    })

    framework = NVIDIATensorFlow(**{
      NVIDIATensorFlow.arg_DATA_SETS: [dataset],
    })

    ci.build_instance(framework)
    framework.accuracy(ci)
    framework.time()
    framework.memory()
    # framework.flops_per_sample()
    # framework.parameters()
    framework.reset()

  @unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
  def test_Conv_Flatten_Pool_Dense_Merge_mutate_recombine(self):
    train_samples = 1000
    data_X, data_Y = make_classification(n_samples=train_samples,
                                         n_features=3072,
                                         n_classes=5,
                                         n_informative=4,
                                         )
    data_X = data_X.reshape((train_samples, 32, 32, 3))
    data_Y = tf.keras.utils.to_categorical(data_Y)
    data_X, data_Y = np.asarray(data_X), np.asarray(data_Y)
    train_X, test_X = data_X[:int(train_samples * .9), :], data_X[int(train_samples * .9):, :]
    train_Y, test_Y = data_Y[:int(train_samples * .9), :], data_Y[int(train_samples * .9):, :]

    batch = None
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     test_X=test_X,
                                     test_Y=test_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.HEIGHT, 32),
                                                                                       (DimNames.WIDTH, 32),
                                                                                       (DimNames.CHANNEL, 3))),
                                                 IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 5)))},
                                     name='Dataset')

    ci = ClassifierIndividualACDG(**{
      ClassifierIndividualACDG.arg_DATA_NTS: dict(
        [(label, (nts, dataset.id_name)) for label, nts in dataset.outputs.items()]),
      ClassifierIndividualACDG.arg_NN_FUNCTIONS: [Conv2D, Pooling2D, Flatten, Dense, Merge],
      ClassifierIndividualACDG.arg_MAX_NN_DEPTH: 10,
    })
    ci = ci.mutate(1)[0]

    framework = NVIDIATensorFlow(**{
      NVIDIATensorFlow.arg_DATA_SETS: [dataset],
    })

    ci.build_instance(framework)
    state = ci.train_instance(framework)
    ci.update_state(**state)

    self.assertTrue(isinstance(framework.accuracy(None), float))
    self.assertTrue(isinstance(framework.time(), float))
    self.assertTrue(isinstance(framework.memory(), float))
    # self.assertTrue(isinstance(framework.flops_per_sample(), float))
    # self.assertTrue(isinstance(framework.parameters(), float))

    framework.reset()

    self.assertGreater(len(ci.network.variable_pool), 0)

    ci2 = ClassifierIndividualACDG(**{
      ClassifierIndividualACDG.arg_DATA_NTS: dict(
        [(label, (nts, dataset.id_name)) for label, nts in dataset.outputs.items()]),
      ClassifierIndividualACDG.arg_NN_FUNCTIONS: [Conv2D, Pooling2D, Flatten, Dense, Merge],
      ClassifierIndividualACDG.arg_MAX_NN_DEPTH: 10,
    })

    ci.build_instance(framework)
    framework.reset()

    ci_rec = ci.recombine(ci2)[0]
    self.assertGreater(len(ci_rec.network.variable_pool), 0)
