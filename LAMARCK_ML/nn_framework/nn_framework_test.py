import unittest
import os

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification

from LAMARCK_ML.data_util import TypeShape, IOLabel, DFloat, Shape, DimNames
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.individuals import ClassifierIndividual
from LAMARCK_ML.nn_framework import NVIDIATensorFlow

@unittest.skipIf((os.environ.get('test_fast', False) in {'True','true', '1'}), 'time consuming')
class TestNVIDIATensorFlowFramework(unittest.TestCase):
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

    ci = ClassifierIndividual(**{
      ClassifierIndividual.arg_DATA_NTS: dict([(label, (nts, dataset.id_name)) for label, nts in dataset.outputs.items()])
    })
    ci = ci.mutate(1)[0]

    framework = NVIDIATensorFlow(**{
      NVIDIATensorFlow.arg_DATA_SETS: [dataset],
    })

    framework.setup_individual(ci)

    framework.accuracy()
    framework.time()
    framework.memory()
    framework.flops_per_sample()
    framework.parameters()

    framework.teardown_individual()

  def test_Conv_Flatten_Pool_Dense_Merge(self):
    pass
