from LAMARCK_ML.models import GenerationalModel
from LAMARCK_ML.metrics import Accuracy
from LAMARCK_ML.selection import TournamentSelection
from LAMARCK_ML.reproduction import Mutation, Recombination
from LAMARCK_ML.replacement import NElitism
from LAMARCK_ML.utils.evaluation import LocalEH
from LAMARCK_ML.utils.stopGenerational import StopByGenerationIndex
from LAMARCK_ML.utils import ModelStateSaverLoader
from LAMARCK_ML.utils.dataSaver import DSSqlite3
from LAMARCK_ML.models.initialization import SimpleRandomClassifierInitializer
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.data_util import IOLabel, DFloat, DimNames, Shape, TypeShape
from LAMARCK_ML.nn_framework import NVIDIATensorFlow
from LAMARCK_ML.architectures.functions import Dense, Merge

import numpy as np
from sklearn.datasets import make_classification
import os

train_samples = 1000
data_X, data_y = make_classification(n_samples=train_samples,
                                     n_features=20,
                                     n_classes=5,
                                     n_informative=4,
                                     )
data_Y = np.zeros((train_samples, 5))
data_Y[np.arange(train_samples), data_y] = 1

data_X, data_Y = np.asarray(data_X), np.asarray(data_Y)
train_X, test_X = data_X[:int(train_samples * .9), :], data_X[int(train_samples * .9):, :]
train_Y, test_Y = data_Y[:int(train_samples * .9), :], data_Y[int(train_samples * .9):, :]
train_X, valid_X = train_X[:int(train_samples * .75), :], train_X[int(train_samples * .75):, :]
train_Y, valid_Y = train_Y[:int(train_samples * .75), :], train_Y[int(train_samples * .75):, :]

batch = None
dataset = UncorrelatedSupervised(train_X=train_X,
                                 train_Y=train_Y,
                                 test_X=test_X,
                                 test_Y=test_Y,
                                 valid_X=valid_X,
                                 valid_Y=valid_Y,
                                 batch=batch,
                                 typeShapes={IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.UNITS, 20))),
                                             IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 5)))},
                                 name='Dataset')


model = GenerationalModel()
model.add([
  SimpleRandomClassifierInitializer(**{
    SimpleRandomClassifierInitializer.arg_GEN_SIZE: 10,
    SimpleRandomClassifierInitializer.arg_MIN_DEPTH: 1,
    SimpleRandomClassifierInitializer.arg_MAX_DEPTH: 4,
    SimpleRandomClassifierInitializer.arg_MAX_BRANCH: 1,
    SimpleRandomClassifierInitializer.arg_DATA_SHAPES: dict([(ts_label, (ts, dataset.id_name))
                                                 for ts_label, ts in dataset.outputs.items()]),
    SimpleRandomClassifierInitializer.arg_FUNCTIONS: [Dense, Merge]
  }),
  LocalEH(**{LocalEH.arg_NN_FRAMEWORK: NVIDIATensorFlow(
    **{NVIDIATensorFlow.arg_DATA_SETS: [dataset],
       }
  )}),
  Accuracy,
  TournamentSelection(),
  Mutation(),
  Recombination(),
  NElitism(),
  StopByGenerationIndex(**{StopByGenerationIndex.arg_GENERATIONS: 10}),
  ModelStateSaverLoader(**{
    ModelStateSaverLoader.arg_REPRODUCTION: True,
    ModelStateSaverLoader.arg_SELECTION: True,
    ModelStateSaverLoader.arg_REPLACEMENT: True,
  }),
  DSSqlite3(),
])
if model.reset():
  print('Successfully initialized model!')
  model.run()

  os.remove('default.db3')
  os.remove('model_state.pb')
  os.remove('checkpoint')
  os.remove('state.ckpt.index')
  os.remove('state.ckpt.data-00000-of-00002')
  os.remove('state.ckpt.data-00001-of-00002')
else:
  print('Model failed!')
