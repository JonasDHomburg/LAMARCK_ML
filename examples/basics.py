from LAMARCK_ML.models import GenerationalModel
from LAMARCK_ML.metrics import Accuracy, FlOps
from LAMARCK_ML.selection import TournamentSelection, ExponentialRankingSelection
from LAMARCK_ML.reproduction import Mutation, Recombination
from LAMARCK_ML.replacement import NElitism
from LAMARCK_ML.utils.evaluation import LocalEH
from LAMARCK_ML.utils.stopGenerational import StopByGenerationIndex
from LAMARCK_ML.utils.modelStateSaverLoader import ModelStateSaverLoader
from LAMARCK_ML.utils.dataSaver import DSSqlite3
from LAMARCK_ML.models.initialization import RandomInitializer
from LAMARCK_ML.individuals import ClassifierIndividualOPACDG
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.data_util import IOLabel, DFloat, DimNames, Shape, TypeShape
from LAMARCK_ML.nn_framework.nvidia_tensorflow import NVIDIATensorFlow
from LAMARCK_ML.architectures.functions import Dense, Merge

import numpy as np
from sklearn.datasets import make_classification
import os


def ind_cmp(this, other):
  t_m = this.metrics
  o_m = other.metrics
  acc_diff = t_m[Accuracy.ID] - o_m[Accuracy.ID]
  return int(acc_diff > 0)


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
  RandomInitializer(**{
    RandomInitializer.arg_GEN_SIZE: 10,
    RandomInitializer.arg_CLASS: ClassifierIndividualOPACDG,
    RandomInitializer.arg_PARAM: {
    ClassifierIndividualOPACDG.arg_MIN_NN_DEPTH: 1,
    ClassifierIndividualOPACDG.arg_MAX_NN_DEPTH: 4,
    ClassifierIndividualOPACDG.arg_MAX_NN_BRANCH: 1,
    ClassifierIndividualOPACDG.arg_DATA_NTS: dict([(ts_label, (ts, dataset.id_name))
                                                   for ts_label, ts in dataset.outputs.items()]),
    ClassifierIndividualOPACDG.arg_NN_FUNCTIONS: [Dense, Merge]},
  }),
  LocalEH(**{LocalEH.arg_NN_FRAMEWORK: NVIDIATensorFlow(
    **{NVIDIATensorFlow.arg_DATA_SETS: [dataset],
       }
  )}),
  Accuracy,
  FlOps,
  # TournamentSelection(),
  ExponentialRankingSelection(**{
    ExponentialRankingSelection.arg_CMP: ind_cmp
  }),
  Recombination(),
  Mutation(**{Mutation.arg_P: 0.05}),
  # Recombination(**{Recombination.arg_DESCENDANTS: 1}),
  NElitism(**{
    NElitism.arg_CMP: ind_cmp,
    NElitism.arg_N: 3,
  }),
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
  # model.run()

  os.remove('checkpoint')
  os.remove('state.ckpt.index')
  os.remove('state.ckpt.data-00000-of-00002')
  os.remove('state.ckpt.data-00001-of-00002')
else:
  print('Model failed!')
