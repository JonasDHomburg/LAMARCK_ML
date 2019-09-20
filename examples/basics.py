from LAMARCK_ML.models import GenerationalModel
from LAMARCK_ML.individuals import ClassifierIndividual
from LAMARCK_ML.metrics import Accuracy
from LAMARCK_ML.selection import TournamentSelection
from LAMARCK_ML.reproduction import Mutation
from LAMARCK_ML.replacement import NElitism
from LAMARCK_ML.utils.evaluation import LocalEH
from LAMARCK_ML.utils.stopGenerational import StopByGenerationIndex
from LAMARCK_ML.utils import ModelStateSaverLoader
from LAMARCK_ML.utils.dataSaver import DSSqlite3
from LAMARCK_ML.models.initialization import InitializationStrategyInterface
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.data_util import IOLabel, DFloat, DimNames, Shape, TypeShape
from LAMARCK_ML.nn_framework import NVIDIATensorFlow
from LAMARCK_ML.architectures.functions import Dense, Merge

import numpy as np
from sklearn.datasets import make_classification

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

batch = None
dataset = UncorrelatedSupervised(train_X=train_X,
                                 train_Y=train_Y,
                                 test_X=test_X,
                                 test_Y=test_Y,
                                 batch=batch,
                                 typeShapes={IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.UNITS, 20))),
                                             IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 5)))},
                                 name='Dataset')


class Initialization(InitializationStrategyInterface):
  def seed_generation(self, func):
    def wrapper(model):
      model.generation = [ClassifierIndividual(**{
        ClassifierIndividual.arg_DATA_NTS: dict([(ts_label, (ts, dataset.id_name))
                                                 for ts_label, ts in dataset.outputs.items()]),
      ClassifierIndividual.arg_MAX_NN_BRANCH:1,
      ClassifierIndividual.arg_MIN_NN_DEPTH:1,
      ClassifierIndividual.arg_MAX_NN_DEPTH:4,
      ClassifierIndividual.arg_NN_FUNCTIONS:[Dense, Merge]})
                           for _ in range(10)]
      func()
    return wrapper


model = GenerationalModel()
model.add([
  Initialization(),
  LocalEH(**{LocalEH.arg_NN_FRAMEWORK: NVIDIATensorFlow(
    **{NVIDIATensorFlow.arg_DATA_SETS: [dataset],
       }
  )}),
  Accuracy,
  TournamentSelection(),
  Mutation(),
  NElitism(),
  StopByGenerationIndex(**{StopByGenerationIndex.arg_GENERATIONS: 10}),
  ModelStateSaverLoader(),
  DSSqlite3(),
])
if model.reset():
  print('Successfully initialized model!')
model.run()
