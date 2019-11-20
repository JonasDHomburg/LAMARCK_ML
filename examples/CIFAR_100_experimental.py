import pickle

import numpy as np

from LAMARCK_ML.architectures.functions import *
from LAMARCK_ML.data_util import IOLabel, DFloat, DimNames, Shape, TypeShape
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.individuals import IndividualInterface, ClassifierIndividualOPACDG
from LAMARCK_ML.metrics import Accuracy, FlOps, Parameters
from LAMARCK_ML.models import GenerationalModel
from LAMARCK_ML.models.initialization import RandomInitializer
from LAMARCK_ML.nn_framework.nvidia_tensorflow_1_12_0 import NVIDIATensorFlow
from LAMARCK_ML.replacement import NElitism
from LAMARCK_ML.reproduction import Mutation, Recombination
from LAMARCK_ML.selection import ExponentialRankingSelection
from LAMARCK_ML.utils.modelStateSaverLoader import ModelStateSaverLoader
from LAMARCK_ML.utils import CompareClass
from LAMARCK_ML.utils.dataSaver import DSSqlite3
from LAMARCK_ML.utils.evaluation import LocalEH
from LAMARCK_ML.utils.stopGenerational import StopByNoProgress
from LAMARCK_ML.utils.telegramNotifier_experimental import TelegramNotifier


def cifar_100():
  cmpClass = CompareClass(**{CompareClass.arg_PRIMARY_THRESHOLD: .6,
                             CompareClass.arg_PRIMARY_OBJECTIVE: Accuracy.ID,
                             CompareClass.arg_SECONDARY_OBJECTIVES:
                               [
                                 {Parameters.ID: 5000 / 0.01},  # allow up to 5k more parameters per 1% acc
                                 {FlOps.ID: 100000 / 0.01},  # allow up to 100k more flops per 1% acc
                               ]})

  def ind_cmp(this: IndividualInterface, other: IndividualInterface):
    t_m = this.metrics
    o_m = other.metrics

    return cmpClass.greaterThan(t_m, o_m)

  def run_ga():
    dataDir = '/path/to/experiment/folder/'
    classes = 100
    with open(dataDir + 'DataDumps/cifar-100_ga_train.p', 'rb') as f:
      data = pickle.load(f)
      train_X, data_Y = np.asarray(data['X']), data['Y']
      train_Y = np.zeros((len(data_Y), classes))
      train_Y[np.arange(len(data_Y)), data_Y] = 1
    with open(dataDir + 'DataDumps/cifar-100_ga_valid.p', 'rb') as f:
      data = pickle.load(f)
      valid_X, data_Y = np.asarray(data['X']), data['Y']
      valid_Y = np.zeros((len(data_Y), classes))
      valid_Y[np.arange(len(data_Y)), data_Y] = 1
    with open(dataDir + 'DataDumps/cifar-100_ga_test.p', 'rb') as f:
      data = pickle.load(f)
      test_X, data_Y = np.asarray(data['X']), data['Y']
      test_Y = np.zeros((len(data_Y), classes))
      test_Y[np.arange(len(data_Y)), data_Y] = 1

    dataset = UncorrelatedSupervised(**{
      UncorrelatedSupervised.arg_SHAPES: {IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.HEIGHT, 32),
                                                                                (DimNames.WIDTH, 32),
                                                                                (DimNames.CHANNEL, 3))),
                                          IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, classes)))},
      UncorrelatedSupervised.arg_NAME: 'CIFAR-100',
      UncorrelatedSupervised.arg_TRAINX: train_X,
      UncorrelatedSupervised.arg_TRAINY: train_Y,
      UncorrelatedSupervised.arg_TESTX: test_X,
      UncorrelatedSupervised.arg_TESTY: test_Y,
      UncorrelatedSupervised.arg_VALIDX: valid_X,
      UncorrelatedSupervised.arg_VALIDY: valid_Y,
    })

    model = GenerationalModel()
    model.add([
      RandomInitializer(**{
        RandomInitializer.arg_GEN_SIZE: 10,
        RandomInitializer.arg_CLASS: ClassifierIndividualOPACDG,
        RandomInitializer.arg_PARAM: {
          ClassifierIndividualOPACDG.arg_NN_FUNCTIONS: [QConv2D, QPooling2D, Flatten, Dense, Merge],
          ClassifierIndividualOPACDG.arg_DATA_NTS: dict([(ts_label, (ts, dataset.id_name))
                                                         for ts_label, ts in dataset.outputs.items()]),
          ClassifierIndividualOPACDG.arg_MIN_NN_DEPTH: 2,
          ClassifierIndividualOPACDG.arg_MAX_NN_DEPTH: 10,
          ClassifierIndividualOPACDG.arg_MAX_NN_BRANCH: 1,
        }
      }),
      LocalEH(**{LocalEH.arg_NN_FRAMEWORK: NVIDIATensorFlow(
        **{NVIDIATensorFlow.arg_DATA_SETS: [dataset],
           NVIDIATensorFlow.arg_TMP_FILE:
             dataDir + 'CIFAR-100/LAMARCK_log/state.ckpt',
           NVIDIATensorFlow.arg_BATCH_SIZE: 32,
           NVIDIATensorFlow.arg_EPOCHS: 100,
           NVIDIATensorFlow.arg_CMP: cmpClass})}),

      Accuracy,
      FlOps,
      Parameters,

      ExponentialRankingSelection(**{
        ExponentialRankingSelection.arg_LIMIT: 4,
        ExponentialRankingSelection.arg_CMP: ind_cmp,
      }),

      Recombination(**{Recombination.arg_LIMIT: 10}),
      Mutation(),

      NElitism(**{NElitism.arg_N: 2,
                  NElitism.arg_CMP: ind_cmp}),

      StopByNoProgress(**{StopByNoProgress.arg_PATIENCE: 15,
                          StopByNoProgress.arg_CMP: ind_cmp}),

      ModelStateSaverLoader(**{
        ModelStateSaverLoader.arg_PREPARATION: True,
        ModelStateSaverLoader.arg_REPRODUCTION: False,
        ModelStateSaverLoader.arg_SELECTION: False,
        ModelStateSaverLoader.arg_REPLACEMENT: False,
        ModelStateSaverLoader.arg_EVALUATION: True,
        ModelStateSaverLoader.arg_FILE: dataDir + 'CIFAR-100/LAMARCK_log/model_state.pb',
      }),
      DSSqlite3(**{DSSqlite3.arg_FILE: dataDir + 'CIFAR-100/LAMARCK_log/history.db3'}),
      # TelegramNotifier(**{
      #   TelegramNotifier.arg_TOKEN: 'telegram-token',
      #   TelegramNotifier.arg_USER_IDS: {'chat ids as int'},
      #   TelegramNotifier.arg_SELECTION: True,
      #   TelegramNotifier.arg_NEA_DONE: True})
    ])

    if model.reset():
      print('Successfully initialized model!')
    model.run()

  run_ga()


if __name__ == '__main__':
  cifar_100()
