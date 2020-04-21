import pickle

import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LAMARCK_ML.utils.stopGenerational import StopByNoProgress, StopByGenerationIndex
from LAMARCK_ML.individuals import IndividualInterface, ClassifierIndividualOPACDG
from LAMARCK_ML.data_util import IOLabel, DFloat, DimNames, Shape, TypeShape
from LAMARCK_ML.utils.telegramNotifier_experimental import TelegramNotifier
from LAMARCK_ML.utils.modelStateSaverLoader import ModelStateSaverLoader
from LAMARCK_ML.reproduction import Mutation, Recombination, RandomStep
from LAMARCK_ML.nn_framework.nvidia_tensorflow import NVIDIATensorFlow
from LAMARCK_ML.architectures.functions import BiasLessDense, Merge
from LAMARCK_ML.models.initialization import RandomInitializer
from LAMARCK_ML.selection import ExponentialRankingSelection
from LAMARCK_ML.datasets import UncorrelatedSupervised
from LAMARCK_ML.utils.sortingClass import SortingClass
from LAMARCK_ML.utils.compareClass import CompareClass
from LAMARCK_ML.utils.dataSaver import DSSqlite3
from LAMARCK_ML.models import GenerationalModel
from LAMARCK_ML.metrics import Accuracy, Nodes
from LAMARCK_ML.replacement import NElitism
from LAMARCK_ML.utils.evaluation import LocalParallelEH_threading

Merge._Merge__max_f = 1.5

cmpClass = CompareClass(**{
  CompareClass.arg_PRIMARY_THRESHOLD: .97,
  CompareClass.arg_PRIMARY_OBJECTIVE: Accuracy.ID,
  CompareClass.arg_SECONDARY_OBJECTIVES: {Nodes.ID: 100 / 0.01}  # 100 nodes per 1% acc
})


def ind_cmp(this: IndividualInterface, other: IndividualInterface):
  t_m = this.metrics
  o_m = other.metrics
  return cmpClass.greaterThan(t_m, o_m)


def MNIST(dataDir, log_folder, token, functions):
  classes = 10
  with open(dataDir + 'DataDumps/mnist_ga_train.p', 'rb') as f:
    data = pickle.load(f)
    train_X, data_Y = np.asarray(data['X']), data['Y']
    train_Y = np.zeros((len(data_Y), classes))
    train_Y[np.arange(len(data_Y)), data_Y] = 1
  with open(dataDir + 'DataDumps/mnist_ga_valid.p', 'rb') as f:
    data = pickle.load(f)
    valid_X, data_Y = np.asarray(data['X']), data['Y']
    valid_Y = np.zeros((len(data_Y), classes))
    valid_Y[np.arange(len(data_Y)), data_Y] = 1
  with open(dataDir + 'DataDumps/mnist_ga_test.p', 'rb') as f:
    data = pickle.load(f)
    test_X, data_Y = np.asarray(data['X']), data['Y']
    test_Y = np.zeros((len(data_Y), classes))
    test_Y[np.arange(len(data_Y)), data_Y] = 1

  train_X = train_X.reshape((train_X.shape[0], -1))
  valid_X = valid_X.reshape((valid_X.shape[0], -1))
  test_X = test_X.reshape((test_X.shape[0], -1))

  dataset = UncorrelatedSupervised(**{
    UncorrelatedSupervised.arg_SHAPES: {IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.UNITS, 784),
                                                                              )),
                                        IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, classes)))},
    UncorrelatedSupervised.arg_NAME: 'MNIST',
    UncorrelatedSupervised.arg_TRAINX: train_X,
    UncorrelatedSupervised.arg_TRAINY: train_Y,
    UncorrelatedSupervised.arg_TESTX: test_X,
    UncorrelatedSupervised.arg_TESTY: test_Y,
    UncorrelatedSupervised.arg_VALIDX: valid_X,
    UncorrelatedSupervised.arg_VALIDY: valid_Y,
  })

  session_cfg = tf.ConfigProto()
  session_cfg.gpu_options.allow_growth = True

  model = GenerationalModel()
  model.add([
    RandomInitializer(**{
      RandomInitializer.arg_GEN_SIZE: 36,
      RandomInitializer.arg_CLASS: ClassifierIndividualOPACDG,
      RandomInitializer.arg_PARAM: {
        ClassifierIndividualOPACDG.arg_NN_FUNCTIONS: functions,
        ClassifierIndividualOPACDG.arg_DATA_NTS: {ts_label: (ts, dataset.id_name)
                                                  for ts_label, ts in dataset.outputs.items()},
        ClassifierIndividualOPACDG.arg_MIN_NN_DEPTH: 1,
        ClassifierIndividualOPACDG.arg_MAX_NN_DEPTH: 2,
        ClassifierIndividualOPACDG.arg_MAX_NN_BRANCH: 1,
      }
    }),

    LocalParallelEH_threading(**{LocalParallelEH_threading.arg_NN_FRAMEWORK_CLASS: NVIDIATensorFlow,
                                 LocalParallelEH_threading.arg_NN_FRAMEWORK_KWARGS: {
                                   NVIDIATensorFlow.arg_DATA_SETS: [dataset],
                                   NVIDIATensorFlow.arg_TMP_FILE:
                                     dataDir + '/' + log_folder + '/tmp/state_{}.ckpt',
                                   NVIDIATensorFlow.arg_BATCH_SIZE: 128,
                                   # NVIDIATensorFlow.arg_EPOCHS: 25,
                                   NVIDIATensorFlow.arg_EPOCHS: 2,
                                   NVIDIATensorFlow.arg_CMP: cmpClass,
                                   NVIDIATensorFlow.arg_SESSION_CFG: session_cfg},
                                 # LocalParallelEH_threading.arg_PARALLEL: 8,
                                 LocalParallelEH_threading.arg_PARALLEL: 4,
                                 LocalParallelEH_threading.arg_ADAPT_KWARGS: {NVIDIATensorFlow.arg_TMP_FILE}
                                 }),
    Accuracy(),
    Nodes(),

    ExponentialRankingSelection(**{
      ExponentialRankingSelection.arg_LIMIT: 20,
      ExponentialRankingSelection.arg_CMP: ind_cmp,
    }),

    Recombination(**{Recombination.arg_LIMIT: 36}),
    # Recombination(**{Recombination.arg_LIMIT: 10}),
    RandomStep(**{RandomStep.arg_LIMIT: 36,
    # RandomStep(**{RandomStep.arg_LIMIT: 10,
                  RandomStep.arg_STEP_SIZE: .3,
                  RandomStep.arg_P: .8}),

    NElitism(**{NElitism.arg_N: 2,
                NElitism.arg_CMP: ind_cmp}),

    StopByNoProgress(**{StopByNoProgress.arg_PATIENCE: 15,
                        StopByNoProgress.arg_CMP: ind_cmp}),
    StopByGenerationIndex(**{StopByGenerationIndex.arg_GENERATIONS: 75}),

    ModelStateSaverLoader(**{
      ModelStateSaverLoader.arg_PREPARATION: True,
      ModelStateSaverLoader.arg_REPRODUCTION: False,
      ModelStateSaverLoader.arg_SELECTION: False,
      ModelStateSaverLoader.arg_REPLACEMENT: False,
      ModelStateSaverLoader.arg_EVALUATION: True,
      ModelStateSaverLoader.arg_FILE: dataDir + '/' + log_folder + '/model_state.pb',
    }),
    DSSqlite3(**{DSSqlite3.arg_FILE: dataDir + log_folder + '/history.db3'}),
    # TelegramNotifier(**{
    #   TelegramNotifier.arg_TOKEN: token,
    #   TelegramNotifier.arg_USER_IDS: {111849723},
    #   TelegramNotifier.arg_SELECTION: True,
    #   TelegramNotifier.arg_NEA_DONE: True})
  ])

  if model.reset():
    print('Successfully initialized model!', flush=True)
    model.run()
  else:
    print('Failed to initialize model!', flush=True)


class dummy:
  def __init__(self, name, metrics):
    self.name = name
    self.metrics = metrics


def evaluate(logDir, log_folder, ga_metrics_f, dataDir, test_metrics_f):
  dataSaver = DSSqlite3(**{DSSqlite3.arg_FILE: logDir + '/' + log_folder + '/history.db3'})
  names = dataSaver.get_individual_names()

  dummies = [dummy(name, dataSaver.get_individual_metrics(name)) for name in names]
  with open(logDir + '/' + log_folder + '/' + ga_metrics_f, 'wb') as f:
    pickle.dump([(d.name, d.metrics) for d in dummies], f)

  classes = 10
  with open(dataDir + 'DataDumps/mnist_ga_valid.p', 'rb') as f:
    data = pickle.load(f)
    train_X, data_Y = np.asarray(data['X']), data['Y']
    train_Y = np.zeros((len(data_Y), classes))
    train_Y[np.arange(len(data_Y)), data_Y] = 1
  with open(dataDir + 'DataDumps/mnist_test.p', 'rb') as f:
    data = pickle.load(f)
    test_X, data_Y = np.asarray(data['X']), data['Y']
    test_Y = np.zeros((len(data_Y), classes))
    test_Y[np.arange(len(data_Y)), data_Y] = 1

  train_X = train_X.reshape((train_X.shape[0], -1))
  test_X = test_X.reshape((test_X.shape[0], -1))

  dataset = UncorrelatedSupervised(**{
    UncorrelatedSupervised.arg_SHAPES: {IOLabel.DATA: TypeShape(DFloat, Shape((DimNames.UNITS, 784),
                                                                              )),
                                        IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, classes)))},
    UncorrelatedSupervised.arg_NAME: 'MNIST',
    UncorrelatedSupervised.arg_TRAINX: train_X,
    UncorrelatedSupervised.arg_TRAINY: train_Y,
    UncorrelatedSupervised.arg_TESTX: test_X,
    UncorrelatedSupervised.arg_TESTY: test_Y,
  })
  session_cfg = tf.ConfigProto()
  session_cfg.gpu_options.allow_growth = True
  lpeh = LocalParallelEH_threading(**{LocalParallelEH_threading.arg_NN_FRAMEWORK_CLASS: NVIDIATensorFlow,
                                      LocalParallelEH_threading.arg_NN_FRAMEWORK_KWARGS: {
                                        NVIDIATensorFlow.arg_DATA_SETS: [dataset],
                                        NVIDIATensorFlow.arg_TMP_FILE:
                                          dataDir + '/' + log_folder + '/tmp/state_{}.ckpt',
                                        NVIDIATensorFlow.arg_BATCH_SIZE: 128,
                                        NVIDIATensorFlow.arg_EPOCHS: 10,
                                        NVIDIATensorFlow.arg_CMP: cmpClass,
                                        NVIDIATensorFlow.arg_SESSION_CFG: session_cfg},
                                      LocalParallelEH_threading.arg_PARALLEL: 9,
                                      LocalParallelEH_threading.arg_ADAPT_KWARGS: {NVIDIATensorFlow.arg_TMP_FILE}
                                      })

  test_metrics = list()
  for i in range(len(dummies) // 36):
    gen = [dataSaver.get_individual_by_name(d.name) for d in dummies[i * 36:(i + 1) * 36]]
    lpeh.evaluate(gen, [Accuracy(), Nodes()])
    for ind in gen:
      test_metrics.append((ind.id_name, ind.metrics))

  with open(logDir + '/' + log_folder + '/' + test_metrics_f, 'wb') as f:
    pickle.dump(test_metrics, f)


def get_best(logFolder, metric_file):
  with open(metric_file, 'rb') as f:
    name_metrics = pickle.load(f)

  dataSaver = DSSqlite3(**{DSSqlite3.arg_FILE: logFolder + '/history.db3'})

  dummies = list()
  for name, metrics in name_metrics:
    dummies.append(dummy(name, metrics))

  # best = sorted([SortingClass(obj=d, cmp=ind_cmp) for d in dummies], reverse=True)[0].obj
  # for b in sorted([SortingClass(obj=d, cmp=ind_cmp) for d in dummies], reverse=True):
  #   found_merge = False
  #   for f in dataSaver.get_individual_by_name(b.obj.name).network.functions:
  #     if isinstance(f, Merge):
  #       found_merge = True
  #       break
  #     print(f.id_name)
  #   if found_merge:
  #     best = b.obj
  #     break

  # for best in sorted([d for d in dummies if d.metrics[Nodes.ID]<=210], key=lambda x: x.metrics[Accuracy.ID], reverse=True):
  #   if any([isinstance(f, Merge) for f in dataSaver.get_individual_by_name(best.name).network.functions]):
  #     break

  best = sorted([d for d in dummies if d.metrics[Nodes.ID] <= 65], key=lambda x: x.metrics[Accuracy.ID], reverse=True)[
    0]

  print(best.name, best.metrics)

  individual = dataSaver.get_individual_by_name(best.name)
  return individual


def export_hbp(individual: ClassifierIndividualOPACDG, path_file: str):
  stack = list(individual.network.functions)
  exported = {'MNIST'}
  netw = list()
  data = {'netw': netw}
  while stack:
    f = stack.pop(0)
    if not all([l in exported for _, l in f.inputs.values()]):
      stack.append(f)
      continue
    layer = {
      'class_name': 'Dense' if isinstance(f, BiasLessDense) else 'Concat',
      'name': f.id_name,
      'size': next(iter(f.outputs.values())).shape[Shape.Dim.Names.UNITS],
      'weights': f.variables[0].value.tolist() if isinstance(f, BiasLessDense) else [],
      'inputs': [_id for _, _id in f.inputs.values()],
    }
    exported.add(f.id_name)
    netw.append(layer)
  if path_file.endswith('.json'):
    with open(path_file, 'w') as f:
      json.dump(data, f)
  elif path_file.endswith('.msgpack'):
    import msgpack
    with open(path_file, 'wb') as f:
      msgpack.dump(data, f, use_single_float=True)


def get_pareto(metric_file_ga: str, metric_file_test: str, logFolder):
  dataSaver = DSSqlite3(**{DSSqlite3.arg_FILE: logFolder + '/history.db3'})

  with open(metric_file_ga, 'rb') as f:
    metricsA = pickle.load(f)
  paretoA = dict()
  for n, metrics in metricsA:
    nodes = metrics[Nodes.ID]
    acc = metrics[Accuracy.ID]
    paretoA[nodes] = max(paretoA.get(nodes, 0), acc)
    # if n == 'ClassifierIndividualOPACDG_1582859786.090028_498141415':
    #   print(metrics)
  n0, ind0 = sorted([(n, d) for n, d in metricsA if d[Nodes.ID] <= 65], key=lambda x: x[1][Accuracy.ID], reverse=True)[
    0]
  ind1 = sorted([SortingClass(obj=dummy(n, d), cmp=ind_cmp) for n, d in metricsA], reverse=True)[0].obj
  n1, ind1 = ind1.name, ind1.metrics
  n2, ind2 = sorted([(n, d) for n, d in metricsA], key=lambda x: x[1][Accuracy.ID], reverse=True)[0]

  xs = [ind0[Nodes.ID], ind1[Nodes.ID], ind2[Nodes.ID]]
  ys = [ind0[Accuracy.ID], ind1[Accuracy.ID], ind2[Accuracy.ID]]

  filtered = dict()
  last_added = None
  for key in sorted(paretoA.keys()):
    if paretoA[key] >= paretoA.get(last_added, 0):
      filtered[key] = paretoA[key]
      last_added = key
  paretoA_ = filtered

  with open(metric_file_test, 'rb') as f:
    metricsB = pickle.load(f)
  paretoB = dict()
  for n, metrics in metricsB:
    nodes = metrics[Nodes.ID]
    acc = metrics[Accuracy.ID]
    paretoB[nodes] = max(paretoB.get(nodes, 0), acc)
    if n == n0:
      print('low', metrics, ind0)
    if n == n1:
      print('middle', metrics, ind1)
    if n == n2:
      print('high', metrics, ind2)

  filtered = dict()
  last_added = None
  for key in sorted(paretoB.keys()):
    if paretoB[key] >= paretoB.get(last_added, 0):
      filtered[key] = paretoB[key]
      last_added = key
  paretoB_ = filtered

  fig = plt.figure(figsize=(10, 6))
  ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
  ax2 = fig.add_axes([0.45, 0.145, 0.48, 0.7])

  ax1.plot(list(paretoA.keys()), list(paretoA.values()), 'x', label='Test Acc as seen by GA')
  ax1.plot(list(paretoA_.keys()), list(paretoA_.values()), label='Test Acc as seen by GA')
  ax1.plot(list(paretoB_.keys()), list(paretoB_.values()), label='Test Acc')
  ax1.plot(xs, ys, 'rx')
  ax1.scatter([xs[-1]], [ys[-1]], s=150, facecolors='none', edgecolors='r', zorder=10)
  ax1.set_xlim([0, 2100])
  ax1.set_ylim([0, 1])
  ax1.set_title('Neurons-Accuracy Pareto', fontsize=16)
  # ax1.set(xlabel='Neurons', ylabel='Accuracy')
  ax1.set_xlabel('Neurons', fontsize=14)
  ax1.set_ylabel('Accuracy', fontsize=14)
  ax1.tick_params(labelsize=10)

  ax2.plot(list(paretoA.keys()), list(paretoA.values()), 'x', label='Test Acc as seen by GA')
  ax2.plot(list(paretoA_.keys()), list(paretoA_.values()), label='Test Acc as seen by GA')
  ax2.plot(list(paretoB_.keys()), list(paretoB_.values()), label='Test Acc')
  ax2.plot(xs, ys, 'rx')
  ax2.scatter([xs[:-1]], [ys[:-1]], s=150, facecolors='none', edgecolors='r', zorder=10)
  ax2.legend(loc='lower right')
  ax2.set_xlim([0, 200])
  ax2.set_ylim([0.9, .985])
  ax2.tick_params(labelsize=10)

  ax1.text(400, 0.15, 'Input: 784', fontsize=12,
           bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
           ha='center')
  ind = dataSaver.get_individual_by_name(n2)
  stack = list(ind.network.functions)
  exported = {'MNIST'}
  y = 0.215
  while stack:
    f = stack.pop(0)
    if not all([l in exported for _, l in f.inputs.values()]):
      stack.append(f)
      continue
    exported.add(f.id_name)
    units = next(iter(f.outputs.values())).shape[Shape.Dim.Names.UNITS]
    ax1.text(400, y, str(units), fontsize=12,
             bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
             ha='center')
    # y += 0.045
    y += 0.065
  ax1.text(400, y, 'SoftMax', fontsize=12,
           bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
           ha='center')
  ax1.plot([xs[-1]-10,400],[ys[-1]-.02,y+0.05],'-', color='#ff3000')

  ax2.text(25,0.95,'Input: 784', fontsize=12,
           bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
           ha='center')
  ind = dataSaver.get_individual_by_name(n0)
  stack = list(ind.network.functions)
  exported = {'MNIST'}
  y = 0.957
  while stack:
    f = stack.pop(0)
    if not all([l in exported for _, l in f.inputs.values()]):
      stack.append(f)
      continue
    exported.add(f.id_name)
    units = next(iter(f.outputs.values())).shape[Shape.Dim.Names.UNITS]
    ax2.text(25, y, str(units), fontsize=12,
             bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
             ha='center')
    y += 0.007
  ax2.text(25, y, 'SoftMax', fontsize=12,
           bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
           ha='center')
  ax2.plot([xs[0]-4,35],[ys[0]+0.001,0.973],'-',color='#ff3000')

  ax2.text(150, 0.92, 'Input: 784', fontsize=12,
           bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
           ha='center')
  ind = dataSaver.get_individual_by_name(n1)
  stack = list(ind.network.functions)
  exported = {'MNIST'}
  y = 0.927
  while stack:
    f = stack.pop(0)
    if not all([l in exported for _, l in f.inputs.values()]):
      stack.append(f)
      continue
    exported.add(f.id_name)
    units = next(iter(f.outputs.values())).shape[Shape.Dim.Names.UNITS]
    ax2.text(150, y, str(units), fontsize=12,
             bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
             ha='center')
    y += 0.007
  ax2.text(150, y, 'SoftMax', fontsize=12,
           bbox={'boxstyle': 'round', 'facecolor': 'w', 'alpha': 0.5},
           ha='center')
  ax2.plot([xs[1]+1, 150], [ys[1]-0.002, y+0.005], '-', color='#ff3000')

  # ax1.add_patch(patches.Rectangle((100,0.05),500,0.05,
  #                                 linewidth=1,
  #                                 edgecolor='k',
  #                                 facecolor='w'))

  plt.savefig('NAS_results.svg', format='svg')
  plt.savefig('NAS_results.pdf', format='pdf')
  plt.savefig('NAS_results.eps', format='eps')
  plt.savefig('NAS_results.png', format='png')
  plt.show()


if __name__ == '__main__':
  logDir = '/media/data1/LAMARCK_DATA/'
  # logDir = '/media/compute/homes/jhomburg/NAS_DATA/'
  # MNIST(dataDir=logDir, log_folder='sequential', token='', functions=[BiasLessDense])
  # MNIST(dataDir=logDir, log_folder='shit2', token='', functions=[BiasLessDense])
  # MNIST(dataDir=logDir, log_folder='acyclicDAG', token='', functions=[BiasLessDense, Merge])

  # evaluate(logDir=logDir, log_folder='acyclicDAG',
  # evaluate(logDir=logDir, log_folder='sequential',
  #          ga_metrics_f='ga_metrics.p', dataDir='/media/data/_Studium/LAMARCK_DATA/', test_metrics_f='test_metrics.p')

  # ind = get_best(logFolder=logDir + '/sequential/', metric_file=logDir + '/sequential/ga_metrics.p')
  # export_hbp(ind, path_file=logDir + '/sequential_network_65.json')
  # ind = get_best(logFolder=logDir + '/acyclicDAG/', metric_file=logDir + '/acyclicDAG/ga_metrics.p')
  # export_hbp(ind, path_file=logDir + '/acyclicDAG_network_60.json')

  get_pareto(logDir + '/sequential/ga_metrics.p', logDir + '/sequential/test_metrics.p', logDir + '/sequential/')
  # get_pareto(logDir + '/acyclicDAG/ga_metrics.p', logDir + '/acyclicDAG/test_metrics.p')
