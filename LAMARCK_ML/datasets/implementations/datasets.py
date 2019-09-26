from typing import Dict

import numpy as np

from LAMARCK_ML.data_util import IOLabel, Shape, TypeShape
from LAMARCK_ML.data_util.attribute import attr2pb
from LAMARCK_ML.datasets.Dataset_pb2 import DatasetProto
from LAMARCK_ML.datasets.interface import DatasetInterface, InvalidBatchSize

IOLabel.DATA = 'DATA'
IOLabel.TARGET = 'TARGET'


class SupervisedData(DatasetInterface):
  arg_BATCH = 'batch'
  arg_TRAINX = 'train_X'
  arg_TRAINY = 'train_Y'
  arg_TESTX = 'test_X'
  arg_TESTY = 'test_Y'
  arg_VALIDX = 'valid_X'
  arg_VALIDY = 'valid_Y'
  arg_SHAPES = 'typeShapes'
  _DF_INPUTS = None

  def __init__(self, **kwargs):
    super(SupervisedData, self).__init__(**kwargs)
    self.batch = kwargs.get(self.arg_BATCH)
    self._namedOutputShapes = kwargs.get(self.arg_SHAPES)
    for _shape in self._namedOutputShapes.values():
      _shape.shape.dim.insert(0, Shape.Dim(Shape.Dim.Names.BATCH, self.batch))
    self.train_X = kwargs.get(self.arg_TRAINX)
    self.train_Y = kwargs.get(self.arg_TRAINY)
    self.test_X = kwargs.get(self.arg_TESTX)
    self.test_Y = kwargs.get(self.arg_TESTY)
    self.valid_X = kwargs.get(self.arg_VALIDX)
    self.valid_Y = kwargs.get(self.arg_VALIDY)
    self.idx = 0
    self.len = 1
    self.data_X = self.test_X
    self.data_Y = self.test_Y
    self.state = ''

  @property
  def outputs(self) -> Dict[str, TypeShape]:
    return self._namedOutputShapes


class UncorrelatedSupervised(SupervisedData):
  def __init__(self, **kwargs):
    super(UncorrelatedSupervised, self).__init__(**kwargs)

  def __call__(self, *args, **kwargs):
    train = kwargs.get('train', kwargs.get('Train', kwargs.get('TRAIN', False)))
    if 'train' in args or \
        'Train' in args or \
        'TRAIN' in args or \
        1 in args or \
        train:
      self.data_X = self.train_X
      self.data_Y = self.train_Y
      self.len = len(self.train_Y)
      self.state = 'train'
    else:
      self.data_X = self.test_X
      self.data_Y = self.test_Y
      self.len = len(self.test_Y)
      self.state = 'test'
    return self

  def __next__(self):
    if self.idx + self.batch <= self.len:
      data_X = self.data_X[self.idx:self.idx + self.batch]
      data_Y = self.data_Y[self.idx:self.idx + self.batch]
      self.idx = (self.idx + self.batch) % self.len
    else:
      data_X = self.data_X[self.idx:]
      data_Y = self.data_Y[self.idx:]
      remaining = self.batch - self.len + self.idx
      random_idx = np.random.permutation(self.len).tolist()
      self.data_X = [self.data_X[i] for i in random_idx]
      self.data_Y = [self.data_Y[i] for i in random_idx]
      if remaining > self.len:
        raise InvalidBatchSize('Batchsize: %i too large for dataset #%i' % (self.batch, self.len))
      data_X += self.data_X[:remaining]
      data_Y += self.data_Y[:remaining]
      self.idx = remaining

    return {IOLabel.DATA: data_X, IOLabel.TARGET: data_Y}

  def get_pb(self, result=None):
    if not isinstance(result, DatasetProto):
      result = DatasetProto()
    result = super(UncorrelatedSupervised, self).get_pb(result)
    if self.train_X is not None: result.attr_val.append(attr2pb('train_X', self.train_X))
    if self.train_Y is not None: result.attr_val.append(attr2pb('train_Y', self.train_Y))
    if self.test_X is not None: result.attr_val.append(attr2pb('test_X', self.test_X))
    if self.test_Y is not None: result.attr_val.append(attr2pb('test_Y', self.test_Y))
    if self.valid_X is not None: result.attr_val.append(attr2pb('valid_X', self.valid_X))
    if self.valid_Y is not None: result.attr_val.append(attr2pb('valid_Y', self.valid_Y))
    result.attr_val.append(attr2pb('len', self.len))
    result.attr_val.append(attr2pb('idx', self.idx))
    result.attr_val.append(attr2pb('state', self.state))
    return result

  def restore_attributes(self, attr: dict):
    self.train_X = [arr for arr in attr.get('train_X')] if 'train_X' in attr else None
    self.train_Y = [arr for arr in attr.get('train_Y')] if 'train_Y' in attr else None
    self.test_X = [arr for arr in attr.get('test_X')] if 'test_X' in attr else None
    self.test_Y = [arr for arr in attr.get('test_Y')] if 'test_Y' in attr else None
    self.valid_X = [arr for arr in attr.get('valid_X')] if 'valid_X' in attr else None
    self.valid_Y = [arr for arr in attr.get('valid_Y')] if 'valid_Y' in attr else None
    self.idx = attr.get('idx')
    self.state = attr.get('state')
    self.len = attr.get('len')
    if self.state != '':
      self(self.state)

# class CorrelatedSupervised(SupervisedData):
#   def __init__(self, **kwargs):
#     super(CorrelatedSupervised, self).__init__(**kwargs)
#     self.raiseError = False
#
#   def __next__(self):
#     if self.raiseError:
#       self.idx = 0
#       raise ResetState()
#     if self.idx + self.batch <= self.len:
#       data_X = self.data_X[self.idx:self.idx + self.batch]
#       data_Y = self.data_Y[self.idx:self.idx + self.batch]
#       self.idx = (self.idx + self.batch) % self.len
#     else:
#       data_X = self.data_X[self.idx:]
#       data_Y = self.data_Y[self.idx:]
#       random_idx = np.random.permutation(self.len).tolist()
#       self.data_X = [self.data_X[i] for i in random_idx]
#       self.data_Y = [self.data_Y[i] for i in random_idx]
#     return (data_X, data_Y)
#
#   pass
