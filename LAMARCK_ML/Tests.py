import unittest
import os

os.environ['test_fast'] = '1'

from LAMARCK_ML.architectures.functions.functions_test import \
  TestFunction, \
  TestSubFunctions
from LAMARCK_ML.architectures.losses.losses_test import \
  TestLoss
from LAMARCK_ML.architectures.neuralNetwork_test import \
  TestNeuralNetwork
from LAMARCK_ML.architectures.overParameterizedNN_test import \
  TestOverParameterizedNN
from LAMARCK_ML.architectures.variables.initializer.initializer_test import \
  TestInitializer
from LAMARCK_ML.architectures.variables.regularisation.regularisation_test import \
  TestRegularisation
from LAMARCK_ML.architectures.variables.variables_test import \
  TestVariable
from LAMARCK_ML.data_util.dataType_test import \
  TestBaseType, \
  TestAllTypes
from LAMARCK_ML.data_util.shape_test import \
  TestShape
from LAMARCK_ML.datasets.datasets_test import \
  TestUncorrelatedSupervised
from LAMARCK_ML.individuals.individuals_test import \
  TestIndividuals
from LAMARCK_ML.models.models_test import \
  TestGenerationalModel
from LAMARCK_ML.nn_framework.nn_framework_test import \
  TestNVIDIATensorFlowFramework
from LAMARCK_ML.replacement.replacement_test import \
  TestReplacementSchemes
from LAMARCK_ML.reproduction.reproduction_test import \
  TestReproduction
from LAMARCK_ML.selection.selection_test import \
  TestSelectionStrategies
from LAMARCK_ML.utils.dataSaver.dataSaver_test import \
  TestDBSqlite3
from LAMARCK_ML.utils.modelStateSaverLoader_test import \
  TestModelStateSaverLoader

from LAMARCK_ML.utils.compareClass_test import \
  TestCompareClass

if __name__ == '__main__':
  unittest.main()
  for t in [
    TestShape,

    TestBaseType,
    TestAllTypes,

    TestVariable,
    TestInitializer,
    TestRegularisation,

    TestGenerationalModel,
    TestModelStateSaverLoader,

    TestFunction,
    TestSubFunctions,

    TestUncorrelatedSupervised,

    TestNeuralNetwork,
    TestOverParameterizedNN,

    TestLoss,

    TestIndividuals,

    TestSelectionStrategies,
    TestReplacementSchemes,
    TestReproduction,

    TestDBSqlite3,
    TestNVIDIATensorFlowFramework
  ]:
    print(t)
