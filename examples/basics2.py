from LAMARCK_ML.models.RandomSearch import RandomSearch
from LAMARCK_ML.metrics import CartesianFitness
from LAMARCK_ML.selection import TournamentSelection, ExponentialRankingSelection, MaxDiversitySelection, LinearRankingSelection
from LAMARCK_ML.reproduction import Mutation, Recombination, RandomStep
from LAMARCK_ML.replacement import NElitism
from LAMARCK_ML.utils.stopGenerational import StopByNoProgress, StopByGenerationIndex
from LAMARCK_ML.utils.dataSaver import DSSqlite3
from LAMARCK_ML.utils.modelStateSaverLoader import ModelStateSaverLoader
from LAMARCK_ML.utils.SlowDown import SlowDown
from LAMARCK_ML.models.initialization import RandomInitializer
from LAMARCK_ML.utils.evaluation import BaseEH
from LAMARCK_ML.individuals import CartesianIndividual
from LAMARCK_ML.utils.benchmark import Benchmark


def run_model():
  model = RandomSearch()
  model.add([
    # Initializing generation
    RandomInitializer(**{
      RandomInitializer.arg_CLASS: CartesianIndividual,
      # RandomInitializer.arg_GEN_SIZE: 36,
      RandomInitializer.arg_GEN_SIZE: 10,
      # RandomInitializer.arg_PARAM: {CartesianIndividual.arg_Dimensions: 2},
    }),
    # Metric
    CartesianFitness(),

    # evaluation
    BaseEH(),
  ])
  #

  #
  #   # Selection strategy
  #   # ExponentialRankingSelection(**{ExponentialRankingSelection.arg_LIMIT: 6,}),
  #   # LinearRankingSelection(**{LinearRankingSelection.arg_LIMIT: 10}),
  #   MaxDiversitySelection(**{
  #     # MaxDiversitySelection.arg_LIMIT: 20,
  #     MaxDiversitySelection.arg_LIMIT: 4,
  #     MaxDiversitySelection.arg_DIVERSITY: .8,
  #   }),
  #
  #   # Reproduction
  #   Recombination(**{
  #     # Recombination.arg_LIMIT: 36,
  #     Recombination.arg_LIMIT: 10,
  #     Recombination.arg_DESCENDANTS: 2,
  #   }),
  #   # Mutation(**{
  #   #   Mutation.arg_P: 4,
  #   #   Mutation.arg_DESCENDANTS: 1,
  #   #   Mutation.arg_LIMIT: 36,
  #   # }),
  #   RandomStep(**{
  #     RandomStep.arg_P: .7,
  #     RandomStep.arg_STEP_SIZE: 3,
  #     RandomStep.arg_DESCENDANTS: 1,
  #     # RandomStep.arg_LIMIT: 36,
  #     RandomStep.arg_LIMIT: 10,
  #   }),
  #
  #   # Replacement method
  #   NElitism(),
  #
  #   # Stopping
  #   # StopByNoProgress(**{StopByNoProgress.arg_PATIENCE: 25,}),
  #   StopByGenerationIndex(**{StopByGenerationIndex.arg_GENERATIONS: 50}),
  #
  #   # Saving states
  #   # ModelStateSaverLoader(**{
  #   #   ModelStateSaverLoader.arg_REPRODUCTION: True,
  #   #   # ModelStateSaverLoader.arg_SELECTION: True,
  #   #   ModelStateSaverLoader.arg_REPLACEMENT: True,
  #   # }),
  #   DSSqlite3(**{DSSqlite3.arg_FILE: '/media/data1/LAMARCK_DATA/shit2/history.db3'}),
  #
  #   # Slowing down the process
  #   SlowDown(**{SlowDown.arg_SLEEP_TIME: 10}),
  #
  #   # evaluation
  #   BaseEH(),
  #
  #   # Benchmark
  #   Benchmark(),
  # ])
  # if model.reset():
  #   model.run()
  #   print([ind.fitness for ind in model.generation])
  # else:
  #   print('Model failed!')
  model._seed_generation()
  print(model.generation)
  print([i.metrics for i in model.generation])
  model._evaluate()
  print([i.metrics for i in model.generation])


if __name__ == '__main__':
  run_model()
