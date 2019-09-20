from setuptools import setup

setup(name='LAMARCK_ML',
      version='0.1',
      description='Library of neuro Architecture search Methods for Application dRiven Construction of networKs',
      url='https://github.com/JonasDHomburg/LAMARCK_ML',
      author='Jonas Dominik Homburg',
      author_email='JonasDHomburg@gmail.com',
      license='BSD-3-Clause',
      packages=[
        'LAMARCK_ML',
        'LAMARCK_ML.architectures',
        'LAMARCK_ML.architectures.functions',
        'LAMARCK_ML.architectures.losses',
        'LAMARCK_ML.architectures.variables',
        'LAMARCK_ML.data_util',
        'LAMARCK_ML.datasets',
        'LAMARCK_ML.individuals',
        'LAMARCK_ML.metrics',
        'LAMARCK_ML.models',
        'LAMARCK_ML.nn_framework',
        'LAMARCK_ML.replacement',
        'LAMARCK_ML.reproduction',
        'LAMARCK_ML.selection',
        'LAMARCK_ML.utils',
        'LAMARCK_ML.utils.dataSaver',
        'LAMARCK_ML.utils.evaluation',
        'LAMARCK_ML.visualization'
        'LAMARCK_ML.visualization.generational'
      ],
      # packages = find_packages(),
      install_requires=[
        'numpy',
        'dash',
        'dash-daq',
        'networkx',
        'protobuf',
        'umap-learn',
      ],
      extras_require={
        #   'telegram': ['python-telegram-bot'],
        'tensorflow-gpu': ['tensorflow-gpu'],
        'tensorfow': ['tensorflow'],
        'sqlite3': ['sqlite3'],
      },
      zip_safe=False)


