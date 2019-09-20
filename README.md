# LAMARCK_ML

Library of neuro Architecture search Methods for Application dRiven Construction of networKs.

## Motivation
Domain specific machine learning tasks can usually be achieved with transfer learning. However these solutions often 
suffer from an overhead in computation and memory. Therefore, this python library provides a framework for
neuro architecture search using neuroevolution to solve this problem by generating application specific neural networks.

## How to use
Please have a look at the [examples](examples/basics.py).

## Previous work
This project started as a [preliminary investigation](https://github.com/JonasDHomburg/CECNAN). A generational genetic 
algorithm has been parameterized and used to generate convolutional neural networks for the MNIST data set. The network 
architectures have been optimized to increase the accuracy and minimize the amount of trainable parameters. The results
have been [published](https://link.springer.com/chapter/10.1007/978-3-030-20518-8_61) in [Advances in Computational 
Intelligence](https://link.springer.com/book/10.1007/978-3-030-20518-8).

## Roadmap
The goal is to build a modular and extendable framework to explore the possibilities of neuroevolution to generate task 
specific network architectures. Next steps:
- develop new representations for network architectures
- support multi gpu/cpu systems for network training
- non generational model

## Features

- modular and extendable
- multi-objective optimization
- evolutionary/genetic algorithm
  * mutation
  * recombination
- different selection strategies:
  * tournament selection
  * ranking selection
  * proportional selection
  * ...
- different replacement schemes:
  * elitism
  * weak elitism
  * ...
- custom ranking methods for selection and replacement


- random generation of initial network architectures
- logging and visualization of development
