from random import shuffle, choice, random, sample, randint
from typing import List, Type, Set, Dict, Tuple
from datetime import datetime

import networkx as nx
import numpy as np
import math

from LAMARCK_ML.architectures import DataFlow
from LAMARCK_ML.architectures.IOMapping_pb2 import IOMappingProto
from LAMARCK_ML.architectures.NeuralNetwork_pb2 import NeuralNetworkProto
from LAMARCK_ML.architectures.functions import Function, InvalidFunctionType
from LAMARCK_ML.architectures.interface import ArchitectureInterface
from LAMARCK_ML.architectures.variables import Variable
from LAMARCK_ML.data_util import TypeShape, IOLabel
from LAMARCK_ML.data_util.attribute import attr2pb, pb2attr, value2pb, pb2val
from LAMARCK_ML.reproduction.methods import Mutation, Recombination


class NeuralNetwork(ArchitectureInterface, DataFlow, Mutation.Interface, Recombination.Interface):
  arg_INPUTS = 'inputs'
  arg_OUTPUT_TARGETS = 'output_targets'
  arg_FUNCTIONS = 'functions'
  arg_NAMELESS = 'nameless'
  arg_RECOMBINATION_PROBABILITY = 'cross_prob'
  arg_MIN_DEPTH = 'min_depth'
  arg_MAX_DEPTH = 'max_depth'
  arg_MAX_BRANCH = 'max_branch'

  # _nameIdx = 0

  def __init__(self, **kwargs):
    super(NeuralNetwork, self).__init__(**kwargs)
    self.attr = dict()
    self.input_mapping = kwargs.get(self.arg_INPUTS, {})
    self.output_targets = kwargs.get(self.arg_OUTPUT_TARGETS, {})
    # TODO: check parameter types
    self.attr[self.arg_RECOMBINATION_PROBABILITY] = kwargs.get(self.arg_RECOMBINATION_PROBABILITY, .5)
    self.function_cls = kwargs.get(self.arg_FUNCTIONS)
    self.attr[self.arg_MIN_DEPTH] = kwargs.get(self.arg_MIN_DEPTH, 2)
    self.attr[self.arg_MAX_DEPTH] = kwargs.get(self.arg_MAX_DEPTH, 8)
    self.attr[self.arg_MAX_BRANCH] = kwargs.get(self.arg_MAX_BRANCH, 3)
    try:
      if not (isinstance(self.function_cls, list) and all(
          [isinstance(f, type) and issubclass(f, Function) for f in self.function_cls])):
        raise InvalidFunctionType("Invalid function type in 'functions'!")
    except Exception:
      raise InvalidFunctionType("Invalid function type in 'functions'!")

    self.variable_pool = dict()

    blueprint, self.output_mapping, input_df_obj = NeuralNetwork.random_networks(
      function_pool=set(self.function_cls),
      input_data_flow=self.input_mapping,
      output_targets=self.output_targets,
      min_depth=self.attr[self.arg_MIN_DEPTH],
      max_depth=self.attr[self.arg_MAX_DEPTH],
      max_recursion_depth=self.attr[self.arg_MAX_BRANCH]
    )
    self._DF_INPUTS = set(self.input_mapping.keys())
    self.functions, network_outputs = NeuralNetwork.build_network(
      data_flow_inputs=input_df_obj,
      blueprint=blueprint,
      variable_pool=self.variable_pool,
      output_mapping=self.output_mapping
    )
    self._inputs = dict()
    for in_label, (nts_id_name, nts, obj) in self.input_mapping.items():
      self._inputs[in_label] = (nts_id_name, obj)
    self._id_name = self.getNewName() if not kwargs.get(self.arg_NAMELESS, False) else None

  def __eq__(self, other):
    if (isinstance(other, self.__class__)
        and self.id_name == other.id_name
        and len(self.function_cls) == len(other.function_cls) == len(
          [f for f in self.function_cls if f in other.function_cls])
        and len(self._inputs) == len(other._inputs) == len(
          [self._inputs.get(key) == other._inputs.get(key) for key in self._inputs])
        and len(self.output_targets) == len(other.output_targets)
        and all([any([ot == st for ot in other.output_targets]) for st in self.output_targets])
        and len(self.variable_pool) == len(other.variable_pool) == len(
          [v for v in self.variable_pool if set(self.variable_pool.get(v)) == set(other.variable_pool.get(v))])
        and len(self.output_mapping) == len(other.output_mapping) == len(
          [True for om in self.output_mapping if om in other.output_mapping])
        and len(self._DF_INPUTS) == len(other._DF_INPUTS) == len(
          [True for dfi in self._DF_INPUTS if dfi in other._DF_INPUTS])
        and len(self.functions) == len(other.functions) == len(
          [True for f in self.functions if f in other.functions])
        and len(self.attr) == len(other.attr) == len(
          {k: self.attr.get(k) for k in self.attr if self.attr.get(k) == other.attr.get(k)})
    ):
      return True
    return False

  def __getstate__(self):
    self.get_pb().SerializeToString()

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _nn = NeuralNetworkProto()
      _nn.ParseFromString(state)
    elif isinstance(state, NeuralNetworkProto):
      _nn = state
    else:
      return
    self._id_name = _nn.id_name
    self.function_cls = [Function.getClassByName(f_cls) for f_cls in _nn.function_cls]
    # self.output_targets = dict([(elem.name, TypeShape.from_pb(elem.v.nts_val)) for elem in _nn.output_ntss.vs])
    self.output_targets = dict([pb2val(v) for v in _nn.output_ntss.v])
    self._inputs = dict([(ioMP.in_label, (ioMP.out_label, ioMP.df_id_name)) for ioMP in _nn.input_mapping])
    self._DF_INPUTS = set(self._inputs.keys())
    self.output_mapping = dict([(ioMP.in_label, (ioMP.out_label, ioMP.df_id_name)) for ioMP in _nn.output_mapping])
    self.functions = [Function.get_instance(_f) for _f in _nn.functions]
    self.variable_pool = dict()
    for _v in _nn.variables:
      v_ = Variable.__new__(Variable)
      v_.__setstate__(_v)
      self.variable_pool[v_.name] = self.variable_pool.get(v_.name, []) + [v_]
    self.attr = dict([pb2attr(attr) for attr in _nn.attr])

  def distinct_copy(self):
    result = NeuralNetwork.__new__(NeuralNetwork)
    result._id_name = self.id_name
    result.function_cls = list(self.function_cls)
    result.output_targets = dict([(label, value.__copy__()) for label, value in self.output_targets.items()])
    result._inputs = dict(self._inputs)
    result._DF_INPUTS = set(self._DF_INPUTS)
    result.output_mapping = dict(self.output_mapping)
    result.functions = list()
    function_mapping = dict()
    for _f in self.functions:
      new_f = Function.__new__(Function)
      new_f.__setstate__(_f.get_pb())
      new_f._name = new_f.getNewName(new_f)
      result.functions.append(new_f)
      function_mapping[_f.id_name] = new_f.id_name
    for _f in result.functions:
      for key, (v1, old_f_id) in _f.input_mapping.items():
        _f.input_mapping[key] = (v1, function_mapping.get(old_f_id, old_f_id))
    for in_label, (out_label, f_id) in result.output_mapping.items():
      result.output_mapping[in_label] = (out_label, function_mapping.get(f_id, f_id))
    result.variable_pool = dict([(key, list(value_l)) for key, value_l in self.variable_pool.items()])
    result.attr = dict([pb2attr(attr) for attr in [attr2pb(key, value) for key, value in self.attr.items()]])
    return result

  @classmethod
  def getNewName(cls):
    # name = cls.__name__ + ':%09i' % (cls._nameIdx)
    # cls._nameIdx += 1
    name = cls.__name__ + '_' + str(datetime.now().timestamp()) + '_%09i'%randint(0, 1e9 - 1)
    return name

  @staticmethod
  def build_network(data_flow_inputs: List[str], blueprint: nx.DiGraph, variable_pool, output_mapping):
    stack = list(blueprint.nodes)
    build_nodes = dict()
    build_functions = list()
    output_functions = dict()
    while stack:
      node = stack.pop(0)
      node_inputs = dict()
      node_param_dict = blueprint.nodes[node]
      f_class = node_param_dict['DataFlowObj']
      if f_class in data_flow_inputs:
        build_nodes[node] = (node_param_dict['ntss'], f_class)
      else:
        node2key = blueprint.nodes[node]['node2key']
        all_found = True
        for pred in blueprint.predecessors(node):
          if pred in build_nodes:
            for (key, value) in node2key[pred]:
              _ntss, _id_name = build_nodes[pred]
              node_inputs[key] = (value, _ntss, _id_name)
          else:
            all_found = False
            break
        if all_found:
          parameters, possibilities = f_class.generateParameters(input_dict=node_inputs,
                                                                 expected_outputs=node_param_dict['ntss'],
                                                                 variable_pool=variable_pool)
          chosen_parameters = np.random.choice(parameters, size=1, replace=False, p=possibilities)[0]
          build_f = f_class(**chosen_parameters)
          build_nodes[node] = (build_f.outputs, build_f.id_name)
          build_functions.append(build_f)
          if node in output_mapping:
            out_nts_id, target_nts_id = output_mapping.pop(node)
            output_mapping[target_nts_id] = (out_nts_id, build_f.id_name)
            output_functions[target_nts_id] = build_f
        else:
          stack.append(node)

    return build_functions, output_functions

  @staticmethod
  def reachable(
      input_nts: TypeShape,
      target_nts: TypeShape,
      max_depth: int,
      function_pool: set,
  ):
    class FoundException(Exception):
      pass

    if input_nts.__cmp__(target_nts) == -1 and target_nts.__cmp__(input_nts) == -1:
      return True

    low_list, high_list = [input_nts], [input_nts]
    try:
      for _ in range(max_depth):
        n_low_list = list(low_list)
        for low_ in low_list:
          for func in function_pool:
            _min = func.min_transform(low_)
            if _min is None:
              continue
            if _min.__cmp__(target_nts) == -1:
              raise FoundException()
            new_nts = True
            for curr in n_low_list:
              cmp = _min.__cmp__(curr)
              if cmp == -1:
                n_low_list.remove(curr)
                n_low_list.append(_min)
                new_nts = False
                break
              if cmp == 0:
                continue
              break
            if new_nts:
              n_low_list.append(_min)
        low_list = n_low_list
    except Exception as e:
      if not isinstance(e, FoundException):
        return False
      try:
        for i in range(max_depth):
          n_high_list = list(high_list)
          for high_ in high_list:
            for func in function_pool:
              _max = func.max_transform(high_)
              if _max is None:
                continue
              if target_nts.__cmp__(_max) == -1:
                raise FoundException()
              new_nts = True
              for curr in n_high_list:
                cmp = curr.__cmp__(_max)
                if cmp == -1:
                  n_high_list.remove(curr)
                  n_high_list.append(_max)
                  new_nts = False
                  break
                if cmp == 0:
                  continue
                break
              if new_nts:
                n_high_list.append(_max)
          high_list = n_high_list
      except Exception as e:
        if not isinstance(e, FoundException):
          return False
        return True

    # min_reached = False
    # max_reached = False
    # stack = [(input_nts, input_nts)]
    # for _ in range(max_depth):
    #   new_stack = list()
    #   for _min, _max in stack:
    #     for func in function_pool:
    #       min_, max_ = func.min_transform(_min), func.max_transform(_max)
    #       if min_ is None or max_ is None:
    #         continue
    #       if not min_reached and min_.__cmp__(target_nts) == -1:
    #         min_reached = True
    #       if not max_reached and target_nts.__cmp__(max_) == -1:
    #         max_reached = True
    #       if min_reached and max_reached:
    #         return True
    #       new_stack.append((min_, max_))
    #   stack = new_stack
    return False

  @staticmethod
  def children_iter(input_ntss: Dict[str, TypeShape], target_output: TypeShape,
                    is_reachable, function_pool,
                    recursion_depth, max_recursion_depth=1):

    random_f_pool = list(function_pool)
    pool_dict = dict()
    while random_f_pool:
      _f = choice(random_f_pool)
      possibility_iter = pool_dict.get(_f)
      if possibility_iter is None:
        possibility_iter = _f.possible_output_shapes(input_ntss, target_output,
                                                     is_reachable,
                                                     max_possibilities=5)
        pool_dict[_f] = possibility_iter
      dts = next(possibility_iter, None)
      if dts is None:
        random_f_pool.remove(_f)
        continue

      remaining_inputs, out_nts, in_out_mapping = dts
      if recursion_depth > max_recursion_depth and len(remaining_inputs) > 0:
        continue
      yield (out_nts, _f, remaining_inputs, in_out_mapping)

  @staticmethod
  def simple_path(
      input_node: str,
      input_ntss: Dict[str, TypeShape],
      output_shape: TypeShape,
      output_label: str,
      blueprint: nx.DiGraph(),
      min_depth: int,
      max_depth: int,
      function_pool: set,
      max_recursion_depth: int = 3,
      recursion_depth: int = 0,
  ):
    def clean_up(network, node_name):
      dep_stack = network.nodes[node_name].get('dep', list())
      while dep_stack:
        node = dep_stack.pop()
        dep_stack.extend(network.nodes[node].get('dep', []))
        network.remove_node(node)
      network.remove_node(node_name)

    if max_depth < 0:
      return
    input_ntss = dict(
      [(label, nts) for label, nts in input_ntss.items() if
       NeuralNetwork.reachable(nts, output_shape, max_depth, function_pool)
       ])
    if not len(input_ntss) > 0:
      return
    elif max_depth == 0:
      for label, nts in input_ntss.items():
        if NeuralNetwork.reachable(nts, output_shape, 0, function_pool):
          yield input_node, {output_label: label}, []

    stack = [(0, input_node, NeuralNetwork.children_iter(input_ntss=input_ntss, target_output=output_shape,
                                                         is_reachable=lambda x, y:
                                                         NeuralNetwork.reachable(x, y, max_depth - 1, function_pool),
                                                         function_pool=function_pool,
                                                         recursion_depth=recursion_depth,
                                                         max_recursion_depth=max_recursion_depth))]

    created_nodes = []
    while stack:
      depth, last_node, children = stack[-1]
      child = next(children, None)
      if child is None:
        stack.pop()
        if depth > 0:
          clean_up(blueprint, last_node)
          created_nodes.remove(last_node)

      elif depth <= max_depth:
        out_ntss, _f, remaining_inputs, in_out_mapping = child
        dep = list()
        node2key = dict([(last_node, [(key, value)]) for key, value in in_out_mapping.items()])
        out_nodes = list()
        for rem_in_label, rem_in_shape in remaining_inputs.items():
          temp_nodes = list(blueprint.nodes)
          out_node = None
          while temp_nodes:
            node = choice(temp_nodes)
            temp_nodes.remove(node)
            out_node, nts_id, nodes = next(NeuralNetwork.simple_path(input_node=node,
                                                                     input_ntss=blueprint.nodes[node]['ntss'],
                                                                     output_shape=rem_in_shape,
                                                                     output_label=rem_in_label,
                                                                     blueprint=blueprint,
                                                                     min_depth=0,
                                                                     max_depth=depth,
                                                                     recursion_depth=recursion_depth + 1,
                                                                     function_pool=function_pool,
                                                                     max_recursion_depth=max_recursion_depth
                                                                     ), (None, None, None))
            if out_node is not None:
              for key, value in nts_id.items():
                node2key[out_node] = node2key.get(out_node, []) + [(key, value)]
              dep.extend(nodes)
              out_nodes.append(out_node)
              break
          if not temp_nodes and not out_node:
            break
        if sum([len(v) for v in node2key.values()]) <= len(remaining_inputs):
          while dep:
            node = dep.pop()
            dep.extend(blueprint.nodes[node]).get('dep', [])
            blueprint.remove_node(node)
          continue
        n_name = '{' + ', '.join(
          [label + ': ' + nts.dtype.__str__() + ', ' + str(nts.shape) for label, nts in out_ntss.items()]) + '}'
        index = 0
        while True:
          if n_name + str(index) not in blueprint.nodes:
            n_name += str(index)
            break
          index += 1
        created_nodes.append(n_name)
        blueprint.add_node(n_name, ntss=out_ntss, DataFlowObj=_f,
                           dep=dep, node2key=node2key)
        for out_node in out_nodes:
          blueprint.add_edge(out_node, n_name)
        blueprint.add_edge(last_node, n_name)
        for label, nts in out_ntss.items():
          if nts.dtype == output_shape.dtype and \
              nts.shape == output_shape.shape and \
              depth >= min_depth:
            yield n_name, {output_label: label}, created_nodes
        pot = max_depth - depth - 1
        if pot >= 0:
          stack.append((depth + 1, n_name, NeuralNetwork.children_iter(input_ntss=out_ntss,
                                                                       target_output=output_shape,
                                                                       function_pool=function_pool,
                                                                       recursion_depth=recursion_depth,
                                                                       max_recursion_depth=max_recursion_depth,
                                                                       is_reachable=lambda x, y:
                                                                       NeuralNetwork.reachable(x, y, pot,
                                                                                               function_pool),
                                                                       )))
        else:
          clean_up(blueprint, n_name)
          created_nodes.remove(n_name)
      else:
        stack.pop()
        if depth > 0:
          clean_up(blueprint, last_node)
          created_nodes.remove(last_node)

  @staticmethod
  def random_networks(function_pool: Set[Type[Function]] = None,
                      input_data_flow: Dict[str, Tuple[str, TypeShape, str]] = None,
                      output_targets: Dict[str, TypeShape] = None,
                      min_depth: int = 2,
                      max_depth: int = 5,
                      max_recursion_depth: int = 3
                      ):
    blueprint = nx.DiGraph()
    nn_inputs = []
    for inkey, (out_label, out_nts, df_obj_id) in input_data_flow.items():
      if df_obj_id not in nn_inputs:
        nn_inputs.append(df_obj_id)
      n_name = '{' + inkey + ': ' + out_nts.dtype.__str__() + ', ' + str(out_nts.shape) + '}'
      index = 0
      while True:
        if n_name + str(index) not in blueprint.nodes:
          n_name += str(index)
          break
        index += 1
      blueprint.add_node(n_name, ntss={out_label: out_nts}, DataFlowObj=df_obj_id)

    inputs = list(blueprint.nodes)
    shuffled_outputs = list(output_targets.items())
    shuffle(shuffled_outputs)
    output_mapping = dict()
    for _out_label, _output in shuffled_outputs:
      temp_inputs = list(inputs)
      out_node = None
      nts_id = None
      while True and temp_inputs:
        _input = choice(temp_inputs)
        temp_inputs.remove(_input)
        out_node, nts_id, _ = next(NeuralNetwork.simple_path(input_node=_input,
                                                             input_ntss=blueprint.nodes[_input]['ntss'],
                                                             output_shape=_output,
                                                             output_label=_out_label,
                                                             blueprint=blueprint,
                                                             min_depth=min_depth,
                                                             max_depth=max_depth,
                                                             function_pool=function_pool,
                                                             max_recursion_depth=max_recursion_depth
                                                             ), (None, None, None))
        if out_node is not None:
          break
      if out_node is None:
        raise Exception('Failed to generate Network!')
      output_mapping[out_node] = (nts_id[_out_label], _out_label)

    return blueprint, output_mapping, nn_inputs

  def outputs(self) -> Set[TypeShape]:
    return set(self.output_targets)

  @property
  def inputs(self) -> Dict[IOLabel, Tuple[IOLabel, str]]:
    return self._inputs

  @property
  def id_name(self) -> str:
    return self._id_name

  def get_pb(self, result=None):
    if not isinstance(result, NeuralNetworkProto):
      result = NeuralNetworkProto()
    result.id_name = self._id_name
    for f_cls in self.function_cls:
      result.function_cls.append(f_cls.get_cls_name())

    # result.output_ntss.vs.extend([attr2pb(name=label, value=ts) for label, ts in self.output_targets.items()])
    result.output_ntss.v.extend([value2pb(v) for v in self.output_targets.items()])
    for in_label, (out_label, id_name) in self._inputs.items():
      ioM = IOMappingProto()
      ioM.in_label = in_label
      ioM.out_label = out_label
      ioM.df_id_name = id_name
      result.input_mapping.append(ioM)
    for in_label, (out_label, id_name) in self.output_mapping.items():
      ioM = IOMappingProto()
      ioM.in_label = in_label
      ioM.out_label = out_label
      ioM.df_id_name = id_name
      result.output_mapping.append(ioM)
    for _f in self.functions:
      result.functions.append(_f.get_pb())
    for v in self.variable_pool.values():
      result.variables.extend([_v.get_pb() for _v in v])
    result.attr.extend([attr2pb(key, value) for key, value in self.attr.items()])
    return result

  def norm(self, other):
    if isinstance(other, self.__class__):
      return sum([_f not in self.functions for _f in other.functions]) + \
             sum([_f not in other.functions for _f in self.functions])
    return -1

  def mutate(self, prob):
    def function_wise_mutation(NN, prob):
      new_functions = list()
      changed = False
      function_mapping = dict()
      for _f in NN.functions:
        if isinstance(_f, Mutation.Interface) and random() < prob:
          new_function = _f.mutate(prob)
        else:
          new_function = _f.__copy__()
        new_functions.append(new_function)
        if new_function != _f:
          changed = True
        function_mapping[_f.id_name] = new_function.id_name
      for _f in new_functions:
        for key, (v1, old_f_id) in _f.input_mapping.items():
          _f.input_mapping[key] = (v1, function_mapping.get(old_f_id, old_f_id))
      for in_label, (out_label, f_id) in NN.output_mapping.items():
        NN.output_mapping[in_label] = (out_label, function_mapping.get(f_id, f_id))
      NN.functions = new_functions
      return changed

    def structural_mutation(NN, prob):
      changed = False
      for _ in range(1):
        if random() >= prob:
          continue
        id2function = dict()
        predecessor = dict()
        ancestor = dict()
        for _f in NN.functions:
          id2function[_f.id_name] = _f
          for _f_input, (other_output, other_id) in _f.inputs.items():
            predecessor[_f.id_name] = predecessor.get(_f.id_name, set()) | {other_id}
            ancestor[other_id] = ancestor.get(other_id, set()) | {_f.id_name}
        random_function = choice(NN.functions)
        ancestors = dict()
        stack = {random_function.id_name: 1}
        while stack:
          curr, depth = stack.popitem()
          for anc in ancestor.get(curr, set()):
            ancestors[anc] = max(ancestors.get(anc, 0), depth + 1)
            stack[anc] = max(stack.get(anc, 0), depth + 1)
        if len(ancestors) <= 0:
          continue
        changed = True
        target_function, depth = choice(list(ancestors.items()))
        target_function = id2function[target_function]

        # remove target function
        NN.functions.remove(target_function)

        # create a new path
        target_ancestor = set()
        stack = {target_function.id_name}
        while stack:
          ancs = set(ancestor.get(stack.pop(), set()))
          stack.update(ancs)
          target_ancestor.update(ancs)

        blueprint = nx.DiGraph()
        random_node = None
        build_nodes = dict()
        for _f in NN.functions:
          if _f.id_name in target_ancestor:
            continue
          n_name = '{' + ', '.join(
            [label + ': ' + nts.dtype.__str__() + ', ' + str(nts.shape) for label, nts in _f.outputs.items()]) + '}'
          index = 0
          while True:
            if n_name + str(index) not in blueprint.nodes:
              n_name += str(index)
              break
            index += 1
          if _f.id_name == random_function.id_name:
            random_node = n_name
          build_nodes[n_name] = (_f.outputs, _f.id_name)
          blueprint.add_node(n_name, ntss=_f.outputs, DataFlowObj=_f)

        output_labels = dict()
        if target_function.id_name in ancestor:
          for anc in ancestor[target_function.id_name]:
            for in_label, (out_label, f_id) in id2function[anc].inputs.items():
              if not f_id == target_function.id_name:
                continue
              output_labels[out_label] = output_labels.get(out_label, []) + [(in_label, anc)]
        for in_label, (out_label, f_id) in NN.output_mapping.items():
          if not f_id == target_function.id_name:
            continue
          output_labels[out_label] = output_labels.get(out_label, []) + [(in_label, None)]
        new_functions = set()
        for out_label, out_nts in target_function.outputs.items():
          out_node, nts_id, new_nodes = \
            next(NeuralNetwork.simple_path(input_node=random_node,
                                           input_ntss=random_function.outputs,
                                           output_shape=out_nts,
                                           output_label=out_label,
                                           blueprint=blueprint,
                                           min_depth=0,
                                           max_depth=depth + 1,
                                           function_pool=NN.function_cls,
                                           max_recursion_depth=self.attr[self.arg_MAX_BRANCH],
                                           ), (None, None, None))
          while new_nodes:
            node = new_nodes.pop(0)
            node_inputs = dict()
            node_param_dict = blueprint.nodes[node]
            f_class = node_param_dict['DataFlowObj']
            node2key = blueprint.nodes[node]['node2key']
            all_found = True
            for pred in blueprint.predecessors(node):
              if pred in build_nodes:
                for (key, value) in node2key[pred]:
                  _ntss, _id_name = build_nodes[pred]
                  node_inputs[key] = (value, _ntss, _id_name)
              else:
                if pred not in new_nodes:
                  new_nodes.append(pred)
                all_found = False
                break
            if not all_found:
              new_nodes.append(node)
              continue
            parameters, possibilities = f_class.generateParameters(input_dict=node_inputs,
                                                                   expected_outputs=node_param_dict['ntss'],
                                                                   variable_pool=NN.variable_pool)
            chosen_parameters = np.random.choice(parameters, size=1, replace=False, p=possibilities)[0]
            build_f = f_class(**chosen_parameters)
            build_nodes[node] = (build_f.outputs, build_f.id_name)
            NN.functions.append(build_f)

            new_functions.add(build_f.id_name)
            id2function[build_f.id_name] = build_f
            for _f_input, (other_output, other_id) in build_f.inputs.items():
              predecessor[build_f.id_name] = predecessor.get(build_f.id_name, set()) | {other_id}
              ancestor[other_id] = ancestor.get(other_id, set()) | {build_f.id_name}
          for in_label, f_id in output_labels[out_label]:
            _, out_node_id = build_nodes[out_node]
            ancestor[out_node_id] = ancestor.get(out_node_id, set()) | {f_id}
            if f_id is None:
              NN.output_mapping[in_label] = (out_label, build_nodes[out_node][1])
            else:
              id2function[f_id].input_mapping[in_label] = (out_label, build_nodes[out_node][1])

        out_functions = set([f_id for _, f_id in NN.output_mapping.values()])

        # remove old path or at least all not necessary functions
        NN.functions.insert(0, target_function)
        stack = {target_function.id_name}
        while stack:
          curr = stack.pop()
          pred = predecessor.get(curr, set())
          for _p in pred:
            if ancestor[_p]:
              ancestor[_p].remove(curr)
              if not ancestor[_p] and _p not in out_functions:
                stack.add(_p)
          NN.functions.remove(id2function[curr])
          predecessor.pop(curr)
      return changed

    result = self.distinct_copy()
    changed = structural_mutation(result, prob)
    changed = function_wise_mutation(result, prob) or changed

    if changed:
      result._id_name = NeuralNetwork.getNewName()
    return [result]

  def recombine(self, other):
    self_id2depth = dict([(id_name, 0) for _, id_name in self.inputs.values()])
    self_depth2id = {0: [id_name for _, id_name in self.inputs.values()]}
    self_id2function = dict()
    self_inputs = set([id_name for _, id_name in self.inputs.values()])
    self_outputs = dict()
    for out_id, (label, key) in self.output_mapping.items():
      self_outputs[key] = self_outputs.get(key, []) + [(out_id, label)]

    # map function to network depth
    stack = list(self.functions)
    while stack:
      f = stack.pop(0)
      max_depth = 0
      all_found = True
      for _, f_id in f.inputs.values():
        if f_id not in self_id2function and f_id not in self_inputs:
          stack.append(f)
          all_found = False
          break
        max_depth = max(max_depth, self_id2depth[f_id])
      if not all_found:
        continue
      self_id2function[f.id_name] = f
      self_id2depth[f.id_name] = max_depth + 1
      self_depth2id[max_depth + 1] = self_depth2id.get(max_depth + 1, []) + [f.id_name]

    other_id2depth = dict([(id_name, 0) for _, id_name in other.inputs.values()])
    other_depth2id = {0: [id_name for _, id_name in other.inputs.values()]}
    other_id2function = dict()
    other_inputs = set([id_name for _, id_name in other.inputs.values()])
    other_outputs = dict()
    for out_id, (label, key) in other.output_mapping.items():
      other_outputs[key] = other_outputs.get(key, []) + [(out_id, label)]

    stack = list(other.functions)
    while stack:
      f = stack.pop(0)
      max_depth = 0
      all_found = True
      for _, f_id in f.inputs.values():
        if f_id not in other_id2function and f_id not in other_inputs:
          stack.append(f)
          all_found = False
          break
        max_depth = max(max_depth, other_id2depth[f_id])
      if not all_found:
        continue
      other_id2function[f.id_name] = f
      other_id2depth[f.id_name] = max_depth + 1
      other_depth2id[max_depth + 1] = other_depth2id.get(max_depth + 1, []) + [f.id_name]

    # create new network
    result = NeuralNetwork.__new__(NeuralNetwork)
    result.output_targets = dict([(label, value.__copy__()) for label, value in self.output_targets.items()])
    result.function_cls = list(set(self.function_cls + other.function_cls))
    result.variable_pool = dict()
    for key in set(self.variable_pool.keys()).union(set(other.variable_pool.keys())):
      self_pool = self.variable_pool.get(key, [])
      self_pool = sample(self_pool, k=int(math.ceil(len(self_pool) / 2))) if len(self_pool) > 0 else []
      other_pool = other.variable_pool.get(key, [])
      other_pool = sample(other_pool, k=int(math.ceil(len(other_pool) / 2))) if len(other_pool) > 0 else []
      result.variable_pool[key] = self_pool + other_pool
    result.functions = list()
    result.output_mapping = dict()
    new_functions = dict()

    _current = [max(self_depth2id.keys()),
                self_depth2id,
                self_id2depth,
                self_id2function,
                self_inputs,
                self_outputs,
                self.attr.get(self.arg_RECOMBINATION_PROBABILITY, .5)]
    _other = [max(other_depth2id.keys()),
              other_depth2id,
              other_id2depth,
              other_id2function,
              other_inputs,
              other_outputs,
              other.attr.get(self.arg_RECOMBINATION_PROBABILITY, .5)]

    # switch functions
    depth, depth2id, id2depth, id2function, _inputs, _outputs, cross_prob = _current
    required_inputs = dict()
    for f_ in depth2id[depth]:
      for out_label, label in _outputs.get(f_, []):
        nts_dict = required_inputs.get(f_, dict())
        out_nts = id2function[f_].outputs.get(label, None)
        nts_dict[out_nts] = nts_dict.get(out_nts, []) + [(None, out_label, label)]
        required_inputs[f_] = nts_dict
        _outputs.pop(f_)
    runs = 0
    while depth >= 0:
      runs += 1
      all_reached = False
      connect = list()
      o_depth, o_depth2id, o_id2depth, o_id2function, _, _, _ = _other
      other_new_depth = o_depth
      if random() < cross_prob:
        required_input_keys = [(_f, nts) for _f in required_inputs.keys() for nts in required_inputs[_f].keys()]
        shuffle(required_input_keys)
        connect = list()
        try:
          for d in range(min(o_depth, depth-1), 0, -1):
            other_new_depth = d
            shuffle(o_depth2id[d])
            for out_nts_label, out_nts, _id in [(nts_label, nts, f_id) for f_id in o_depth2id[d] for nts_label, nts in
                                                o_id2function[f_id].outputs.items()]:
              for _f, target_nts in required_input_keys:
                connect_param = next(
                  NeuralNetwork.children_iter(input_ntss={out_nts_label: out_nts},
                                              target_output=target_nts,
                                              function_pool=result.function_cls,
                                              recursion_depth=1,
                                              max_recursion_depth=0,
                                              is_reachable=lambda x, y:
                                              NeuralNetwork.reachable(x, y, 0, function_pool=set())), None)
                # TODO: think about using functions with more than one input
                if connect_param is not None:
                  connect.append((_f, target_nts, out_nts, _id, connect_param))
                  required_input_keys.remove((_f, target_nts))
                  if len(required_input_keys) <= 0:
                    all_reached = True
                    raise Exception
        except:
          pass
      if all_reached:
        other_used_functions = set()
        # Create intermediate connection functions
        for _f, target_nts, out_nts, out_id, (out_ntss, f_class, _, in_out_mapping) in connect:
          out_f = o_id2function[out_id]
          node_inputs = dict([(in_label, (out_label, out_f.outputs, out_id))
                              for in_label, out_label in in_out_mapping.items()])
          parameters, probs = f_class.generateParameters(input_dict=node_inputs,
                                                         expected_outputs=out_ntss,
                                                         variable_pool=result.variable_pool)
          build_f = f_class(**np.random.choice(parameters, size=1, replace=False, p=probs)[0])
          result.functions.append(build_f)
          new_functions[build_f.id_name] = build_f
          # connect intermediate functions to existing functions
          for f_id, in_label, out_label in required_inputs[_f][target_nts]:
            if f_id is None:
              result.output_mapping[in_label] = (choice([label for label, ts in build_f.outputs.items()
                                                         if ts.__eq__(target_nts)]),
                                                 build_f.id_name)
            else:
              new_f = new_functions[f_id]
              new_f.input_mapping[in_label] = (choice([out_label
                                                       for out_label, nts in out_ntss.items()
                                                       if nts.__eq__(target_nts)]), build_f.id_name)

          required_inputs[_f].pop(target_nts)
          if not required_inputs[_f]:
            required_inputs.pop(_f)
          new_out_f = out_f.__copy__()
          new_out_f._name = new_out_f.getNewName(new_out_f)
          result.functions.append(new_out_f)
          new_functions[new_out_f.id_name] = new_out_f
          other_used_functions.add(new_out_f.id_name)
          for in_label, out_label in in_out_mapping.items():
            build_f.input_mapping[in_label] = (out_label, new_out_f.id_name)

        # Update missing network outputs
        missing_outputs = set([out_key for t_list in _outputs.values() for out_key, _ in t_list])
        o_outputs = _other[5]
        for other_f in list(o_outputs.keys()):
          n_list = [(out_key, label) for out_key, label in o_outputs[other_f] if out_key in missing_outputs]
          if len(n_list) > 0:
            o_outputs[other_f] = n_list
          else:
            o_outputs.pop(other_f)

        # check network outputs
        for f_ in list(o_outputs.keys()):
          f_depth = o_id2depth[f_]
          if f_depth >= other_new_depth:
            new_out_f = o_id2function[f_].__copy__()
            new_out_f._name = new_out_f.getNewName(new_out_f)
            result.functions.append(new_out_f)
            new_functions[new_out_f.id_name] = new_out_f
            other_used_functions.add(new_out_f.id_name)
            for out_label, label in o_outputs.get(f_, []):
              result.output_mapping[out_label] = (label, new_out_f.id_name)
            o_outputs.pop(f_)
          elif f_depth == other_new_depth - 1:
            for out_label, label in o_outputs.get(f_, []):
              nts_dict = required_inputs.get(f_, dict())
              out_nts = o_id2function[f_].outputs.get(label, None)
              nts_dict[out_nts] = nts_dict.get(out_nts, []) + [(None, out_label, label)]
              required_inputs[f_] = nts_dict
              o_outputs.pop(f_)

        # update required inputs
        while other_used_functions:
          other_f_id = other_used_functions.pop()
          for in_label, (out_label, in_id) in new_functions[other_f_id].inputs.items():
            if in_id in o_id2function:
              if o_id2depth[in_id] >= other_new_depth:
                in_f = o_id2function[in_id]
                new_in_f = in_f.__copy__()
                new_in_f._name = new_in_f.getNewName()
                result.functions.append(new_in_f)
                new_functions[new_in_f.id_name] = new_in_f
                other_used_functions.add(new_in_f.id_name)
                new_functions[other_f_id].inputs[in_label] = (out_label, new_in_f.id_name)
              else:
                nts_dict = required_inputs.get(in_id, dict())
                out_nts = o_id2function[in_id].outputs.get(out_label, None)
                nts_dict[out_nts] = nts_dict.get(out_nts, []) + [(other_f_id, in_label, out_label)]
                required_inputs[in_id] = nts_dict

        _current[0] = depth - 1
        _other[0] = other_new_depth - 1
        _other, _current = _current, _other
      else:
        for _f in depth2id[depth]:
          # only add functions if they are required
          if _f not in required_inputs:
            continue
          f_ = id2function.get(_f, None)
          if f_ is None:
            continue
          new_out_f = f_.__copy__()
          new_out_f._name = new_out_f.getNewName(new_out_f)
          result.functions.append(new_out_f)
          new_functions[new_out_f.id_name] = new_out_f
          for target_nts in required_inputs.get(_f, []):
            for f_id, in_label, out_label in required_inputs[_f][target_nts]:
              if f_id is None:
                result.output_mapping[in_label] = (out_label, new_out_f.id_name)
              else:
                required_by = new_functions[f_id]
                required_by.input_mapping[in_label] = (out_label, new_out_f.id_name)

          required_inputs.pop(_f, None)
          for in_label, (out_label, in_id) in f_.inputs.items():
            if in_id in id2function:
              nts_dict = required_inputs.get(in_id, dict())
              out_nts = id2function[in_id].outputs.get(out_label, None)
              nts_dict[out_nts] = nts_dict.get(out_nts, []) + [(new_out_f.id_name, in_label, out_label)]
              required_inputs[in_id] = nts_dict
        for _f in list(_outputs.keys()):
          if id2depth[_f] == depth - 1:
            nts_dict = required_inputs.get(_f, dict())
            for out_label, label in _outputs[_f]:
              out_nts = id2function[_f].outputs.get(label, None)
              nts_dict[out_nts] = nts_dict.get(out_nts, []) + [(None, out_label, label)]
            required_inputs[_f] = nts_dict
            _outputs.pop(_f)
        _current[0] = depth - 1
      depth, depth2id, id2depth, id2function, _inputs, _outputs, cross_prob = _current

    result._DF_INPUTS = self._DF_INPUTS
    result._inputs = dict()
    result._inputs = dict(self._inputs)
    result._id_name = result.getNewName()
    result.attr = {**self.attr, **other.attr}
    return [result]

  def update_state(self, *args, **kwargs):
    for f in self.functions:
      value_dict = kwargs.get(f.id_name)
      for variable in f.variables:
        variable.value = value_dict[variable.name]
        variable.trainable = False
        self.variable_pool[variable.name] = self.variable_pool.get(variable.name, []) + [variable]
