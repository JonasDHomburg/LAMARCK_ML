import warnings
from datetime import datetime
from random import shuffle, choice, random, randint
from typing import List, Type, Set, Dict, Tuple

import networkx as nx
import numpy as np

from LAMARCK_ML.architectures import DataFlow
from LAMARCK_ML.architectures.IOMapping_pb2 import IOMappingProto
from LAMARCK_ML.architectures.NeuralNetwork_pb2 import NeuralNetworkProto
from LAMARCK_ML.architectures.functions import *
from LAMARCK_ML.architectures.interface import ArchitectureInterface
from LAMARCK_ML.architectures.variables import Variable
from LAMARCK_ML.data_util import TypeShape, IOLabel
from LAMARCK_ML.data_util.attribute import attr2pb, pb2attr, value2pb, pb2val
from LAMARCK_ML.metrics.implementations import FlOps, Parameters
from LAMARCK_ML.reproduction.methods import Mutation, Recombination
from LAMARCK_ML.utils import SortingClass, CompareClass


class OverParameterizedNeuralNetwork(ArchitectureInterface, DataFlow,
                                     Mutation.Interface,
                                     Recombination.Interface,
                                     FlOps.Interface,
                                     Parameters.Interface):
  arg_INPUTS = 'inputs'
  arg_OUTPUT_TARGETS = 'output_targets'
  arg_FUNCTIONS = 'functions'
  arg_MIN_DEPTH = 'min_depth'
  arg_MAX_DEPTH = 'max_depth'
  arg_MAX_BRANCH = 'max_branch'
  arg_RANKING_OFFSET = 'ranking_offset'
  arg_PC = 'pc'
  arg_CONSCIOUSNESS = 'consciousness'

  attr_META_EDGES = 'meta_graph'
  attr_META_FUNCTIONS = 'meta_functions'
  attr_INPUT_MAPPING = 'input_mapping'
  attr_META_FUNCTION_CONSCIOUSNESS = 'meta_function_consciousness'

  meta_LABEL = 'label'
  meta_QUALITY = 'quality'

  __depth_default = 0

  def __init__(self, **kwargs):
    super(OverParameterizedNeuralNetwork, self).__init__(**kwargs)
    self.attr = dict()
    self.input_mapping = kwargs.get(self.arg_INPUTS, {})
    self.output_targets = kwargs.get(self.arg_OUTPUT_TARGETS, {})
    # TODO: check parameter types
    self.function_cls = kwargs.get(self.arg_FUNCTIONS)
    self.attr[self.arg_MIN_DEPTH] = kwargs.get(self.arg_MIN_DEPTH, 2)
    self.attr[self.arg_MAX_DEPTH] = kwargs.get(self.arg_MAX_DEPTH, 8)
    self.attr[self.arg_MAX_BRANCH] = kwargs.get(self.arg_MAX_BRANCH, 3)
    self.attr[self.arg_RANKING_OFFSET] = kwargs.get(self.arg_RANKING_OFFSET, 1)
    self.attr[self.arg_PC] = kwargs.get(self.arg_PC, 0.4)
    self.attr[self.arg_CONSCIOUSNESS] = kwargs.get(self.arg_CONSCIOUSNESS, 5)
    try:
      if not (isinstance(self.function_cls, list) and all(
          [isinstance(f, type) and issubclass(f, Function) for f in self.function_cls])):
        raise InvalidFunctionType("Invalid function type in 'functions'!")
    except Exception:
      raise InvalidFunctionType("Invalid function type in 'functions'!")

    self.variable_pool = dict()

    blueprint, self.output_mapping, input_df_obj = OverParameterizedNeuralNetwork.random_networks(
      function_pool=set(self.function_cls),
      input_data_flow=self.input_mapping,
      output_targets=self.output_targets,
      min_depth=self.attr[self.arg_MIN_DEPTH],
      max_depth=self.attr[self.arg_MAX_DEPTH],
      max_recursion_depth=self.attr[self.arg_MAX_BRANCH]
    )
    self._DF_INPUTS = set(self.input_mapping.keys())
    self.build_network(
      data_flow_inputs=input_df_obj,
      blueprint=blueprint
    )
    self._inputs = dict()
    for in_label, (nts_id_name, nts, obj) in self.input_mapping.items():
      self._inputs[in_label] = (nts_id_name, obj)
    self.setup_meta_graph()

    self._id_name = self.getNewName()

  def __eq__(self, other):
    if (isinstance(other, self.__class__)
        and self.id_name == other.id_name
        and len(self.function_cls) == len(other.function_cls) == len(
          [f for f in self.function_cls if f in other.function_cls])
        and len(self._inputs) == len(other._inputs) == len(
          [self._inputs.get(key) == other._inputs.get(key) for key in self._inputs])
        and len(self.input_mapping) == len(other.input_mapping) == len(
          {k: None for k in self.input_mapping if self.input_mapping.get(k) == other.input_mapping.get(k)})
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
          {k: None for k in self.attr if self.attr.get(k) == other.attr.get(k)})
        and len(self.meta_functions) == len(other.meta_functions) == len(
          {k: None for k in self.meta_functions
           if self.meta_functions.get(k) == other.meta_functions.get(k)})
        and len(self.meta_edges) == len(other.meta_edges) == len(
          {k: None for k in self.meta_edges if self.meta_edges.get(k) == self.meta_edges.get(k)})
    ):
      return True
    return False

  def __getstate__(self):
    self.get_pb().SerializeToString()

  def __setstate__(self, state):
    def build_function(pb):
      result = Function.__new__(Function)
      result.__setstate__(pb)
      return result

    if isinstance(state, str) or isinstance(state, bytes):
      _nn = NeuralNetworkProto()
      _nn.ParseFromString(state)
    elif isinstance(state, NeuralNetworkProto):
      _nn = state
    else:
      return
    self._id_name = _nn.id_name
    self.function_cls = [Function.getClassByName(f_cls) for f_cls in _nn.function_cls]
    self.output_targets = dict([pb2val(v) for v in _nn.output_ntss.v])
    self._inputs = dict([(ioMP.in_label, (ioMP.out_label, ioMP.df_id_name)) for ioMP in _nn.input_mapping])
    self._DF_INPUTS = set(self._inputs.keys())
    self.output_mapping = dict([(ioMP.in_label, (ioMP.out_label, ioMP.df_id_name)) for ioMP in _nn.output_mapping])
    self.functions = [build_function(_f) for _f in _nn.functions]
    self.variable_pool = dict()
    for _v in _nn.variables:
      v_ = Variable.__new__(Variable)
      v_.__setstate__(_v)
      self.variable_pool[v_.name] = self.variable_pool.get(v_.name, []) + [v_]
    self.attr = dict([pb2attr(attr) for attr in _nn.attr])
    self.meta_edges = self.attr.pop(self.attr_META_EDGES)
    meta_functions = self.attr.pop(self.attr_META_FUNCTIONS)
    self.meta_functions = dict()
    for _f in meta_functions:
      f = object.__new__(Function)
      f.__setstate__(_f)
      self.meta_functions[f.id_name] = f
    self.meta_function_consciousness = self.attr.pop(self.attr_META_FUNCTION_CONSCIOUSNESS)
    self.input_mapping = self.attr.pop(self.attr_INPUT_MAPPING)

  @classmethod
  def getNewName(cls):
    name = cls.__name__ + '_' + str(datetime.now().timestamp()) + '_%09i' % randint(0, 1e9 - 1)
    return name

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
       OverParameterizedNeuralNetwork.reachable(nts, output_shape, max_depth, function_pool)
       ])
    if not len(input_ntss) > 0:
      return
    elif max_depth == 0:
      for label, nts in input_ntss.items():
        if OverParameterizedNeuralNetwork.reachable(nts, output_shape, 0, function_pool):
          yield input_node, {output_label: label}, []

    stack = [
      (0, input_node, OverParameterizedNeuralNetwork.children_iter(
        input_ntss=input_ntss, target_output=output_shape,
        is_reachable=lambda x, y:
        OverParameterizedNeuralNetwork.reachable(x, y, max_depth - 1, function_pool),
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
            out_node, nts_id, nodes = next(
              OverParameterizedNeuralNetwork.simple_path(input_node=node,
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
          stack.append((depth + 1, n_name, OverParameterizedNeuralNetwork.children_iter(
            input_ntss=out_ntss,
            target_output=output_shape,
            function_pool=function_pool,
            recursion_depth=recursion_depth,
            max_recursion_depth=max_recursion_depth,
            is_reachable=lambda x, y:
            OverParameterizedNeuralNetwork.reachable(x, y, pot, function_pool),
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
        out_node, nts_id, _ = next(
          OverParameterizedNeuralNetwork.simple_path(input_node=_input,
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

  def build_network(self, data_flow_inputs: List[str], blueprint: nx.DiGraph):
    stack = list(blueprint.nodes)
    build_nodes = dict()
    self.functions = list()
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
                                                                 variable_pool=self.variable_pool)
          chosen_parameters = np.random.choice(parameters, size=1, replace=False, p=possibilities)[0]
          build_f = f_class(**chosen_parameters)
          build_nodes[node] = (build_f.outputs, build_f.id_name)
          self.functions.append(build_f)

          if node in self.output_mapping:
            out_nts_id, target_nts_id = self.output_mapping.pop(node)
            self.output_mapping[target_nts_id] = (out_nts_id, build_f.id_name)
        else:
          stack.append(node)

  def setup_meta_graph(self):
    def f2depth(function, id2function, mem=dict()):
      if function.id_name not in mem:
        mem[function.id_name] = max([f2depth(id2function[id_name], id2function, mem) if id_name in id2function
                                     else OverParameterizedNeuralNetwork.__depth_default
                                     for _, id_name in function.input_mapping.values()]) + 1
      return mem[function.id_name]

    id2function = dict()
    for f in self.functions:
      id2function[f.id_name] = f
    network_inputs = {id_name: (out_label, ts) for out_label, ts, id_name in self.input_mapping.values()}

    consciousness = self.attr[self.arg_CONSCIOUSNESS]

    self.meta_functions = dict()
    self.meta_function_consciousness = dict()
    self.meta_edges = dict()

    for f in self.functions:
      depth = f2depth(f, id2function)
      Node = (depth, f.id_name)
      for in_label, (out_label, id_name) in f.input_mapping.items():
        if id_name in id2function:
          in_f = id2function[id_name]
          in_depth = f2depth(in_f, id2function)
          prevNode = (in_depth, in_f.outputs[out_label])
          self.meta_edges[prevNode, in_label, Node] = None
        else:
          out_label, ts = network_inputs[id_name]
          dataNode = (OverParameterizedNeuralNetwork.__depth_default, id_name)
          prevNode = (OverParameterizedNeuralNetwork.__depth_default, ts)
          self.meta_edges[dataNode, out_label, prevNode] = None
          self.meta_edges[prevNode, in_label, Node] = None
      for out_label, ts in f.outputs.items():
        newNode = (depth, ts)
        self.meta_edges[Node, out_label, newNode] = None

      meta_function = f.__copy__()
      meta_function.input_mapping = {}
      self.meta_functions[f.id_name] = meta_function
      self.meta_function_consciousness[f.id_name] = consciousness

  def update_meta_graph_edges(self, quality=None):
    mem = dict()

    def f2depth(function, id2function):
      if function.id_name not in mem:
        stack = {id_name for _, id_name in function.input_mapping.values()
                 if id_name in id2function and id_name not in mem}
        while stack:
          _f = id2function[stack.pop()]
          missing_in_mem = [id_name for _, id_name in _f.input_mapping.values()
                            if id_name in id2function and id_name not in mem]
          if missing_in_mem:
            stack.add(_f.id_name)
            stack.update(missing_in_mem)
            continue
          mem[_f.id_name] = max([mem[id_name] if id_name in id2function
                                 else OverParameterizedNeuralNetwork.__depth_default
                                 for _, id_name in _f.input_mapping.values()]) + 1

        mem[function.id_name] = max([mem[id_name] if id_name in id2function
                                     else OverParameterizedNeuralNetwork.__depth_default
                                     for _, id_name in function.input_mapping.values()]) + 1

      return mem[function.id_name]

    id2function = dict()
    for f in self.functions:
      id2function[f.id_name] = f
    network_inputs = {id_name: (out_label, ts) for out_label, ts, id_name in self.input_mapping.values()}

    consciousness = self.attr[self.arg_CONSCIOUSNESS] + 1

    for f in self.functions:
      depth = f2depth(f, id2function)
      Node = (depth, f.id_name)
      for in_label, (out_label, id_name) in f.input_mapping.items():
        if id_name in id2function:
          in_f = id2function[id_name]
          in_depth = f2depth(in_f, id2function)
          prevNode = (in_depth, in_f.outputs[out_label])
          if self.cmp.greaterThan(quality, self.meta_edges[prevNode, in_label, Node]):
            self.meta_edges[prevNode, in_label, Node] = quality
        else:
          out_label, ts = network_inputs[id_name]
          dataNode = (OverParameterizedNeuralNetwork.__depth_default, id_name)
          prevNode = (OverParameterizedNeuralNetwork.__depth_default, ts)
          if self.cmp.greaterThan(quality, self.meta_edges[dataNode, out_label, prevNode]):
            self.meta_edges[dataNode, out_label, prevNode] = quality
          if self.cmp.greaterThan(quality, self.meta_edges[prevNode, in_label, Node]):
            self.meta_edges[prevNode, in_label, Node] = quality
      for out_label, ts in f.outputs.items():
        newNode = (depth, ts)
        if self.cmp.greaterThan(quality, self.meta_edges[Node, out_label, newNode]):
          self.meta_edges[Node, out_label, newNode] = quality

      self.meta_function_consciousness[f.id_name] = consciousness

    f_to_delete = set()
    for f, c in self.meta_function_consciousness.items():
      if c - 1 > 0:
        self.meta_function_consciousness[f] = c - 1
      else:
        f_to_delete.add(f)

    connectivity = dict()
    for k in list(self.meta_edges.keys()):
      (_, f_id0), _, (_, f_id1) = k
      if (f_id1 in f_to_delete) or (f_id0 in f_to_delete):
        del self.meta_edges[k]
      else:
        n0, l, n1 = k
        connectivity[n1] = connectivity.get(n1, 0) + 1

    for k in list(self.meta_edges.keys()):
      (d0, f_id0), _, (d1, f_id1) = k
      if d0 != 0 and connectivity.get((d0, f_id0), 0) < 1:
        del self.meta_edges[k]

    for f_id in f_to_delete:
      del self.meta_functions[f_id]
      del self.meta_function_consciousness[f_id]

  @property
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
    result.attr.append(attr2pb(self.attr_META_FUNCTIONS, [_f.__getstate__() for _f in self.meta_functions.values()]))
    result.attr.append(attr2pb(self.attr_META_FUNCTION_CONSCIOUSNESS, self.meta_function_consciousness))
    result.attr.append(attr2pb(self.attr_META_EDGES, self.meta_edges))
    result.attr.append(attr2pb(self.attr_INPUT_MAPPING, self.input_mapping))
    return result

  def mutate(self, prob):
    p_c = self.attr[self.arg_PC]
    if not hasattr(self, 'cmp') or not isinstance(self.cmp, CompareClass):
      self.cmp = CompareClass()
    backward_edges = dict()
    nodes = dict()
    depth2ts = dict()
    fID2depth = dict()
    for _from, _label, _to in self.meta_edges:
      backward_edges[_to] = backward_edges.get(_to, []) + [(_from, _label)]

      depth, target_ts = _to
      if isinstance(target_ts, TypeShape):
        ts_set = depth2ts.get(depth, set())
        ts_set.add(target_ts)
        depth2ts[depth] = ts_set
        nodes[target_ts] = nodes.get(target_ts, []) + [_to]
      elif target_ts in self.meta_functions:
        fID2depth[target_ts] = depth

      # depth, target_ts = _from
      # if isinstance(target_ts, TypeShape):
      #   ts_set = depth2ts.get(depth, set())
      #   ts_set.add(target_ts)
      #   depth2ts[depth] = ts_set
      #   nodes[target_ts] = nodes.get(target_ts, []) + [_from]
      # elif target_ts in self.meta_functions:
      #   fID2depth[target_ts] = depth

    outputTS = dict()
    ts2explore = list()
    for out_id, target_ts in self.output_targets.items():
      possible_nodes = nodes[target_ts]
      possible_edges = [_from + (_to,) for _to in possible_nodes for _from in backward_edges[_to]]
      sorted_edges = [sc[0] for sc in
                      sorted([(e, SortingClass(obj=self.meta_edges[e], cmp=self.cmp.greaterThan))
                              for e in possible_edges],
                             key=lambda x: x[1],
                             reverse=True)]
      sel_prob = np.asarray([((1 - p_c) ** i) * p_c for i in range(len(sorted_edges))])
      sel_prob[-1] = sel_prob[-1] / p_c
      sel_prob = sel_prob / sel_prob.sum()
      selected_edge_f2ts = sorted_edges[np.random.choice(len(sorted_edges), 1, replace=False, p=sel_prob)[0]]
      ts2explore.append(selected_edge_f2ts[2])
      outputTS[out_id] = selected_edge_f2ts[2]

    used_meta_edges = dict()
    new_meta_functions = dict()
    used_data_nodes = set()
    used_nodes = set()
    while ts2explore:
      ts_node = ts2explore.pop(0)
      if ts_node[0] == 0:
        used_data_nodes.add(ts_node)
        for in_put in backward_edges[ts_node]:
          used_meta_edges[in_put + (ts_node,)] = self.meta_edges[in_put + (ts_node,)]
        continue
      if ts_node in used_nodes:
        continue
      used_nodes.add(ts_node)
      possible_edges = backward_edges[ts_node]
      sorted_edges = [sc[0] for sc in
                      sorted([(e + (ts_node,), SortingClass(obj=self.meta_edges[e + (ts_node,)],
                                                            cmp=self.cmp.greaterThan))
                              for e in possible_edges],
                             key=lambda x: x[1],
                             reverse=True)]
      sel_prob = np.asarray([((1 - p_c) ** i) * p_c for i in range(len(sorted_edges) + 1)])
      sel_prob[-2] = sel_prob[-2] / p_c
      sel_prob[-1] = prob
      sel_prob = sel_prob / sel_prob.sum()
      selected_edge_f2ts = np.random.choice(sorted_edges + [None], 1, replace=False, p=sel_prob)[0]
      replaced = False
      if selected_edge_f2ts is None:
        replaced = True
        sel_prob = sel_prob[:-1]
        sel_prob = sel_prob / sel_prob.sum()
        selected_edge_f2ts = sorted_edges[np.random.choice(len(sorted_edges), 1, replace=False, p=sel_prob)[0]]
        func_node, edge_label, to_ts = selected_edge_f2ts
        sorted_edges_ts2f = [sc[0] for sc in sorted([(e, SortingClass(
          obj=self.meta_edges[e + (func_node,)],
          cmp=self.cmp.greaterThan)) for e in backward_edges[func_node]],
                                                    key=lambda x: x[1],
                                                    reverse=True)]
        sel_prob = np.asarray([((1 - p_c) ** i) * p_c for i in range(len(sorted_edges_ts2f))])
        sel_prob[-1] = sel_prob[-1] / p_c
        sel_prob = sel_prob / sel_prob.sum()
        from_ts, from_label = sorted_edges_ts2f[np.random.choice(len(sorted_edges_ts2f), 1, replace=False,
                                                                 p=sel_prob)[0]]
        inputs = {from_label: from_ts[1]}
        f_to_replace = OverParameterizedNeuralNetwork.children_iter(input_ntss=inputs,
                                                                    target_output=to_ts[1],
                                                                    is_reachable=lambda x, y:
                                                                    OverParameterizedNeuralNetwork.reachable(
                                                                      x, y, 0, self.function_cls),
                                                                    function_pool=self.function_cls,
                                                                    recursion_depth=1)
        allFound = False
        depth = from_ts[0]
        while not allFound:
          tmp = next(f_to_replace, None)
          if tmp is None:
            replaced = False
            break
          mapping, _f, remaining_inputs, in_out_mapping = tmp
          additional_edges = [(label, (depth, inputs[_key])) for label, _key in in_out_mapping.items()]
          allFound = True
          for label, target_ts in remaining_inputs.items():
            p1_found = False
            for d in range(func_node[0] - 1, -1, -1):
              if target_ts in depth2ts[d]:
                additional_edges.append((label, (d, target_ts)))
                p1_found = True
                break
            if not p1_found:
              allFound = False
              break
        if replaced:
          params, poss = _f.generateParameters(
            input_dict={label: ('none', {'none': target_ts}, 'none') for label, (_, target_ts) in additional_edges},
            expected_outputs=mapping,
            variable_pool=self.variable_pool)
          new_meta_f = _f(**np.random.choice(params, size=1, replace=False, p=poss)[0])
          new_meta_f.input_mapping = {}
          depth = to_ts[0]
          new_meta_f_node = (depth, new_meta_f.id_name)

          for label, target_ts in mapping.items():
            used_meta_edges[new_meta_f_node, label, (depth, target_ts)] = None
          for label, node in additional_edges:
            used_meta_edges[node, label, new_meta_f_node] = None
          new_meta_functions[new_meta_f.id_name] = new_meta_f

          ts2explore.extend([from_ts for _, from_ts in additional_edges])
      if not replaced:
        func_node, edge_label, ts_node = selected_edge_f2ts
        required_ts2 = backward_edges[func_node]
        sorted_edges_ts2f = [sc[0] for sc in sorted([((e + (func_node,)), SortingClass(
          obj=self.meta_edges[e + (func_node,)],
          cmp=self.cmp.greaterThan)) for e in required_ts2],
                                                    key=lambda x: x[1],
                                                    reverse=True)]
        sel_prob = np.asarray([((1 - p_c) ** i) * p_c for i in range(len(sorted_edges_ts2f) + 1)])
        sel_prob[-2] = sel_prob[-2] / p_c
        sel_prob[-1] = prob
        sel_prob = sel_prob / sel_prob.sum()
        selected_edge_ts2f = np.random.choice(sorted_edges_ts2f + [None], 1, replace=False, p=sel_prob)[0]
        if selected_edge_ts2f is None:
          replaced = True
          orig_depth, target_ts = ts_node
          rand = random()
          if rand < .25:
            depth = orig_depth - 1
          elif .25 <= rand < .5:
            depth = orig_depth - 2
          else:
            depth = orig_depth - 3
          depth = max(0, depth)
          p1_inputs = {'%i' % i: _ts for i, _ts in enumerate(depth2ts[depth])}
          p1_children = OverParameterizedNeuralNetwork.children_iter(
            input_ntss=p1_inputs,
            target_output=target_ts,
            is_reachable=lambda x, y: OverParameterizedNeuralNetwork.reachable(x, y, 1, self.function_cls),
            function_pool=self.function_cls,
            recursion_depth=1)
          p1_child = next(p1_children, None)
          path_created = False
          while p1_child is not None and not path_created:
            p1_mapping, p1_f, p1_remaining_in, p1_in_out_mapping = p1_child
            p1_edges = [(label, (depth, p1_inputs[_key])) for label, _key in p1_in_out_mapping.items()]
            p1_found = True
            try:
              for _label, _ts in p1_remaining_in.items():
                try:
                  for d in range(min(depth, orig_depth - 1), -1, -1):
                    if _ts in depth2ts[d]:
                      p1_edges.append((_label, (d, _ts)))
                      raise Exception
                except Exception as e:
                  continue
                raise Exception
            except Exception as e:
              p1_found = False
            if not p1_found:
              p1_child = next(p1_children, None)
              continue
            p2_children = OverParameterizedNeuralNetwork.children_iter(
              input_ntss=p1_mapping,
              target_output=target_ts,
              is_reachable=lambda x, y: OverParameterizedNeuralNetwork.reachable(x, y, 0, self.function_cls),
              function_pool=self.function_cls,
              recursion_depth=1)
            p2_child = next(p2_children, None)
            p2_found = False
            while p2_child is not None and not path_created:
              p2_mapping, p2_f, p2_reamining_in, p2_in_out_mapping = p2_child
              p2_edges = [(label, (-depth - 1, p1_mapping[_key])) for label, _key in p2_in_out_mapping.items()]
              p2_ts2explore = set()
              p2_found = True
              try:
                for _label, _ts in p2_reamining_in.items():
                  try:
                    for d in range(min(depth + 1, orig_depth - 1), -1, -1):
                      if _ts in depth2ts[d]:
                        p2_edges.append((_label, (d, _ts)))
                        p2_ts2explore.add((d, _ts))
                        raise Exception
                  except Exception as e:
                    continue
                  raise Exception
              except Exception as e:
                p2_found = False
              if not p2_found:
                p2_child = next(p2_children, None)
                continue
              path_created = True
            if not p2_found:
              p1_child = next(p1_children, None)
          if path_created:
            params, poss = p1_f.generateParameters(
              input_dict={label: ('none', {'none': ts}, 'none') for label, (_, ts) in p1_edges},
              expected_outputs=p1_mapping,
              variable_pool=self.variable_pool,
            )
            p1_build_f = p1_f(**np.random.choice(params, size=1, replace=False, p=poss)[0])
            p1_build_f.input_mapping = {}
            new_meta_functions[p1_build_f.id_name] = p1_build_f
            p1_depth = depth + 1
            p1_f_node = (-p1_depth, p1_build_f.id_name)
            for label, ts in p1_mapping.items():
              _node_ = (-p1_depth, ts)
              if _node_ not in used_data_nodes:
                used_meta_edges[p1_f_node, label, _node_] = None
                used_data_nodes.add(_node_)
            for label, node in p1_edges:
              used_meta_edges[node, label, p1_f_node] = None
              ts2explore.append(node)

            params, poss = p2_f.generateParameters(
              input_dict={label: ('none', {'none': ts}, 'none') for label, (_, ts) in p2_edges},
              expected_outputs=p2_mapping,
              variable_pool=self.variable_pool,
            )
            p2_build_f = p2_f(**np.random.choice(params, size=1, replace=False, p=poss)[0])
            p2_build_f.input_mapping = {}
            new_meta_functions[p2_build_f.id_name] = p2_build_f
            p2_depth = depth + 2
            p2_f_node = (p2_depth, p2_build_f.id_name)
            for label, ts in p2_mapping.items():
              if ts == target_ts:
                used_meta_edges[p2_f_node, label, (orig_depth, ts)] = None
              else:
                used_meta_edges[p2_f_node, label, (p2_depth, ts)] = None
            for label, node in p2_edges:
              used_meta_edges[node, label, p2_f_node] = None
            ts2explore.extend(p2_ts2explore)

          else:
            replaced = False
        if not replaced:
          used_meta_edges[selected_edge_f2ts] = self.meta_edges[selected_edge_f2ts]
          for e in sorted_edges_ts2f:
            used_meta_edges[e] = self.meta_edges[e]
          ts2explore.extend([from_ts for from_ts, _, _ in sorted_edges_ts2f])

    # instantiate functions
    used_forward = dict()
    used_backward = dict()
    for _from, _label, _to in used_meta_edges.keys():
      used_forward[_from] = used_forward.get(_from, []) + [(_label, _to)]
      used_backward[_to] = used_backward.get(_to, []) + [(_from, _label)]
      used_forward[_to] = used_forward.get(_to, [])

    real_functions = dict()
    for _from, _label, _to in used_meta_edges.keys():
      if not isinstance(_to[1], TypeShape):
        continue
      f_id = _from[1]
      if f_id in real_functions:
        continue
      meta_function = self.meta_functions.get(f_id)
      if meta_function is None:
        meta_function = new_meta_functions.get(f_id)
      if meta_function is not None:
        r_func = meta_function.__copy__()
        r_func.input_mapping = {l_in: (l_out, n_f[1])
                                for n_ts, l_in in used_backward[_from] for n_f, l_out in used_backward.get(n_ts)}
        real_functions[f_id] = r_func

    result = object.__new__(OverParameterizedNeuralNetwork)
    # result.__setstate__(self.get_pb())
    result._id_name = self.id_name
    result.function_cls = list(self.function_cls)
    result.output_targets = {k: v for k, v in self.output_targets.items()}
    result._inputs = {k: (v0, v1) for k, (v0, v1) in self._inputs.items()}
    result._DF_INPUTS = set(result._inputs.keys())
    result.output_mapping = {k: (v0, v1) for k, (v0, v1) in self.output_mapping.items()}
    result.variable_pool = dict()
    result.attr = dict(self.attr)
    result.meta_edges = {k: dict(d) if isinstance(d, dict) else None for k, d in self.meta_edges.items()}
    result.meta_functions = {key: f.__copy__() for key, f in self.meta_functions.items()}
    result.meta_function_consciousness = {key: v for key, v in self.meta_function_consciousness.items()}
    result.input_mapping = {k: (v0, v1, v2) for k, (v0, v1, v2) in self.input_mapping.items()}
    result.functions = list(real_functions.values())
    result.cmp = self.cmp

    id2function = {f.id_name: f for f in result.functions}

    mem = dict()

    def f2depth(function):
      if function not in mem:
        stack = {id_name for _, id_name in id2function[function].input_mapping.values()
                 if id_name in id2function and id_name not in mem}
        while stack:
          _f_id = stack.pop()
          if _f_id in mem:
            continue
          _f = id2function[_f_id]
          missing_in_mem = [id_name for _, id_name in _f.input_mapping.values()
                            if id_name in id2function and id_name not in mem]
          if missing_in_mem:
            stack.add(_f.id_name)
            stack.update(missing_in_mem)
            continue
          mem[_f.id_name] = max([mem[id_name] if id_name in id2function
                                 else OverParameterizedNeuralNetwork.__depth_default
                                 for _, id_name in _f.input_mapping.values()]) + 1

        mem[function] = max([mem[id_name] if id_name in id2function
                             else OverParameterizedNeuralNetwork.__depth_default
                             for _, id_name in id2function[function].input_mapping.values()]) + 1
      return mem[function]

    rename = dict()
    for f in result.functions:
      rename[f.id_name] = f.id_name
      if f.id_name in result.meta_functions:
        depth = f2depth(f.id_name)
        f_node = (depth, f.id_name)
        if any([(f_node, label, (depth, ts)) not in result.meta_edges for label, ts in f.outputs.items()]):
          new_id = f.getNewName()
          rename[f.id_name] = new_id
          meta_function = result.meta_functions.get(f.id_name)
          new_meta = meta_function.__copy__()
          new_meta._name = new_id
          new_meta_functions[new_id] = new_meta

    network_inputs = {id_name: (out_label, ts) for out_label, ts, id_name in result.input_mapping.values()}
    for f in result.functions:
      depth = f2depth(f.id_name)
      Node = (depth, rename[f.id_name])
      for out_label, ts in f.outputs.items():
        newNode = (depth, ts)
        e = Node, out_label, newNode
        if e not in result.meta_edges:
          result.meta_edges[e] = None
      new_mapping = dict()
      for in_label, (out_label, id_name) in f.input_mapping.items():
        if id_name in id2function:
          in_f = id2function[id_name]
          in_depth = f2depth(id_name)
          e = (in_depth, in_f.outputs[out_label]), in_label, Node
          if e not in result.meta_edges:
            result.meta_edges[e] = None
        else:
          _, ts = network_inputs[id_name]
          e = (OverParameterizedNeuralNetwork.__depth_default, ts), in_label, Node
          if e not in result.meta_edges:
            result.meta_edges[e] = None
        new_mapping[in_label] = (out_label, rename.get(id_name, id_name))
      f.input_mapping = new_mapping

      f._name = rename[f.id_name]

    consciousness = self.attr[self.arg_CONSCIOUSNESS]
    for key, f in new_meta_functions.items():
      result.meta_functions[f.id_name] = f
      result.meta_function_consciousness[f.id_name] = consciousness

    result.output_mapping = {out_id: (_label, rename[_from[1]])
                             for out_id, outTS in outputTS.items()
                             for _from, _label in used_backward[outTS]}

    return [result]

  def recombine(self, other):
    if not hasattr(self, 'cmp') or not isinstance(self.cmp, CompareClass):
      self.cmp = CompareClass()
    result = object.__new__(OverParameterizedNeuralNetwork)
    result._id_name = self.getNewName()
    result.function_cls = set(self.function_cls)
    result.output_targets = dict([(label, value.__copy__()) for label, value in self.output_targets.items()])
    result._inputs = dict(self._inputs)
    result._DF_INPUTS = set(self._DF_INPUTS)
    result.output_mapping = dict(self.output_mapping)
    result.functions = list()
    for _f in self.functions:
      # new_f = Function.__new__(Function)
      # new_f.__setstate__(_f.get_pb())
      new_f = _f.__copy__()
      result.functions.append(new_f)
    result.variable_pool = dict([(key, list(value_l)) for key, value_l in self.variable_pool.items()])
    result.attr = dict([pb2attr(attr) for attr in [attr2pb(key, value) for key, value in self.attr.items()]])
    result.input_mapping = dict(self.input_mapping)
    result.cmp = self.cmp

    result.meta_edges = dict()
    result.meta_functions = dict()
    for e, d in self.meta_edges.items():
      result.meta_edges[e] = dict(d) if d is not None else None
    for _id, f in self.meta_functions.items():
      newMetaF = f.__copy__()
      result.meta_functions[_id] = newMetaF

    for e, d in other.meta_edges.items():
      if e in result.meta_edges:
        # crucial merge of weights!!
        r_quality = result.meta_edges[e]
        o_quality = d

        takeOther = self.cmp.greaterThan(o_quality, r_quality)
        if takeOther:
          result.meta_edges[e][self.meta_QUALITY] = o_quality
        _, _, _id = e
        if _id in result.meta_functions and takeOther:
          result.meta_functions[_id] = other.meta_functions[_id].__copy__()
      else:
        result.meta_edges[e] = dict(d) if d is not None else None

    result.meta_function_consciousness = {f: c for f, c in self.meta_function_consciousness.items()}
    for f, c in other.meta_function_consciousness.items():
      if c > result.meta_function_consciousness.get(f, 0):
        result.meta_function_consciousness[f] = c
        result.meta_functions[f] = other.meta_functions[f].__copy__()

    # for _id, f in other.meta_functions.items():
    #   if _id not in result.meta_functions:
    #     newMetaF = f.__copy__()
    #     result.meta_functions[_id] = newMetaF
    #
    # result.meta_function_consciousness = {f: c for f, c in self.meta_function_consciousness.items()}
    # for f, c in other.meta_function_consciousness.items():
    #   result.meta_function_consciousness[f] = max(result.meta_function_consciousness.get(f, 0), c)

    return [result]

  def update_state(self, *args, **kwargs):
    if not hasattr(self, 'cmp') or not isinstance(self.cmp, CompareClass):
      self.cmp = CompareClass()
    for f in self.functions:
      value_dict = kwargs.get(f.id_name, dict())
      for variable in self.meta_functions[f.id_name].variables:
        variable.value = value_dict.get(variable.name, None)
        variable.trainable = True
      for variable in f.variables:
        variable.value = value_dict.get(variable.name, None)
        variable.trainable = True
    self.update_meta_graph_edges(quality=kwargs.get(self.meta_QUALITY))

  def flops_per_sample(self):
    result = 0
    for f in self.functions:
      if isinstance(f, FlOps.Interface):
        result += f.flops_per_sample()
      else:
        warnings.warn('Skipping function {} since it does not implement flops interface!'.format(f.id_name), Warning)
    return result

  def parameters(self):
    result = 0
    for f in self.functions:
      if isinstance(f, Parameters.Interface):
        result += f.parameters()
      else:
        warnings.warn('Skipping function {} since it does not implement parameters interface!'.format(f.id_name),
                      Warning)
    return result

  def norm(self, other):
    if isinstance(other, self.__class__):
      other_ids = set([f.id_name for f in other.functions])
      self_ids = set([f.id_name for f in self.functions])
      return len(other_ids.union(self_ids))-len(other_ids.intersection(self_ids))
    return -1
