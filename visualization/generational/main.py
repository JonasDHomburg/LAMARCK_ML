import colorsys
import math
import os
import random
import time
from shutil import copyfile

import dash
import dash_core_components as dcc
import dash_html_components as html
import numba
import plotly
import plotly.graph_objs as go
from dash.dependencies import Input, Output, ClientsideFunction, State
from dash.exceptions import PreventUpdate
from sklearn.manifold import TSNE
import umap
import numpy as np

from LAMARCK_ML.architectures.neuralNetwork import NeuralNetwork
from LAMARCK_ML.metrics import LayoutCrossingEdges, LayoutDistanceX, LayoutDistanceY
from LAMARCK_ML.models import GenerationalModel
from LAMARCK_ML.models.initialization import RandomGraphLayoutInitializer
from LAMARCK_ML.replacement import NElitism
from LAMARCK_ML.reproduction import Mutation, Recombination
from LAMARCK_ML.selection import ExponentialRankingSelection
from LAMARCK_ML.utils.dataSaver.dbSqlite3 import DSSqlite3
from LAMARCK_ML.utils.evaluation import GraphLayoutEH
from LAMARCK_ML.utils.stopGenerational import StopByGenerationIndex, StopByNoProgress
from LAMARCK_ML.reproduction import AncestryEntity

random.seed()


def test_save():
  print('Debug save')



class dashVis():
  """
  Visualizes the NEA parameters and its progress if connected to a database.
  For Unix:
  - expose server with 'server = obj.server'
  - start server with 'gunicorn my_python_file:server'
  """

  arg_DB_CONFIGS = 'db_configs'
  arg_AS_SERVER = 'asServer'

  blank_label = '#blank'

  def __init__(self, **kwargs):
    @numba.njit
    def metric(a, b):
      return a[0].norm(b[0])

    db_config = kwargs.get(self.arg_DB_CONFIGS)

    self.projection = []
    projection_texts = dict()
    projection_distances = dict()
    individuals = dict()

    # ancestry_inidividuals = set()
    # ancestry_edges = dict()

    metrics_individuals = set()
    # metrics_values =

    # train_samples = 1
    # # train_X = [np.random.rand(12288).reshape((64, 64, 3)) for _ in range(train_samples)]
    # train_X = [np.random.rand(20) for _ in range(train_samples)]
    #
    # # train_Y = [np.random.rand(1024) for _ in range(train_samples)]
    # train_Y = [np.random.rand(10) for _ in range(train_samples)]
    #
    # _data = TypeShape(DFloat, Shape((DimNames.HEIGHT, 32),
    #                                 (DimNames.WIDTH, 32),
    #                                 (DimNames.CHANNEL, 3)))
    # _data2 = TypeShape(DFloat, Shape((DimNames.HEIGHT, 32),
    #                                 (DimNames.WIDTH, 32),
    #                                 (DimNames.CHANNEL, 3)))
    # # _data = TypeShape(DFloat, Shape((DimNames.UNITS, 20)))
    # batch = 1
    # dataset = UncorrelatedSupervised(train_X=train_X,
    #                                  train_Y=train_Y,
    #                                  batch=batch,
    #                                  typeShapes={IOLabel.DATA: _data,
    #                                              IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 10)))},
    #                                  name='Dataset')
    # IOLabel.DATA2 = 'DATA2'
    # dataset2 = UncorrelatedSupervised(train_X=train_X,
    #                                  train_Y=train_Y,
    #                                  batch=batch,
    #                                  typeShapes={IOLabel.DATA2: _data2,
    #                                              IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 10)))},
    #                                  name='Dataset2')
    # datasets = [dataset]
    # IOLabel.DS1 = 'DS1'
    # IOLabel.DS2 = 'DS2'
    # self.inputs = {IOLabel.DS1: (IOLabel.DATA, _data, dataset.id_name),
    #                IOLabel.DS2: (IOLabel.DATA2, _data, dataset2.id_name)}
    # # self.inputs = {IOLabel.DS1: (IOLabel.DATA, _data, dataset.id_name)}
    #
    # # outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 512))
    # outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 10))
    # outShape1 = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 15))
    # self.outputs = {'out0': TypeShape(DFloat, outShape), 'out1':TypeShape(DFloat, outShape1)}
    # # self.outputs = {'out': TypeShape(DFloat, outShape)}
    # self.functions = [Merge, Conv2D, Flatten, Dense]
    # # self.functions = [Dense, Merge]
    # blueprints = dict()
    # self.projection = None
    # self.index = 0
    # _nn = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: dict(self.inputs),
    #                        NeuralNetwork.arg_OUTPUT_TARGETS: self.outputs,
    #                        NeuralNetwork.arg_FUNCTIONS: self.functions,
    #                        NeuralNetwork.arg_RECOMBINATION_PROBABILITY: 1})
    # blueprints['Network_' + str(self.index)] = _nn
    # self.index += 1
    # for i in range(5):
    #   nn = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: dict(self.inputs),
    #                         NeuralNetwork.arg_OUTPUT_TARGETS: self.outputs,
    #                         NeuralNetwork.arg_FUNCTIONS: self.functions,
    #                        NeuralNetwork.arg_RECOMBINATION_PROBABILITY: 1})
    #   blueprints['Network_' + str(self.index)] = nn
    #   self.index += 1
    #   print(i)
    #
    # for i in range(5):
    #   nn = random.choice(list(blueprints.values())).mutate(1)[0]
    #   blueprints['Network_' + str(self.index)] = nn
    #   self.index += 1
    #   print(i)
    #
    # for i in range(5):
    #   nn = random.choice(list(blueprints.values())).recombine(random.choice(list(blueprints.values())))[0]
    #   blueprints['Network_' + str(self.index)] = nn
    #   self.index += 1
    #   print(i)

    # self.manifold = MDS(n_components=2,
    #                     max_iter=1,
    #                     n_init=1,
    #                     dissimilarity='precomputed'
    #                     )
    self.manifold = umap.UMAP(n_components=2,
                              n_neighbors=2,
                              min_dist=1,
                              random_state=0,
                              # metric='precomputed',
                              metric='euclidean',
                              # n_epochs=11,
                              # learning_rate=.1,
                              )
    # self.manifold = TSNE()

    self.metrics = [('metric_id0', 'metric label 0'),
                    ('metric_id1', 'metric label 1')]

    ss_info_ancestry_layout = {
      'plot_bgcolor': '#333',
      'paper_bgcolor': '#333',
      'xaxis': {
        'showgrid': False,
        'zeroline': False,
        'showticklabels': False
      },
      'yaxis': {
        'showgrid': False,
        'zeroline': False,
        'showticklabels': False,
      },
      'margin': go.layout.Margin(
        l=0, r=0, b=0, t=25, pad=0,
      ),
      'showlegend': True,
      'hovermode': 'closest',
      'legend': {'font': {'color': '#ffffff'},
                 'xanchor': 'center',
                 'yanchor': 'top',
                 'x': 0.5,
                 'y': 0, },
      # 'height': 800,
    }

    self.app = dash.Dash(__name__)
    self.app.layout = html.Div(id='root', children=[
      html.H1(children="LAMARCK_ML"),
      dcc.Tabs(parent_className='custom-tabs', className='contentArea', id='plotArea', children=[
        dcc.Tab(className='custom-tab', selected_className='custom-selected-tab', label='NEA Parameters',
                style={'display': 'none'},
                id='neaParamA',
                children=[html.H1('NEA Parameters'),
                          html.P([
                            'Web-UI update interval [s]: ',
                            dcc.Input(
                              id='update-interval',
                              type='number',
                              value='300'
                            )
                          ]),
                          ]),
        dcc.Tab(className='custom-tab', selected_className='custom-selected-tab', label='Search Space', id='SSViewA',
                children=[html.H1('Search Space View'),
                          dcc.Interval(
                            id='auto-update-projection',
                            interval=1 * 1000,
                            # n_intervals=0
                          ),
                          html.Div(className='graphArea', children=[
                            dcc.Graph(className='graph', id='searchSpaceProjection',
                                      config={'modeBarButtonsToRemove': [
                                        'select2d',
                                        'lasso2d',
                                        'hoverCompareCartesian',
                                        'toggleSpikelines'
                                      ],
                                        'displaylogo': False,
                                        'displayModeBar': True,
                                      }),
                          ]),
                          html.Div(className='infoArea', children=[
                            dcc.Tabs(className='SideInfo', children=[
                              dcc.Tab(className='custom-tab', selected_className='custom-selected-tab',
                                      label='Structure',
                                      children=[dcc.Graph(className='graph', id='network-structure',
                                                          config={'modeBarButtonsToRemove': [
                                                            # 'zoom2d',
                                                            # 'pan2d',
                                                            # 'zoomIn2d',
                                                            # 'zoomOut2d',
                                                            # 'autoScale2d',
                                                            # 'resetScale2d',
                                                            'select2d',
                                                            'lasso2d',
                                                            # 'hoverClosestCartesian',
                                                            'hoverCompareCartesian',
                                                            # 'toImage',
                                                            'toggleSpikelines'
                                                          ],
                                                            'displaylogo': False,
                                                            'displayModeBar': True,
                                                          }),
                                                ]),
                              dcc.Tab(className='custom-tab', selected_className='custom-selected-tab',
                                      label='Ancestry',
                                      children=[dcc.Graph(className='graph', id='searchSpaceAncestry',
                                                          config={'modeBarButtonsToRemove': ['select2d',
                                                                                             'lasso2d',
                                                                                             'hoverCompareCartesian',
                                                                                             'toggleSpikelines'],
                                                                  'displaylogo': False,
                                                                  'displayModeBar': True,
                                                                  }),
                                                ]),
                              dcc.Tab(className='custom-tab', selected_className='custom-selected-tab',
                                      label='Download',
                                      children=[
                                        html.Div(className='infoArea H1', children='Graphic',
                                                 style={'marginTop': 10}),
                                        dcc.RadioItems(id='SSDownloadGraphic',
                                                       options=[
                                                         {'label': 'Projection', 'value': 'Projection'},
                                                         {'label': 'Structure', 'value': 'Structure'},
                                                         {'label': 'Ancestry', 'value': 'Ancestry'}
                                                       ], value='Projection',
                                                       labelStyle={'display': 'block'}),
                                        html.Div(className='infoArea spacer'),
                                        html.Div(className='infoArea H1', children='Format'),
                                        dcc.RadioItems(id='SSDownloadFormat',
                                                       options=[
                                                         {'label': 'SVG', 'value': 'svg'},
                                                         {'label': 'PNG', 'value': 'png'},
                                                         {'label': 'JPEG', 'value': 'jpeg'},
                                                         {'label': 'WebP', 'value': 'webp'},
                                                         {'label': 'PDF', 'value': 'pdf'},
                                                         {'label': 'EPS', 'value': 'eps'},
                                                       ], value='svg',
                                                       labelStyle={'display': 'block'}),
                                        html.Div(className='infoArea spacer'),
                                        html.Div(className='infoArea H1', children='Filename'),
                                        dcc.Input(id='SSDownloadFileName',
                                                  style={'width': '98.5%',
                                                         'height': '12pt',
                                                         },
                                                  value='LAMARCK_plot'),
                                        html.Button('Download',
                                                    id='SearchSpaceDownload',
                                                    className='downloadButton',
                                                    style={'width': '100%',
                                                           'height': '50px',
                                                           'margin-top': 10,
                                                           },
                                                    ),
                                        html.Div(children='test', style={'display': 'none'},
                                                 id='dummy')
                                      ])
                            ]),
                          ]),
                          ]),
        dcc.Tab(className='custom-tab', selected_className='custom-selected-tab', label='Ancestry', id='AncestryViewA',
                children=[html.H1('Ancestry View'),
                          dcc.Interval(
                            id='auto-update-ancestry',
                            interval=1 * 1000,
                            # n_intervals=0
                          ),
                          html.Div(className='graphArea', children=[
                            dcc.Graph(className='graph', id='ancestry-vis',
                                      config={'modeBarButtonsToRemove': [
                                        'select2d',
                                        'lasso2d',
                                        'hoverCompareCartesian',
                                        'toggleSpikelines'
                                      ],
                                        'displaylogo': False,
                                        'displayModeBar': True,
                                      }),
                          ]),
                          html.Div(className='infoArea', children=[
                            html.Div(className='infoArea H1',
                                     children='Reproduction'),
                            dcc.RadioItems(options=[
                              {'label': 'Hover', 'value': 'hover'},
                              {'label': 'All', 'value': 'all'},
                            ], value='hover', id='ancestry-rep-style'),
                            html.Div(className='infoArea spacer'),
                            html.Div(className='infoArea H1', children='Styling'),
                            dcc.Checklist(options=[
                              {'label': 'Repeat Individuals', 'value': 'repInd'}
                            ], id='ancestry-ind-style', value=[]),
                          ]),
                          ]),
        dcc.Tab(className='custom-tab', selected_className='custom-selected-tab', label='Metrics', id='plotsA',
                children=[html.H1('Metrics'),
                          dcc.Interval(
                            id='auto-update-metrics',
                            interval=1 * 1000,
                            # n_intervals=0
                          ),
                          html.Div(className='metricArea', children=[
                            dcc.Tabs(id='metric-tabs',
                                     children=[
                                       dcc.Tab(className='custom-tab', selected_className='custom-selected-tab',
                                               label=metric_label, children=[
                                           dcc.Graph(className='graph', id=metric_id),
                                         ]) for metric_id, metric_label in self.metrics]),
                          ]),
                          ])
      ]),
      html.Div(id="output-clientside"),
    ])

    self.app.clientside_callback(
      ClientsideFunction(namespace="clientside", function_name="resize"),
      Output("output-clientside", "children"),
      [Input('neaParamA', ''),
       Input('SSViewA', ''),
       Input('AncestryViewA', ''),
       Input('plotsA', '')]
    )

    @self.app.callback(
      output=[Output(component_id='dummy', component_property='children')],
      inputs=[Input(component_id='SearchSpaceDownload', component_property='n_clicks')],
      state=[State(component_id='SSDownloadFileName', component_property='value'),
             State(component_id='SSDownloadGraphic', component_property='value'),
             State(component_id='SSDownloadFormat', component_property='value'),
             State(component_id='searchSpaceProjection', component_property='figure'),
             State(component_id='network-structure', component_property='figure'),
             State(component_id='searchSpaceAncestry', component_property='figure')
             ]
    )
    def searchSpaceDownload(n_clicks,
                            _f, graphic_label, format_label,
                            projection_figure, structure_figure, ancestry_figure):
      if graphic_label == 'Projection':
        fig = projection_figure
      elif graphic_label == 'Structure':
        fig = structure_figure
      else:
        fig = ancestry_figure
      if fig is None:
        raise PreventUpdate

      fig = go.Figure(fig)
      plotly.io.write_image(fig, file='/tmp/' + _f, format=format_label)
      time.sleep(1)
      path = os.path.expanduser('~/Downloads')
      file_name = '{}/{}.{}'.format(path, _f, format_label)
      idx = 0
      while os.path.exists(file_name):
        file_name = '{}/{}_{}.{}'.format(path, _f, str(idx), format_label)
        idx += 1
      copyfile('/tmp/' + _f,
               file_name)
      return ['Test']

    def getStructureColor(name, colors=dict(), remaining_val=[10]):
      if name == dashVis.blank_label:
        return '#777'
      color = colors.get(name)
      if color is None:
        r = remaining_val.pop(0)
        color = '#' + ''.join(['%0.2X' % int(v * 255) for v in colorsys.hsv_to_rgb(r / 360, .75, .75)]), r
        colors[name] = color
        if len(remaining_val) == 0:
          add = 360 / (len(colors) * 2)
          remaining_val.extend([v[1] + add for v in colors.values()])
      return color[0]

    def getAncColor(name, colors=dict(), remaining_val=[10]):
      return getStructureColor(name, colors=colors, remaining_val=remaining_val)

    def NetworkLayout2(nn: NeuralNetwork, datasets=[]):
      edges = set()
      for f in nn.functions:
        for _, o_id in f.inputs.values():
          edges.add((o_id, f.id_name))

      model = GenerationalModel()
      model.add([
        RandomGraphLayoutInitializer(**{
          RandomGraphLayoutInitializer.arg_DISTANCE: 1,
          RandomGraphLayoutInitializer.arg_GEN_SIZE: 30,
          RandomGraphLayoutInitializer.arg_EDGES: edges,
          RandomGraphLayoutInitializer.arg_METRIC_WEIGHTS:
            {
              # LayoutCrossingEdges.ID: .25,
              # LayoutDistanceX.ID: .5,
              # LayoutDistanceY.ID: .25,
            }
        }),
        GraphLayoutEH(),
        LayoutCrossingEdges(),
        LayoutDistanceX(),
        LayoutDistanceY(),
        # MaxDiversitySelection(**{MaxDiversitySelection.arg_LIMIT: 20}),
        ExponentialRankingSelection(**{ExponentialRankingSelection.arg_LIMIT: 6}),
        Recombination(),
        Mutation(**{Mutation.arg_P: .25,
                    Mutation.arg_DESCENDANTS: 1}),
        NElitism(**{NElitism.arg_N: 2}),
        StopByGenerationIndex(**{StopByGenerationIndex.arg_GENERATIONS: 750}),
        StopByNoProgress(**{StopByNoProgress.arg_PATIENCE: 100})
      ])
      model.reset()
      model.run()
      ind = max(model.generation)
      # print(ind.metrics, ind.fitness, model.generation_idx)
      # print(min(ind.node2X.values()), max(ind.node2X.values()))
      del model

      n2d = dict([(n, d) for d, nodes in ind.depth2nodes.items() for n in nodes])

      nodes = dict()
      edges = dict()
      for _in in set([id_name for _, id_name in nn.inputs.values()]):
        node_x, node_y, node_text = nodes.get(_in, ([], [], []))
        node_x.append(ind.node2X[_in])
        node_y.append(n2d[_in])
        node_text.append(',<br />'.join(
          ['{' + nts_id_name + ': ' + nts.dtype.__str__() + ', ' + str(nts.shape) + '}'
           for d in datasets if d.id_name == _in for nts_id_name, nts in d.outputs.items()]))
        if _in not in nodes:
          nodes[_in] = (node_x, node_y, node_text)

      pseudo_edges = dict()
      stack = list(ind.edges)
      while stack:
        e = stack.pop(0)
        e0, e1 = e
        if e0 in ind.real_nodes:
          tmp = [e]
          container = [e1]
          while container[0] not in ind.real_nodes:
            for e_ in stack:
              v0, v1 = e_
              if v0 == container[0]:
                tmp.append(e_)
                stack.remove(e_)
                container[0] = v1
                break
          pseudo_edges[(e0, container[0])] = tmp
        else:
          stack.append(e)

      for _f in nn.functions:
        f_name = _f.__class__.__name__
        node_x, node_y, node_text = nodes.get(f_name, ([], [], []))
        node_x.append(ind.node2X[_f.id_name])
        node_y.append(n2d[_f.id_name])
        node_text.append(',<br />'.join(
          ['{' + nts_id_name + ': ' + nts.dtype.__str__() + ', ' + str(nts.shape) + '}' for nts_id_name, nts in
           _f.outputs.items()]))
        if f_name not in nodes:
          nodes[f_name] = (node_x, node_y, node_text)

        for label_to, (label_from, node_from) in _f.inputs.items():
          to_x, to_y = ind.node2X[_f.id_name], n2d[_f.id_name]
          from_x, from_y = ind.node2X[node_from], n2d[node_from]
          intermediat_edges = pseudo_edges[(node_from, _f.id_name)]
          if len(intermediat_edges) <= 1:
            m_x, m_y = (to_x + from_x) / 2, (to_y + from_y) / 2
            edge_x, edge_y = edges.get(label_to, ([], []))
            edge_x.extend([m_x, to_x, None])
            edge_y.extend([m_y, to_y, None])
            if label_to not in edges:
              edges[label_to] = (edge_x, edge_y)

            edge_x, edge_y = edges.get(label_from, ([], []))
            edge_x.extend([from_x, m_x, None])
            edge_y.extend([from_y, m_y, None])
            if label_from not in edges:
              edges[label_from] = (edge_x, edge_y)
          else:
            edge_x, edge_y = edges.get(label_from, ([], []))
            m_id = intermediat_edges[0][1]
            m_x, m_y = ind.node2X[m_id], n2d[m_id]
            edge_x.extend([from_x, m_x, None])
            edge_y.extend([from_y, m_y, None])
            if label_from not in edges:
              edges[label_from] = (edge_x, edge_y)
            print(intermediat_edges)
            for i in range(1, len(intermediat_edges) - 1):
              # print(intermediat_edges[i])
              m_id = intermediat_edges[i][0]
              from_x, from_y = ind.node2X[m_id], n2d[m_id]
              m_id = intermediat_edges[i][1]
              m_x, m_y = ind.node2X[m_id], n2d[m_id]
              edge_x, edge_y = edges.get(dashVis.blank_label, ([], []))
              edge_x.extend([from_x, m_x, None])
              edge_y.extend([from_y, m_y, None])
              if dashVis.blank_label not in edges:
                edges[dashVis.blank_label] = (edge_x, edge_y)
            m_id = intermediat_edges[-1][0]
            m_x, m_y = ind.node2X[m_id], n2d[m_id]
            edge_x, edge_y = edges.get(label_to, ([], []))
            edge_x.extend([m_x, to_x, None])
            edge_y.extend([m_y, to_y, None])
            if label_to not in edges:
              edges[label_to] = (edge_x, edge_y)
      return nodes, edges, \
             (min(ind.depth2nodes.keys()) - 1, max(ind.depth2nodes.keys()) + 1), \
             (min(ind.node2X.values()) - 1, max(ind.node2X.values()) + 1)

    def NetworkLayout(nn: NeuralNetwork, datasets=[]):
      stack = list(nn.functions)
      df_name2obj = dict([(_f.id_name, _f) for _f in stack])
      inputs = set([id_name for _, id_name in nn.inputs.values()])
      pos_y = dict([(id_name, 0) for id_name in inputs])
      y_pos = dict([(0, [id_name]) for id_name in inputs])
      while stack:
        _f = stack.pop()
        y_coord = 0
        all_found = True
        for predecessor in [id_name for _, id_name in _f.inputs.values()]:
          if (predecessor not in pos_y
              and predecessor not in inputs):
            predecessor = df_name2obj.get(predecessor)
            stack.append(_f)
            try:
              stack.remove(predecessor)
              stack.append(predecessor)
            except ValueError:
              pass

            all_found = False
            break
          else:
            y_coord = max(pos_y.get(predecessor) + 1, y_coord)
        if all_found:
          pos_y[_f.id_name] = y_coord
          y_pos[y_coord] = y_pos.get(y_coord, []) + [_f]

      pos_x = dict([(_id, x) for x, _id in enumerate(inputs)])
      y_x = {0: len(inputs)}
      for y in range(1, max(y_pos.keys()) + 1):
        for _f in y_pos.get(y, []):
          predecessors = list([id_name for _, id_name in _f.inputs.values()])
          x_pos = set()
          pred_x = 0
          for pred in predecessors:
            x = pos_x[pred]
            if x in x_pos:
              x += 1
              for n in y_pos[pos_y[pred]]:
                if isinstance(n, str):
                  _x = pos_x[n]
                  pos_x[n] = _x + 1 if _x >= x else _x
                else:
                  _x = pos_x[n.id_name]
                  pos_x[n.id_name] = _x + 1 if _x >= x else _x
            x_pos.add(x)
            pred_x += x

          _y_x = 0 if y_x.get(y) is None else y_x[y] + 1
          y_x[y] = _y_x + pred_x
          pred_x = max(pred_x * 1.0 / (len(predecessors) if len(predecessors) > 0 else 1), _y_x)
          pos_x[_f.id_name] = pred_x

      nodes = dict()
      for _in in inputs:
        node_x, node_y, node_text = nodes.get(_in, ([], [], []))
        node_x.append(pos_x[_in])
        node_y.append(pos_y[_in])
        node_text.append(',<br />'.join(
          ['{' + nts_id_name + ': ' + nts.dtype.__str__() + ', ' + str(nts.shape) + '}'
           for d in datasets if d.id_name == _in for nts_id_name, nts in d.outputs.items()]))
        if _in not in nodes:
          nodes[_in] = (node_x, node_y, node_text)
      for _f in nn.functions:
        f_name = _f.__class__.__name__
        node_x, node_y, node_text = nodes.get(f_name, ([], [], []))
        node_x.append(pos_x[_f.id_name])
        node_y.append(pos_y[_f.id_name])
        node_text.append(',<br />'.join(
          ['{' + nts_id_name + ': ' + nts.dtype.__str__() + ', ' + str(nts.shape) + '}' for nts_id_name, nts in
           _f.outputs.items()]))
        if f_name not in nodes:
          nodes[f_name] = (node_x, node_y, node_text)

      edges = dict()
      for _f in nn.functions:
        for label_to, (label_from, node_from) in _f.inputs.items():
          to_x, to_y = pos_x[_f.id_name], pos_y[_f.id_name]
          from_x, from_y = pos_x[node_from], pos_y[node_from]
          m_x, m_y = (to_x + from_x) / 2, (to_y + from_y) / 2
          edge_x, edge_y = edges.get(label_to, ([], []))
          edge_x.extend([m_x, to_x, None])
          edge_y.extend([m_y, to_y, None])
          if label_to not in edges:
            edges[label_to] = (edge_x, edge_y)

          edge_x, edge_y = edges.get(label_from, ([], []))
          edge_x.extend([from_x, m_x, None])
          edge_y.extend([from_y, m_y, None])
          if label_from not in edges:
            edges[label_from] = (edge_x, edge_y)
      return nodes, edges, max(y_pos.keys()) + 1

    @self.app.callback(
      [Output(component_id='network-structure', component_property='figure'),
       Output(component_id='searchSpaceAncestry', component_property='figure')],
      [Input(component_id='searchSpaceProjection', component_property='clickData')]
    )
    def update_searchSpace_info(input_data):
      if input_data is None:
        return [{'layout': ss_info_ancestry_layout}, {}]
      id_name = input_data['points'][0]['text']

      dataSaver = DSSqlite3(**db_config)
      individual = dataSaver.get_individual_by_name(id_name)
      # TODO: got exception in next line
      _, ancestry = dataSaver.get_ancestry_for_ind(id_name)
      if ancestry is not None:
        levelOneAnc = [dataSaver.get_ancestry_for_ind(ind)[1] for ind in ancestry.ancestors]
      del dataSaver

      nodes, edges, y_range, x_range = NetworkLayout2(individual.network)
      adapted_layout = dict(ss_info_ancestry_layout)
      adapted_layout['height'] = y_range[1] * 30 + 200
      adapted_layout['yaxis'] = dict(adapted_layout['yaxis'])
      adapted_layout['yaxis']['range'] = [y_range[0], y_range[1]]
      adapted_layout['xaxis'] = dict(adapted_layout['xaxis'])
      adapted_layout['xaxis']['range'] = [x_range[0], x_range[1]]
      nodes = [{'x': node_x,
                'y': node_y,
                'text': node_text,
                'mode': 'markers',
                'marker': {'size': 10,
                           'symbol': 'circle',
                           'color': getStructureColor(name)},
                'hoverinfo': 'text',
                # 'textposition': 'center right',
                'showlegend': True,
                'name': name
                } for name, (node_x, node_y, node_text) in nodes.items()]
      edges = [{'x': edge_x,
                'y': edge_y,
                'mode': 'lines',
                'hoverinfo': 'none',
                'name': name,
                'showlegend': name != dashVis.blank_label,
                'line': {'color': getStructureColor(name)}
                } for name, (edge_x, edge_y) in edges.items()]
      structure_fig = {
        'data': edges + nodes,
        'layout': adapted_layout,
      }

      # ==================

      nodes = list()
      edges = dict()
      nodes.append({
        'x': [0],
        'y': [0],
        'mode': 'markers',
        'hoverinfo': 'text',
        'text': id_name,
        'name': 'selected',
        'showlegend': False,
        'marker': {'size': 10,
                   'symbol': 'dot',
                   'color': '#a55'}
      })
      if ancestry is not None:
        tmp = -(len(ancestry.ancestors) - 1) * .5
        offsets = [tmp + i for i in range(len(ancestry.ancestors))]
        xs, ys = edges.get(ancestry.method, ([], []))
        xs.extend([x for offset in offsets for x in [0, offset, None]])
        ys.extend([y for _ in offsets for y in [0, 1, None]])
        edges[ancestry.method] = xs, ys
        nodes.append({
          'x': [offset for offset in offsets],
          'y': [1 for _ in offsets],
          'mode': 'markers',
          'hoverinfo': 'text',
          'text': ancestry.ancestors,
          'name': 'ancestors 0',
          'showlegend': False,
          'marker': {'size': 10,
                     'symbol': 'dot',
                     'color': [getAncColor(c)
                               for c in ancestry.ancestors]}
        })
        # TODO: fix this
        # for anc, mid in zip(levelOneAnc, offsets):
        #   if anc is not None:
        #     anc_l = len(anc.ancestors)
        #     tmp = -(anc_l - 1) / anc_l * .8 + mid
        #     _offsets = [tmp + i * .8 / (anc_l - 1) for i in range(anc_l)]
        #     xs, ys = edges.get(anc.method, ([], []))
        #     xs.extend([x for offset in _offsets for x in [mid, offset, None]])
        #     ys.extend([y for _ in _offsets for y in [1, 2, None]])
        #     edges[anc.method] = xs, ys
        #     nodes.append({
        #       'x': [offset for offset in _offsets],
        #       'y': [2 for _ in _offsets],
        #       'mode': 'markers',
        #       'hoverinfo': 'text',
        #       'text': anc.ancestors,
        #       'name': 'ancestors 1',
        #       'showlegend': False,
        #       'marker': {'size': 10,
        #                  'symbol': 'dot',
        #                  'color': [getAncColor(c)
        #                            for c in anc.ancestors]}
        #     })

      edges = [{
        'x': xs,
        'y': ys,
        'mode': 'lines',
        'hoverinfo': 'none',
        'name': method,
        'showleged': method != dashVis.blank_label,
        'line': {'color': getAncColor(method)}
      } for method, (xs, ys) in edges.items()]
      adapted_layout = dict(ss_info_ancestry_layout)
      adapted_layout['yaxis'] = dict(adapted_layout['yaxis'])
      adapted_layout['yaxis']['range'] = [-.5, 2.5]
      ancestry_fig = {
        'data': edges + nodes,
        'layout': adapted_layout,
      }
      return [structure_fig, ancestry_fig]

    @self.app.callback(
      [Output(component_id='auto-update-projection', component_property='interval'),
       Output(component_id='auto-update-ancestry', component_property='interval'),
       Output(component_id='auto-update-metrics', component_property='interval')],
      [Input(component_id='update-interval', component_property='value')]
    )
    def change_update_interval(input_data):
      interval = int(input_data) * 1000
      return interval, interval, interval

    @self.app.callback(
      Output(component_id='searchSpaceProjection', component_property='figure')
      , [Input(component_id='auto-update-projection', component_property='n_intervals')]
    )
    def auto_update_projection(data):
      print('=============== begin projection ===============')
      dataSaver = DSSqlite3(**db_config)
      abstract_time_stamps = sorted(dataSaver.get_abstract_time_stamps())
      base_ind = [set(dataSaver.get_individual_functions(name)) for name in
                  dataSaver.get_individual_names_by_abstract_time_stamp(abstract_time_stamps[0])]

      function_vectors = dict()
      projection_texts = dict()
      # for time_stamp in abstract_time_stamps[15:]:
      for name in dataSaver.get_individual_names():
        print(name)
        ind_f = set(dataSaver.get_individual_functions(name))
        v = [len(i.union(ind_f))-len(i.intersection(ind_f)) for i in base_ind]
        function_vectors[name] = v
        projection_texts[name] = dataSaver.get_individual_metrics(name).get('ACC', 0)
        #next(iter(ind.metrics.values())) if ind.metrics else 0)

      xs, ys = zip(*self.manifold.fit_transform(list(function_vectors.values())))
      del dataSaver

      # all_ind_names = set(dataSaver.get_individual_names())
      # set_names = set(projection_texts.keys())
      # new_ind_names = [n for n in all_ind_names if n not in set_names]
      # all_ind_names = set_names.union(new_ind_names)
      #
      # idx_m = len(new_ind_names)
      # print(new_ind_names)
      # for i, ind0 in enumerate(new_ind_names):
      #   # ind0_ = dataSaver.get_individual_by_name(ind0)
      #   ind0_functions = set(dataSaver.get_individual_functions(ind0))
      #   # projection_texts[ind0] = (next(iter(ind0_.metrics.values())) if ind0_.metrics else 0)
      #   projection_texts[ind0] = dataSaver.get_individual_metrics(ind0).get('ACC', 0)
      #   # print(ind0, projection_texts[ind0])
      #   for ind1 in set_names:
      #     if (ind0, ind1) not in projection_distances:
      #       # dist = ind0_.norm(dataSaver.get_individual_by_name(ind1))
      #       ind1_functions = set(dataSaver.get_individual_functions(ind1))
      #       dist = len(ind0_functions.union(ind1_functions))-len(ind0_functions.intersection(ind1_functions))
      #       projection_distances[ind0, ind1] = dist
      #       projection_distances[ind1, ind0] = dist
      #   for j in range(i + 1, idx_m):
      #     ind1 = new_ind_names[j]
      #     # dist = ind0_.norm(dataSaver.get_individual_by_name(ind1))
      #     ind1_functions = set(dataSaver.get_individual_functions(ind1))
      #     dist = len(ind0_functions.union(ind1_functions)) - len(ind0_functions.intersection(ind1_functions))
      #     projection_distances[ind0, ind1] = dist
      #     projection_distances[ind1, ind0] = dist
      #   projection_distances[ind0, ind0] = 0
      #
      # del dataSaver
      #
      # if len(new_ind_names) > 0:
      #   distance = [projection_distances[ind0, ind1] for ind0 in all_ind_names for ind1 in all_ind_names]
      #   distance = np.asarray(distance)
      #   distance = distance.reshape((len(all_ind_names), -1))
      #   self.projection = self.manifold.fit_transform(distance)
      # xs, ys = zip(*self.projection)

      ssprojection = {
        'data': [{
          'x': list(xs),
          'y': list(ys),
          'text': list(projection_texts.keys()),
          'name': 'text',
          'mode': 'markers',
          'marker': {'size': 7,
                     'symbol': ['cross-thin-open' if ev else 'dot' for ev in projection_texts.values()],
                     'color': list(projection_texts.values()),
                     'colorscale': 'Oranges',
                     'showscale': True,
                     },
          'hoverinfo': 'text',
          'showlegend': False,
        }],
        'layout': {
          'plot_bgcolor': '#333',
          'paper_bgcolor': '#333',
          'xaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': False
          },
          'yaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': False,
          },
          'margin': go.layout.Margin(
            l=0, r=0, b=0, t=25, pad=0,
          ),
          'hovermode': 'closest',
        }
      }
      print('update projection')
      return ssprojection

    def hash2binary(h_value, size, mem=dict(), values=dict()):
      if h_value not in mem.get(size, dict()):
        _values = values.get(size, set())
        _size = 2 ** (size ** 2)
        if len(_values) >= _size:
          raise Exception('collision!')
        v = h_value % _size
        while v in _values:
          v += 1
          v = v % _size
        _values.add(v)
        values[size] = _values
        binary = [v & (1 << i) != 0 for i in range(size ** 2)]
        mem_size = mem.get(size, dict())
        mem_size[h_value] = binary
        mem[size] = mem_size
      return mem[size][h_value]

    def binary2coordinates(binary, center=(0, 0), size=1.0):
      edge_l = int(math.sqrt(len(binary)))
      c_x, c_y = center
      coordinates = {True: [], False: []}
      if edge_l % 2 == 0:
        d = size / edge_l
        idx = 0
        for i in range(1, edge_l):
          for j in range(i):
            xs = [c_x + j * d * 2 - (i - 1) * d + k * d for k in range(2)] + \
                 [c_x + j * d * 2 - (i - 1) * d - k * d for k in range(2)] + \
                 [c_x + j * d * 2 - (i - 1) * d, None]
            ys = [c_y + size - k * d - (i - 1) * d for k in range(3)] + \
                 [c_y + size - k * d - (i - 1) * d for k in range(1, -1, -1)] + [None]
            coordinates[binary[idx]].append((xs, ys))
            idx += 1
        for i in range(edge_l, 0, -1):
          for j in range(i):
            xs = [c_x + j * d * 2 - (i - 1) * d + k * d for k in range(2)] + \
                 [c_x + j * d * 2 - (i - 1) * d - k * d for k in range(2)] + \
                 [c_x + j * d * 2 - (i - 1) * d, None]
            ys = [c_y - size - k * d + (i + 1) * d for k in range(3)] + \
                 [c_y - size - k * d + (i + 1) * d for k in range(1, -1, -1)] + [None]
            coordinates[binary[idx]].append((xs, ys))
            idx += 1
      else:
        pass
      return coordinates

    @self.app.callback(
      Output(component_id='ancestry-vis', component_property='figure'),
      [Input(component_id='auto-update-ancestry', component_property='n_intervals'),
       Input(component_id='ancestry-rep-style', component_property='value'),
       Input(component_id='ancestry-ind-style', component_property='value')]
    )
    def auto_update_ancestry(int_v, rep_style, ind_style):
      # return {}
      print('rep_style', rep_style)
      print('ind_style', ind_style)
      marker_size = 1
      marker_gapX = .5
      marker_gapY = 1

      def get_width_depth(individual, anc_dict, mem):
        if individual not in mem:
          if individual in anc_dict:
            mem[individual] = (sum([get_width_depth(anc, anc_dict, mem)[0]
                                    for anc in anc_dict[individual].ancestors]) + \
                               (len(anc_dict[individual].ancestors) - 1) * marker_gapX,
                               max([get_width_depth(anc, anc_dict, mem)[1]
                                    for anc in anc_dict[individual].ancestors]) + marker_gapY + marker_size)
          else:
            mem[individual] = marker_size, marker_size
        return mem[individual]

      def get_x_y(individual, anc_dict, mem, edges, coords, x_offset, y_offset):
        w, h = get_width_depth(individual, anc_dict, mem)
        if individual in anc_dict:
          x = 0
          y = y_offset + marker_size / 2
          edge_to = list()
          _x_offset = x_offset
          w -= (len(anc_dict[individual].ancestors) - 1) * marker_gapX
          for anc in anc_dict[individual].ancestors:
            dw, dh = get_width_depth(anc, anc_dict, mem)
            dx, dy = get_x_y(anc, anc_dict, mem, edges, coords, _x_offset, y_offset + marker_gapY + marker_size)
            edge_to.append((dx, dy))
            x += dw / w * dx
            _x_offset += dw + marker_gapX
          for e in edge_to:
            edges[anc_dict[individual].method] = edges.get(anc_dict[individual].method, []) + [(e, (x, y))]
        else:
          x, y = w / 2 + x_offset, y_offset + marker_size / 2
        coords[(x, y)] = individual
        return x, y

      coordinates = {True: [], False: []}
      ancestry_edges = dict()

      dataSaver = DSSqlite3(**db_config)

      abstract_time_ancestries = dict()
      for abstract_time, ancestry in dataSaver.get_ancestries():
        abstract_time_ancestries[abstract_time] = abstract_time_ancestries.get(abstract_time, []) + [ancestry]

      y_offset = 0
      for abstract_time in sorted(abstract_time_ancestries.keys())[:5]:
        descendants = set()
        ancestors = set()
        anc_dict = dict()
        for ancestry in abstract_time_ancestries[abstract_time]:
          if isinstance(ancestry, AncestryEntity):
            descendants.add(ancestry.descendant)
            ancestors.update(set(ancestry.ancestors))
            anc_dict[ancestry.descendant] = ancestry
          else:
            pass

        mem = dict()
        max_depth = max([get_width_depth(next_gen_ind, anc_dict, mem)[1]
                         for next_gen_ind in descendants.difference(ancestors)])
        x_offset = 0
        for next_gen_ind in descendants.difference(ancestors):
          coords = dict()
          edges = dict()
          coords[get_x_y(next_gen_ind, anc_dict, mem, edges, coords, x_offset, y_offset)] = next_gen_ind
          for (x, y), individual in coords.items():
            c = binary2coordinates(hash2binary(hash(individual), 4), center=(x, y), size=.5)
            coordinates[True].extend(c[True])
            coordinates[False].extend(c[False])
          x_offset += get_width_depth(next_gen_ind, anc_dict, mem)[0] + 2 * marker_gapX
          for anc_method, edge_list in edges.items():
            method_edges = ancestry_edges.get(anc_method, {'x': [], 'y': []})
            for (f_x, f_y), (t_x, t_y) in edge_list:
              method_edges['x'].extend([f_x, t_x, None])
              method_edges['y'].extend([f_y, t_y, None])
            ancestry_edges[anc_method] = method_edges

        y_offset -= max_depth + marker_gapY

      ancestry = {
        'data': [{
          'x': coordinates['x'],
          'y': coordinates['y'],
          'mode': 'lines',
          'line': {
            'color': getAncColor(method),
            'width': 2.0,
          },
          'name': method,
          'showlegend': True
        } for method, coordinates in ancestry_edges.items()] + [{
          'x': xs,
          'y': ys,
          'mode': 'none',
          'fill': 'tozeroy',
          'fillcolor': '#ffffffff',
          'hoverinfo': 'none',
          'line': {
            'color': '#000f',
            'width': .5,
          },
          'showlegend': False,
        } for xs, ys in coordinates[True]] + [{
          'x': xs,
          'y': ys,
          'mode': 'none',
          'fill': 'tozeroy',
          'fillcolor': '#000000ff',
          'hoverinfo': 'none',
          'line': {
            'color': '#ffff',
            'width': .5,
          },
          'showlegend': False,
        } for xs, ys in coordinates[False]],
        'layout': {
          'plot_bgcolor': '#333',
          'paper_bgcolor': '#333',
          'xaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': False,
          },
          'yaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': True,
            'scaleanchor': 'x',
            'scaleratio': 1,
          },
          'margin': go.layout.Margin(
            l=0, r=0, b=0, t=25, pad=0,
          ),
          'hovermode': 'closest',
        }
      }
      return ancestry

    @self.app.callback(
      [Output(component_id='metric-tabs', component_property='children')],
      [Input(component_id='auto-update-metrics', component_property='n_intervals')]
    )
    def auto_update_metrics(data):
      dataSaver = DSSqlite3(**db_config)
      individual_names = dataSaver.get_individual_names()
      data = dict()
      count = 0
      for ind in individual_names:
        if ind in individuals:
          continue
        count += 1
        if count > 10:
          break
        # individual = dataSaver.get_individual_by_name(ind)
        metrics = dataSaver.get_individual_metrics(ind)
        time_stamps = dataSaver.time_stamps_by_individual_name(ind)
        individuals[ind] = metrics, time_stamps
        # if len(time_stamps) > 1:
        #   print(metrics.items())
        #   print(time_stamps)
        # print(ind)
      for metrics, time_stamps in individuals.values():
        for m, value in metrics.items():
          data_m = data.get(m, dict())
          for real, abstract in time_stamps:
            data_m[abstract] = data_m.get(abstract, []) + [value]
          data[m] = data_m
      del dataSaver
      # print('================')
      tabs = list()
      for m_label, m_data in data.items():
        xs = list()
        ys = list()
        for t in sorted(m_data.keys()):
          for v in m_data[t]:
            xs.append(t)
            ys.append(v)
        tabs.append(dcc.Tab(className='custom-tab', selected_className='custom-selected-tab',
                            label=m_label, children=[
            dcc.Graph(className='graph', id=m_label, figure={'data': [{'x': xs,
                                                                       'y': ys,
                                                                       'mode': 'markers'}],
                                                             'layout': {}})
          ]))
      return [tabs]

    if not kwargs.get(self.arg_AS_SERVER, False):
      self.app.run_server(debug=False, port=8050)

  @property
  def server(self):
    return self.app.server

  pass


if __name__ == '__main__':
  dashVis_ = dashVis(**{dashVis.arg_AS_SERVER: False,
                        dashVis.arg_DB_CONFIGS: {
                          DSSqlite3.arg_FILE: '/path/to/dabase/file.db3'
                        }})
  # else:
  #   dashVis_ = dashVis(asServer=True)
  #   server = dashVis_.server

  #
  # dcc.Dropdown(
  #   options=[
  #     {'label': 'New York City', 'value': 'NYC'},
  #     {'label': u'Montréal', 'value': 'MTL'},
  #     {'label': 'San Francisco', 'value': 'SF'}
  #   ],
  #   value='MTL'
  # ),
  # dcc.Dropdown(
  #   options=[
  #     {'label': 'New York City', 'value': 'NYC'},
  #     {'label': u'Montréal', 'value': 'MTL'},
  #     {'label': 'San Francisco', 'value': 'SF'}
  #   ],
  #   value=['MTL', 'SF'],
  #   multi=True
  # ),
  # dcc.RadioItems(
  #   options=[
  #     {'label': 'New York City', 'value': 'NYC'},
  #     {'label': u'Montréal', 'value': 'MTL'},
  #     {'label': 'San Francisco', 'value': 'SF'}
  #   ],
  #   value='MTL'
  # ),
  # dcc.Checklist(
  #   options=[
  #     {'label': 'New York City', 'value': 'NYC'},
  #     {'label': u'Montréal', 'value': 'MTL'},
  #     {'label': 'San Francisco', 'value': 'SF'}
  #   ],
  #   value=['MTL', 'SF']
  # ),
  # dcc.Input(value='MTL', type='text'),
  # dcc.Slider(
  #   min=0,
  #   max=9,
  #   marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
  #   value=5,
  # ),

  # html.Button(id='submit-button', n_clicks=0, children='Submit'),
  # [Input('submit-button', 'n_clicks')],
  # dcc.Input(id='input-1-state', type='text', value='Montréal'),
  # dcc.Input(id='input-2-state', type='text', value='Canada'),
  # [State('input-1-state', 'value'),
  #  State('input-2-state', 'value')])

  # dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
