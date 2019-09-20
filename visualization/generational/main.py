import random

import dash
import dash_core_components as dcc
import dash_html_components as html
import numba
import numpy as np
import plotly.graph_objs as go
# from sklearn.manifold import MDS
import umap
from dash.dependencies import Input, Output, ClientsideFunction

from LAMARCK_ML.architectures.functions import Dense, Merge, Conv2D, Flatten, Pooling2D
from LAMARCK_ML.architectures.neuralNetwork import NeuralNetwork
from LAMARCK_ML.data_util import Shape, DimNames, DFloat, TypeShape, IOLabel
from LAMARCK_ML.datasets import UncorrelatedSupervised

random.seed()


class dashVis():
  """
  Visualizes the NEA parameters and its progress if connected to a database.
  For Unix:
  - expose server with 'server = obj.server'
  - start server with 'gunicorn my_python_file:server'
  """

  def __init__(self, **kwargs):
    # TODO: remove test data
    """begin test data"""

    @numba.njit
    def metric(a, b):
      return a[0].norm(b[0])

    train_samples = 1
    # train_X = [np.random.rand(12288).reshape((64, 64, 3)) for _ in range(train_samples)]
    train_X = [np.random.rand(20) for _ in range(train_samples)]

    # train_Y = [np.random.rand(1024) for _ in range(train_samples)]
    train_Y = [np.random.rand(10) for _ in range(train_samples)]

    _data = TypeShape(DFloat, Shape((DimNames.HEIGHT, 32),
                                    (DimNames.WIDTH, 32),
                                    (DimNames.CHANNEL, 3)))
    _data2 = TypeShape(DFloat, Shape((DimNames.HEIGHT, 32),
                                    (DimNames.WIDTH, 32),
                                    (DimNames.CHANNEL, 3)))
    # _data = TypeShape(DFloat, Shape((DimNames.UNITS, 20)))
    batch = 1
    dataset = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA: _data,
                                                 IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 10)))},
                                     name='Dataset')
    IOLabel.DATA2 = 'DATA2'
    dataset2 = UncorrelatedSupervised(train_X=train_X,
                                     train_Y=train_Y,
                                     batch=batch,
                                     typeShapes={IOLabel.DATA2: _data2,
                                                 IOLabel.TARGET: TypeShape(DFloat, Shape((DimNames.UNITS, 10)))},
                                     name='Dataset2')
    datasets = [dataset]
    IOLabel.DS1 = 'DS1'
    IOLabel.DS2 = 'DS2'
    self.inputs = {IOLabel.DS1: (IOLabel.DATA, _data, dataset.id_name),
                   IOLabel.DS2: (IOLabel.DATA2, _data, dataset2.id_name)}
    # self.inputs = {IOLabel.DS1: (IOLabel.DATA, _data, dataset.id_name)}

    # outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 512))
    outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 10))
    outShape1 = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 15))
    self.outputs = {'out0': TypeShape(DFloat, outShape), 'out1':TypeShape(DFloat, outShape1)}
    # self.outputs = {'out': TypeShape(DFloat, outShape)}
    self.functions = [Merge, Conv2D, Flatten, Dense]
    # self.functions = [Dense, Merge]
    blueprints = dict()
    self.projection = None
    self.index = 0
    _nn = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: dict(self.inputs),
                           NeuralNetwork.arg_OUTPUT_TARGETS: self.outputs,
                           NeuralNetwork.arg_FUNCTIONS: self.functions,
                           NeuralNetwork.arg_RECOMBINATION_PROBABILITY: 1})
    blueprints['Network_' + str(self.index)] = _nn
    self.index += 1
    for i in range(5):
      nn = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: dict(self.inputs),
                            NeuralNetwork.arg_OUTPUT_TARGETS: self.outputs,
                            NeuralNetwork.arg_FUNCTIONS: self.functions,
                           NeuralNetwork.arg_RECOMBINATION_PROBABILITY: 1})
      blueprints['Network_' + str(self.index)] = nn
      self.index += 1
      print(i)

    for i in range(5):
      nn = random.choice(list(blueprints.values())).mutate(1)[0]
      blueprints['Network_' + str(self.index)] = nn
      self.index += 1
      print(i)

    for i in range(5):
      nn = random.choice(list(blueprints.values())).recombine(random.choice(list(blueprints.values())))[0]
      blueprints['Network_' + str(self.index)] = nn
      self.index += 1
      print(i)

    # self.manifold = MDS(n_components=2,
    #                     max_iter=1,
    #                     n_init=1,
    #                     dissimilarity='precomputed'
    #                     )
    self.manifold = umap.UMAP(n_components=2,
                              n_neighbors=2,
                              min_dist=1,
                              random_state=0,
                              metric='precomputed',
                              n_epochs=11,
                              learning_rate=.1,
                              )

    """end test data"""

    # TODO: add DB class
    self.db_config = kwargs.get('dbConfig')

    self.metrics = [('metric_id0', 'metric label 0'),
                    ('metric_id1', 'metric label 1')]

    ss_info_ancestry_layout = {
      'plot_bgcolor': '#00000000',
      'paper_bgcolor': '#00000000',
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
                 'y': -0.1, },
      'height': 800,
    }

    self.app = dash.Dash(__name__)
    self.app.layout = html.Div(id='root', children=[
      html.H1(children="LAMARCK_ML Inspector!!"),
      dcc.Tabs(parent_className='custom-tabs', className='contentArea', id='plotArea', children=[
        dcc.Tab(className='custom-tab', selected_className='custom-selected-tab', label='NEA Parameters',
                id='neaParamA',
                children=[html.H1('NEA Parameters'),
                          html.P([
                            'Web-UI update interval [s]: ',
                            dcc.Input(
                              id='update-interval',
                              type='number',
                              value='120'
                            )
                          ]),
                          ]),
        dcc.Tab(className='custom-tab', selected_className='custom-selected-tab', label='Search Space', id='SSViewA',
                children=[html.H1('Search Space View'),
                          dcc.Interval(
                            id='auto-update-projection',
                            interval=1 * 1000,
                            n_intervals=0
                          ),
                          html.Div(className='graphArea', children=[
                            dcc.Graph(className='graph', id='searchSpaceProjection'),
                          ]),
                          html.Div(className='infoArea', children=[
                            dcc.Tabs(className='SideInfo', children=[
                              dcc.Tab(className='custom-tab', selected_className='custom-selected-tab',
                                      label='Structure',
                                      children=[
                                        dcc.Graph(className='graph', id='network-structure',
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
                                                  })
                                      ]),
                              dcc.Tab(className='custom-tab', selected_className='custom-selected-tab',
                                      label='Ancestry',
                                      children=[
                                        dcc.Graph(className='graph', id='searchSpaceAncestry')
                                      ]),
                            ]),
                          ]),
                          ]),
        dcc.Tab(className='custom-tab', selected_className='custom-selected-tab', label='Ancestry', id='AncestryViewA',
                children=[html.H1('Ancestry View'),
                          dcc.Interval(
                            id='auto-update-ancestry',
                            interval=1 * 1000,
                            n_intervals=0
                          ),
                          html.Div(className='graphArea', children=[
                            dcc.Graph(className='graph', id='ancestry-vis'),
                            # TODO: ancestry plot
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
                            n_intervals=0
                          ),
                          html.Div(className='metricArea', children=[
                            dcc.Tabs(children=[
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
      blueprint = blueprints.get(id_name)
      nodes, edges, y_range = NetworkLayout(blueprint, datasets=datasets)

      adapted_layout = dict(ss_info_ancestry_layout)
      adapted_layout['height'] = y_range * 50 + 100
      structure_fig = {
        'data': [
                  {'x': edge_x,
                   'y': edge_y,
                   'mode': 'lines',
                   'hoverinfo': 'none',
                   'name': name,
                   'showlegend': True
                   }
                  for name, (edge_x, edge_y) in edges.items()] +
                [{'x': node_x,
                  'y': node_y,
                  'text': node_text,
                  'mode': 'markers',
                  'marker': {'size': 10,
                             'symbol': 'dot-open'},
                  'hoverinfo': 'text',
                  # 'textposition': 'center right',
                  # 'textfont': {'color': '#ffffff'},
                  'showlegend': True,
                  'name': name
                  } for name, (node_x, node_y, node_text) in nodes.items()],
        # 'layout': ss_info_ancestry_layout,
        'layout': adapted_layout,
      }

      ancestry_fig = {}
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
      # nn = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: dict(self.inputs),
      #                       NeuralNetwork.arg_OUTPUT_TARGETS: self.outputs,
      #                       NeuralNetwork.arg_FUNCTIONS: self.functions})
      # blueprints['Network_' + str(self.index)] = nn
      # self.index += 1
      # print('prev', self.projection, isinstance(self.projection, np.ndarray))
      nn_texts, networks = zip(*blueprints.items())
      # print(nn_texts)
      distance = [ind0.norm(ind1) for ind0 in networks for ind1 in networks]
      distance = np.asarray(distance)
      distance = distance.reshape((len(networks), len(networks)))
      # print(distance)
      # if self.projection is not None:
      #   self.projection = np.concatenate((self.projection, np.asarray([[0, 0]])))
      self.projection = self.manifold.fit_transform(distance)
      xs, ys = zip(*self.projection)
      # print(xs)
      # print(ys)

      ssprojection = {
        'data': [{
          'x': list(xs),
          'y': list(ys),
          # 'x': [1, 2, 3, 4],
          # 'y': [4, 1, 3, 5],
          'text': nn_texts,
          'name': 'Trace1',
          'mode': 'markers',
          'marker': {'size': 20,
                     'symbol': 'cross-open'},
          'hoverinfo': 'text',
        },
        ],
        'layout': {
          'plot_bgcolor': '#00000000',
          'paper_bgcolor': '#00000000',
          'xaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': False
          },
          'yaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': False,
            # 'ticks':'outside',
            # 'tickwidth':2,
            # 'tickcolor':'crimson',
            # 'ticklen':10,
          },
          'margin': go.layout.Margin(
            l=0, r=0, b=0, t=25, pad=0,
          )
        }
      }
      print('update projection')
      return ssprojection

    @self.app.callback(
      [Output(component_id=cid, component_property='figure') for cid, _ in self.metrics],
      [Input(component_id='auto-update-metrics', component_property='n_intervals')]
    )
    def auto_update_metrics(data):
      print('update metrics')
      # TODO: db connection
      metrics = [{'data': [{
        'x': [0],
        'y': [0],
        'mode': 'markers',
      }]} for _, metric_label in self.metrics]
      return metrics

    @self.app.callback(
      Output(component_id='ancestry-vis', component_property='figure'),
      [Input(component_id='auto-update-ancestry', component_property='n_intervals'),
       Input(component_id='ancestry-rep-style', component_property='value'),
       Input(component_id='ancestry-ind-style', component_property='value')]
    )
    def auto_update_ancestry(int_v, rep_style, ind_style):
      print('rep_style', rep_style)
      print('ind_style', ind_style)
      return {}

    if not kwargs.get('asServer', False):
      self.app.run_server(debug=False, port=8050)

  @property
  def server(self):
    return self.app.server

  pass


if __name__ == '__main__':
  dashVis_ = dashVis(asServer=False)
else:
  dashVis_ = dashVis(asServer=True)
  server = dashVis_.server

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
