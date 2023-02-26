import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import dash
import dash_table
from dash_table.Format import Format, Group
import dash_table.FormatTemplate as FormatTemplate
from datetime import datetime as dt
from app import app
import dash_bootstrap_components as dbc
from dash_table.Format import Format, Group, Scheme
from param_layout import (fasttext_input_boxes, fasttext_dropdowns, glove_input_boxes, 
    bert_input_boxes, roberta_gpt2_input_boxes, kmeans_dropdowns,kmeans_dropdowns, 
    kmeans_input_boxes, agglomerative_dropdowns, doc2vec_input_boxes, word2vec_input_boxes, word2vec_dropdowns,
    sentence_bert_input_boxes, sentence_bert_dropdowns)
import plotly.express as px
from constants import available_methods
from style import (corporate_colors, externalgraph_rowstyling, externalgraph_colstyling, filterdiv_borderstyling,
    navbarcurrentpage, recapdiv, corporate_font_family, get_emptyrow, dropdown_style)


def get_header():

    header = html.Div([

        html.Div([], className = 'col-2'), #Same as img width, allowing to have the title centrally aligned

        html.Div([
            html.H1(children='Document Clustering Dashboard',
                    style = {'textAlign' : 'center'}
            )],
            className='col-8',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.Img(
                    src = app.get_asset_url('logo.png'),
                    height = '43 px',
                    width = 'auto')
            ],
            className = 'col-2',
            style = {
                    'align-items': 'center',
                    'padding-top' : '1%',
                    'height' : 'auto'})

        ],
        className = 'row',
        style = {'height' : '4%',
                'background-color' : corporate_colors['superdark-green']}
        )

    return header

def get_navbar(p = 'tab1'):


    navbar_tab1 = html.Div([

        html.Div([], className = 'col-1'),

        html.Div([
            dcc.Link(
                html.H4(children = 'User Screen',
                        style = navbarcurrentpage),
                href='/apps/tab1'
                )
        ],
        className='col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Simulator Screen'),
                href='/apps/tab2'
                )
        ],
        className='col-3'),
        html.Div([
            dcc.Link(
                html.H4(children = 'File Upload Screen'),
                    href='/apps/tab3'
                )
        ],
        className='col-3'),



        html.Div([], className = 'col-1')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    );

    navbar_tab2 = html.Div([

        html.Div([], className = 'col-1'),

        html.Div([
            dcc.Link(
                html.H4(children = 'User Screen'),
                href='/apps/tab1'
                )
        ],
        className='col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Simulator Screen',
                style = navbarcurrentpage),
                    href='/apps/tab2'
                )
        ],
        className='col-3'),
        html.Div([
            dcc.Link(
                html.H4(children = 'File Upload Screen'),
                    href='/apps/tab3'
                )
        ],
        className='col-3'),

        html.Div([], className = 'col-1')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    );

    navbar_tab3 = html.Div([

        html.Div([], className = 'col-1'),

        html.Div([
            dcc.Link(
                html.H4(children = 'User Screen'),
                href='/apps/tab1'
                )
        ],
        className='col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Simulator Screen',),
                    href='/apps/tab2'
                )
        ],
        className='col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'File Upload Screen', 
                    style = navbarcurrentpage),
                    href='/apps/tab3'
                )
        ],
        className='col-3'),

        html.Div([], className = 'col-1')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    );

    print("p: ", p)

    if p == 'tab1':
        return navbar_tab1
    elif p == 'tab2':
        return navbar_tab2
    elif p == 'tab3':
        return navbar_tab3
    else:
        return navbar_tab1

tab1 = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('tab1'),

            #####################
    #Row 5 : Charts
    html.Div([ # External row

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

        html.Div([ # External 10-column

            html.H2(children = "Graphs",
                    style = {'color' : corporate_colors['white']}),
            html.Div([ # Internal row
                dcc.Dropdown(id="my-dynamic-dropdown", 
                    style={'font-size': '13px', 'color' : 'white', 'white-space': 'nowrap', 'text-overflow': 'ellipsis', 
                    'backgroundColor': corporate_colors['dark-green'], 'width':750, 'margin-right':20},
                    placeholder='Search for a word'),
                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='wordcount-graph'),
                    dcc.Input(
                        id="input_range", type="number", placeholder="input with range",
                        min=1, max=100, step=1, value=1, style = dropdown_style
                    ),
                    html.Div([
                         dbc.Modal(
                        [
                            dbc.ModalHeader(id = 'modal-header', style={'backgroundColor':corporate_colors['dark-green']}),
                            dbc.ModalBody(id = 'modal-body', style={'backgroundColor': corporate_colors['light-grey']}),
                            dbc.ModalFooter(
                                dbc.Button("Close", id="close", className="ml-auto")
                                , style={'backgroundColor': corporate_colors['dark-green']}
                            ),
                        ],
                        id="modal",
                        scrollable=True,

                    ),
                    ]),
                    html.Div([
                         dbc.Modal(
                        [
                            dbc.ModalHeader(id = 'modal-header2', style={'backgroundColor':corporate_colors['dark-green']}),
                            dbc.ModalBody(id = 'modal-body2', style={'backgroundColor': corporate_colors['light-grey']}),
                            dbc.ModalFooter(
                                dbc.Button("Close", id="close2", className="ml-auto")
                                , style={'backgroundColor': corporate_colors['dark-green']}
                            ),
                        ],
                        id="modal2",
                        scrollable=True,

                    ),
                    ]),
                    html.Div(id='dynamic-list'),
                    html.Div(id='dynamic-list2')
                    
                ],
                className = 'col-12'),



            ],
            className = 'row'), # Internal row

            html.Div([ # Internal row
                                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='embedding-graph',
                        figure=px.scatter()),
                    html.Div([
                      dcc.Dropdown(
                        id='embedding-method2',
                        options=[{'label': key, 'value': key} for key in available_methods],
                        value='doc2vec',
                        style = dropdown_style
                    ),dcc.RadioItems(
                        id='embedding-method-options2',
                        labelStyle={'display': 'inline-block', 'margin': '10px', 'color': 'white'},
                        value="avg"
                    ),dcc.Dropdown(
                        id='vis-method',
                        options=[{'label': i, 'value': i} for i in ["pca", "tsne"]],
                        value='pca',
                        style = dropdown_style
                    ), dcc.Input(
                        id="cluster_count", type="number", placeholder="clusters",
                        min=1, max=100, step=1, value=2,
                        style = dropdown_style
                    ),dcc.Dropdown(
                        id='clustering-method2',
                        options=[{'label': i, 'value': i} for i in ['kmeans_scores', 'hierarthical_scores', 'dbscan_scores']],
                        value='hierarthical_scores',
                        style = dropdown_style
                    ),dcc.Dropdown(
                        id='clustering-index2',
                        options=[{'label': i, 'value': i} for i in ['Silhouette Index', 'DB Index', 'Calinski Harabasz Score', 'Rand Index', 'Homogeneity Score']],
                        value='Silhouette Index',
                        style = dropdown_style
                    ),html.P(children="perplexity", style = {'color' : corporate_colors['white']}),
                    dcc.Input(
                        id="perplexity", type="number", placeholder="perplexity",
                        min=5, max=50, step=1, value=30,
                        style = dropdown_style
                    )],
                      ),
                ],
                className = 'col-12'),


            ],
            className = 'row') # Internal row


        ],
        className = 'col-10',
        style = externalgraph_colstyling), # External 10-column

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

    ],
    className = 'row',
    style = externalgraph_rowstyling
    ), # External row


    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 3 : Filters
    html.Div( # External row

        [],
        className = 'row',
        style = externalgraph_rowstyling

    ) # External row
]);


tab2 = html.Div([
      #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('tab2'),

    html.Div([ # External row

        html.Div([ # External 12-column

            html.Div([ # Internal row

                #Internal columns
                html.Div([
                ],
                className = 'col-2'), # Blank 2 columns

                #Filter pt 1
                html.Div([

                    dcc.Dropdown(
                        id='embedding-method',
                        options=[{'label': key, 'value': key} for key in available_methods],
                        value='fasttext',
                        style={'backgroundColor': corporate_colors['dark-green']}
                    ),
                    dcc.RadioItems(
                        id='embedding-method-options',
                        labelStyle={'display': 'inline-block', 'margin': '10px'},
                        value="avg"
                    ),

                    html.Div([
                        html.Button('Simulate', id='simulate1', n_clicks=0, 
                            style= {'width': '30%', 'height': '25%', 'justifyContent': 'center', 'alignItems': 'center'}),
                         html.Button('Batch Simulate', id='batch_simulate', n_clicks=0, 
                            style= {'width': '30%', 'height': '25%', 'justifyContent': 'center', 'alignItems': 'center'}),
                        html.Button('Recalculate Scores', id='recalculate_clusters', n_clicks=0, 
                            style= {'width': '30%', 'height': '25%', 'justifyContent': 'center', 'alignItems': 'center'}),

                    ], style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'center', 'padding':'1%', 'height':'180px'})
                ],
                style={'marginTop': '1%'},
                className = 'col-8'), # Filter part 1
                dcc.Loading(
                        id="loading-1",
                        type="circle",
                        children=html.Div(id="loading-output-1"),
                        parent_className = 'col-2',
                        style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'flex-end'}
                    ),
                dcc.Loading(
                        id="loading-3",
                        type="circle",
                        children=html.Div(id="loading-output-3"),
                        parent_className = 'col-2',
                        style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'flex-end'}
                    ),
                 dbc.Alert(
                            id="recalculation_result",
                            fade=True,
                            is_open=False,
                            dismissable=True,
                            color='info'
                        )

            ],
            className = 'row') # Internal row

        ],
        className = 'col-12',
        style = filterdiv_borderstyling) # External 12-column

    ],
    className = 'row'), # External row
     #####################
    #Row 4
    get_emptyrow(),

    html.Div([ # External row

        html.Div([ # External 12-column

             html.Div([ # Internal row

                #Internal columns
                html.Div([
                ],
                className = 'col-2'), # Blank 2 columns

                
                html.Div([
                    #Filter pt 1
                    dbc.Row(dbc.Col( html.Div(children = fasttext_input_boxes,
                        id = 'fasttext_input_boxes'
                    ))),
                    dbc.Row(dbc.Col( html.Div(children = glove_input_boxes,
                        id = 'glove_input_boxes'
                    ))),
                    dbc.Row(dbc.Col( html.Div(children = bert_input_boxes,
                        id = 'bert_input_boxes'
                    ))),
                    dbc.Row(dbc.Col( html.Div(children = roberta_gpt2_input_boxes,
                        id = 'roberta_gpt2_input_boxes'
                    ))),
                    dbc.Row(dbc.Col( html.Div(children = doc2vec_input_boxes,
                        id = 'doc2vec_input_boxes'
                    ))),
                    dbc.Row(dbc.Col( html.Div(children = word2vec_input_boxes,
                        id = 'word2vec_input_boxes'
                    ))),
                    dbc.Row(dbc.Col( html.Div(children = sentence_bert_input_boxes,
                        id = 'sentence_bert_input_boxes'
                    ))),
                ],
                id="input_boxes_container",
                className = 'col-4'), # Filter part 1

                #Filter pt 2
                html.Div([
                    dbc.Row(dbc.Col(html.Div(children = fasttext_dropdowns,
                        id='fasttext_dropdowns'
                    ))),
                    dbc.Row(dbc.Col(html.Div(children = word2vec_dropdowns,
                        id='word2vec_dropdowns'
                    ))),
                    dbc.Row(dbc.Col(html.Div(children = sentence_bert_dropdowns,
                        id='sentence_bert_dropdowns'
                    )))                  
                ],
                className = 'col-4',
                ), # Filter part 2

                html.Div([
                ],
                className = 'col-2') # Blank 2 columns


            ],
            className = 'row'), # Internal row

             html.Div([

                html.Div([
                    html.Button('Simulate Individual', id='simulate2', n_clicks=0, 
                        style= {'width': '32%', 'height': '100%', 'justifyContent': 'center', 'alignItems': 'center', 'marginRight': 50}),

                    
                ], style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'center', 'padding':'1%',},
                className = 'col-10'),
                dcc.Loading(
                    id="loading-2",
                    type="circle",
                    children=html.Div(id="loading-output-2"),
                    parent_className = 'col-2'
                ),
            ], className='row')
                
        ],
        className = 'col-12',
        style = filterdiv_borderstyling), # External 12-column

    ],
    className = 'row'),
    #Row 5 : Charts
    html.Div([ # External row
        html.Div([
        ],
        className = 'col-1'), # Blank 1 column


     html.Div(
     [ # External 10-column

            html.H2(children = "Clustering Results",
                    style = {'color' : corporate_colors['white']}),

        html.Div([ # Internal row - RECAPS

            html.Div([],className = 'col-1'), # Empty column


            html.Div([
                    dcc.Dropdown(
                        id='clustering-index',
                        options=[{'label': i, 'value': i} for i in ['Silhouette Index', 'DB Index', 'Calinski Harabasz Score', 'Rand Index', 'Homogeneity Score']],
                        value='Rand Index',
                        style = dropdown_style
                    ),

            ],
            className = 'col-10'),

            html.Div([],className = 'col-1') # Empty column

        ],
        className = 'row',
        ),




            html.Div([ # Internal row - RECAPS

                html.Div([],className = 'col-4'), # Empty column

                html.Div([
                    dash_table.DataTable(
                        id='recap-table',
                        style_header = {
                            'backgroundColor': 'transparent',
                            'fontFamily' : corporate_font_family,
                            'font-size' : '1rem',
                            'color' : corporate_colors['light-green'],
                            'border': '0px transparent',
                            'textAlign' : 'center'},
                        style_cell = {
                            'backgroundColor': 'transparent',
                            'fontFamily' : corporate_font_family,
                            'font-size' : '0.85rem',
                            'color' : corporate_colors['white'],
                            'border': '0px transparent',
                            'textAlign' : 'center'},
                        cell_selectable = False,
                        column_selectable = False
                    )
                ],
                className = 'col-4'),

                html.Div([],className = 'col-4') # Empty column

            ],
            className = 'row',
            style = recapdiv
            ), # Internal row - RECAPS

            html.Div([ # Internal row - RECAPS

                html.Div([],className = 'col-1'), # Empty column


                html.Div([
                    dcc.Graph(id = 'best-embedding-results')
                ],
                className = 'col-10'),

                html.Div([],className = 'col-1') # Empty column

            ],
            className = 'row',
            ), # Internal row - RECAPS
            html.Div([
                dcc.Dropdown(
                        id='clustering-method',
                        options=[{'label': i, 'value': i} for i in ['kmeans_scores', 'hierarthical_scores', 'dbscan_scores']],
                        value='hierarthical_scores',
                        style = dropdown_style
                    ),
            ]),
             html.Div([ # Internal row

                #Internal columns
                html.Div([
                ],
                className = 'col-2'), # Blank 2 columns
                html.Div([
                    #Filter pt 1
                    dbc.Row(dbc.Col( html.Div(children = kmeans_input_boxes,
                        id = 'kmeans_input_boxes'
                    )))
                ],
                className = 'col-4'), # Filter part 1

                #Filter pt 2
                html.Div([
                    dbc.Row(dbc.Col( html.Div(children = agglomerative_dropdowns,
                        id = 'agglomerative_dropdowns'
                    ))),
                    dbc.Row(dbc.Col(html.Div(children = kmeans_dropdowns,
                        id='kmeans_dropdowns'
                    )))
                ],
                className = 'col-4',
                ), # Filter part 2

                html.Div([
                ],
                className = 'col-2') # Blank 2 columns


            ],
            className = 'row'), # Internal row

                        html.Div([ # Internal row - RECAPS

                html.Div([],className = 'col-1'), # Empty column

                html.Div([
                    dcc.Graph(id = 'correlation-coef-of-embeddings-graph')
                ],
                className = 'col-10'),

                html.Div([],className = 'col-1') # Empty column

            ],
            className = 'row',
            ), # Internal row - RECAPS


        ],
        className = 'col-10',
        style = externalgraph_colstyling), # External row)
      html.Div([
        ],
        className = 'col-1'), # Blank 1 column

    ],
    className = 'row',
    style = externalgraph_rowstyling
    ), # External row
]);

tab3 = html.Div([
    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('tab3'),

    html.Div([
        dcc.Upload(
            id='upload-files',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False,
            accept='.zip'
        ),
        dcc.Loading(
                            id="loading-4",
                            type="circle",
                            children=html.Div(id="loading-output-4"),
                            parent_className = 'col-2',
                            style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'flex-end'}
                        ),

        dcc.Upload(
            id='upload-annotation-file',
            children=html.Div(
                ['Drag and Drop Annotation File']
            ),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False,
            accept='.json'
        ),
        dcc.Loading(
                        id="loading-5",
                        type="circle",
                        children=html.Div(id="loading-output-5"),
                        parent_className = 'col-2',
                        style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'flex-end'}
                    ),
        dbc.Alert(
            children=html.Div(
                ['Document does not exist!!!']
            ),
            id="doc_not_exists_alert",
            fade=True,
            is_open=False,
            dismissable=True
        ),

    ])
]);






