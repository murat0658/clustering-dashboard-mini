# -*- coding: utf-8 -*-
from app import app
from dash.dependencies import Input, Output, State, ALL
import dash
import os, re
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from hyperopt import hp
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_html_components as html
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dash.exceptions import PreventUpdate
import json
from itertools import combinations
from scipy.stats import pearsonr
import dash_bootstrap_components as dbc
from dash_table.Format import Format, Scheme

from constants import available_methods, method_option_desc, index_value_key, initial_files_path
from callbacks2.ScoreCalculator import ScoreCalculator
from callbacks2.TrialsDAO import TrialsDAO
from callbacks2.Tokenizer import Tokenizer
from callbacks2.FileUploader import FileUploader

from style import corporate_colors, corporate_layout
from search_spaces import affinity_vals, linkage_vals, init_vals, algorithm_vals
from runMethod import runMethod

tokenizer = None
df = None
word_dict = None
scoreCalculator = None

if(os.path.isdir('../initial-files')):
    tokenizer = Tokenizer()
    print("tokenizer initialized")
    df = tokenizer.create_word_data_frame()
    print("Word data frame initialized")
    word_dict = tokenizer.create_word_dict()
    print("Word dict initialized")
    scoreCalculator = ScoreCalculator()
    print("Score calculator initialized")

##########################################


def get_embeddings(embedding_method, embedding_method_option, clustering_index = 'silhouette_score', clustering_method = 'hierarthical_scores'):
    if embedding_method == 'skip thoughts':
        one_element_list = scoreCalculator.get_skip_thoughts_embeddings(clustering_index, clustering_method)
    else:
        one_element_list = TrialsDAO(embedding_method, embedding_method_option).get_document(clustering_index, clustering_method)
    if(len(one_element_list) > 0):
        return one_element_list[0]['embeddings']
    return {}

def get_columns_of_recap_table(cluster_counts):
    return [{"name": "Method", 'id': "method"}, {"name": "option", "id": "option"}] +  [{"name": str(cluster_count) + " Cluster", "id": str(cluster_count), 'type' : 'numeric', 'format' : Format(scheme=Scheme.fixed, precision=4)} for cluster_count in cluster_counts]

@app.callback(
    Output('loading-2', 'children'),    
    Output('loading-1', 'children'),
    Output('recap-table', 'data'),
    Output('recap-table', 'columns'),
    Input('simulate1', 'n_clicks'),
    Input('batch_simulate', 'n_clicks'),
    Input('simulate2', 'n_clicks'),
    Input('clustering-index', 'value'),
    Input('clustering-method', 'value'),
    State('embedding-method', 'value'),
    State('model', 'value'),
    State('lr', 'value', ),
    State('dim', 'value', ),
    State('ws', 'value', ),
    State('minCount', 'value'),
    State('minn', 'value'),
    State('maxn', 'value'),
    State('neg', 'value'),
    State('wordNgrams', 'value'),
    State('loss', 'value'),
    State('vocab_min_count', 'value'),
    State('vector_size', 'value'),
    State('max_iter', 'value'),
    State('window_size', 'value'),
    State('x_max', 'value'),
    State('attention_probs_dropout_prob', 'value'),
    State('num_train_epochs', 'value'),
    State('learning_rate', 'value'),
    State('train_batch_size', 'value'),
    State('warmup_steps', 'value'),
    State('max_seq_length', 'value'),
    State('vector_size_doc2vec', 'value'),
    State('window_doc2vec', 'value'),
    State('vector_size_word2vec', 'value'),
    State('window_word2vec', 'value'),
    State('min_count', 'value'),
    State('hs', 'value'),
    State('alpha', 'value'),
    State('embedding-method-options', 'value'),
    prevent_initial_call = False
)
def update_output(n_clicks, n_clicks2, n_clicks3,
    clustering_index, clustering_method,
    embedding_method, model, lr, dim, ws, minCount, minn, maxn, neg, wordNgrams, loss,  # fasttext
    vocab_min_count, vector_size, max_iter, window_size, x_max, # glove
    attention_probs_dropout_prob, # bert
    num_train_epochs, learning_rate, train_batch_size, warmup_steps, max_seq_length, # roberta, gpt2, scibert,
    vector_size_doc2vec, window_doc2vec, # doc2vec
    vector_size_word2vec, window_word2vec, min_count, hs, alpha, # word2vec
    embedding_method_option):
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    cluster_counts = []
    if(clustering_index in ['Rand Index', 'Homogeneity Score']):
        cluster_counts = [scoreCalculator.get_annotation_cluster_number()]
    elif clustering_method == 'dbscan_scores'and clustering_index not in ['Rand Index', 'Homogeneity Score']:
        cluster_counts = ["Not attendant"]
    else :
        cluster_counts =  [5, 10, 15, 20]
    
    columns = get_columns_of_recap_table(cluster_counts)
    if(context == 'simulate1'):
        res = runMethod(embedding_method, embedding_method_option)
        best_data = scoreCalculator.get_scores(clustering_index, clustering_method)
        return [True, True, best_data, columns]
    elif(context == 'batch_simulate'):
        for i in range(5):
            runMethod(embedding_method, embedding_method_option)
        best_data = scoreCalculator.get_scores(clustering_index, clustering_method)
        return [True, True, best_data, columns]
    elif(context == 'simulate2'):
        params = None
        if embedding_method == 'fasttext':
            print("model:", model)
            runMethod(embedding_method, embedding_method_option, {
                'model': model,
                'lr': lr,
                'dim': dim,
                'ws': ws,
                'minCount': minCount,
                'minn': minn,
                'maxn': maxn,
                'neg': neg,
                'wordNgrams': wordNgrams,
                'loss': loss
            })
        elif embedding_method == 'glove':
            runMethod(embedding_method, embedding_method_option, {
                'vocab_min_count': vocab_min_count,
                'vector_size': vector_size,
                'max_iter': max_iter,
                'window_size': window_size,
                'x_max': x_max
            })
        elif embedding_method == 'bert':
            runMethod(embedding_method, embedding_method_option, {
                'attention_probs_dropout_prob': attention_probs_dropout_prob
            })
        elif embedding_method == 'roberta' or embedding_method == 'gpt2' or embedding_method == 'scibert':
            runMethod(embedding_method, embedding_method_option, {'num_train_epochs': num_train_epochs,
                                             'learning_rate': learning_rate,
                                             'train_batch_size': train_batch_size,
                                             'warmup_steps': warmup_steps,
                                             'max_seq_length': max_seq_length})
        elif embedding_method == 'doc2vec':
            runMethod(embedding_method, embedding_method_option, {'vector_size': vector_size_doc2vec,
                                             'window': window_doc2vec})
        elif embedding_method == 'word2vec':
            runMethod(embedding_method, embedding_method_option, {'vector_size': vector_size_word2vec,
                                            'window': window_word2vec, 'min_count': min_count,
                                            'hs': hs, 'alpha': alpha })
        else:
            raise Exception("No such method exists")

        best_data = []#get_scores(clustering_index, clustering_method)
        return [True, True, best_data, columns]
    else:
        data = scoreCalculator.get_scores(clustering_index, clustering_method)
        print("data: ", data)
        return [False, False, data, columns]

@app.callback(
    Output('wordcount-graph', 'figure'),
    Input('input_range', 'value'),
    prevent_initial_call=False)
def update_histogram(input_value):
    start = 0
    end = 50
    if(input_value):
        start = (input_value-1)*50
        end = input_value*50
    data = go.Bar(
        x = df[start:end]['word'],
        y = df[start:end]['count'],
        marker = {'color' : corporate_colors['light-green'], 'opacity' : 0.75})
    return go.Figure(data=data, layout=corporate_layout)

@app.callback(
    [Output("modal-body", "children"), Output("modal-header", "children"), Output("modal", "is_open")],
    [Input("wordcount-graph", "clickData"), Input("close", "n_clicks"), Input('input_range', 'value')], 
    prevent_initial_call = True
)
def toggle_modal(clickData, close, page_value):
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if context == 'wordcount-graph':
        word_list = tokenizer.get_word_list()
        token_papers = tokenizer.get_token_papers()

        index = clickData["points"][0]["pointIndex"]
        word = word_list[index + (page_value-1)*50]

        token_papers_str = []
        modal_per_paper = []
        for token_paper in token_papers[word]:
            token_papers_str.append(html.Li(token_paper, style={'cursor': 'pointer'}, id={'type': 'paper-item', 'index': token_paper}))
        return [token_papers_str, word_list[index + (page_value-1)*50] + ": " + str(len(token_papers_str)), True]

    if context == 'close':
        return  ["", "",False]
    return ["","",False]

def cosine_similarity(list1, list2):
    return np.dot(list1, list2)/(np.linalg.norm(list1)*np.linalg.norm(list2))


def get_neighbor_papers(paper_name):
    papers = get_embeddings('fasttext', 'avg')
    the_embedding = papers[paper_name]
    top_ten_list = []
    top_ten_sim_list = []
    for i in range(10):
        top_ten_sim_list.append(0)
        top_ten_list.append('')
    for paper in papers:
        _embedding = papers[paper]
        sim = cosine_similarity(the_embedding, _embedding)
        for i in range(10):
            if(top_ten_sim_list[i] < sim):
                top_ten_sim_list.insert(i, sim)
                top_ten_sim_list.pop(10)
                top_ten_list.insert(i, paper)
                top_ten_list.pop(10)
                break
    print(top_ten_list)
    return top_ten_list


@app.callback(
    [Output("dynamic-list", "children")],
    [Input({'type': 'paper-item', 'index': ALL}, 'n_clicks')], 
    prevent_initial_call = True
)
def toggle_third_modal(paper):
    if paper is None:
        raise PreventUpdate
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    context = json.loads(context)
    if(context != None and context["type"] == "paper-item"):
        neighbor_papers = get_neighbor_papers(context["index"])
        return [create_neighboring_model(context["index"], neighbor_papers)]
    else :
        return PreventUpdate



draft_template = go.layout.Template()
draft_template.layout = corporate_layout

show = {'display': 'inline-block'}
hide = {'display': 'none'}

@app.callback(  
    Output('fasttext_input_boxes', 'style'),
    Output('fasttext_dropdowns', 'style'),
    Output('glove_input_boxes', 'style'),
    Output('bert_input_boxes', 'style'),
    Output('roberta_gpt2_input_boxes', 'style'),
    Output('doc2vec_input_boxes', 'style'),
    Output('word2vec_input_boxes', 'style'),
    Output('word2vec_dropdowns', 'style'),
     Output('sentence_bert_input_boxes', 'style'),
    Output('sentence_bert_dropdowns', 'style'),
    Input('embedding-method', 'value'),
    prevent_initial_call = False)
def getParamSelectionDiv(embedding_method):
    if(embedding_method == "fasttext"):
        return [show, show, hide, hide, hide, hide, hide, hide, hide, hide]
    elif(embedding_method == "glove"):
        return [hide, hide, show, hide, hide, hide, hide, hide, hide, hide]
    elif(embedding_method == "bert"):
        return [hide, hide, hide, show, hide, hide, hide, hide, hide, hide]
    elif(embedding_method == "gpt2" or embedding_method == "roberta" or embedding_method == 'scibert'):
        return [hide, hide, hide, hide, show, hide, hide, hide, hide, hide]
    elif(embedding_method == "doc2vec"):
        return [hide, hide, hide, hide, hide, show, hide, hide, hide, hide]
    elif(embedding_method == "word2vec"):
        return [hide, hide, hide, hide, hide, hide, show, show, hide, hide]
    elif(embedding_method == "sentence bert"):
        return [hide, hide, hide, hide, hide, hide, hide, hide, show, show]
    else:
        return [hide, hide, hide, hide, hide, hide, hide, hide, hide, hide]

@app.callback(  
    Output('kmeans_input_boxes', 'style'),
    Output('kmeans_dropdowns', 'style'),
    Output('agglomerative_dropdowns', 'style'),
    Input('clustering-method', 'value'),
    prevent_initial_call = False)
def getParamSelectionDiv(clustering_method):
    if(clustering_method == "hierarthical"):
        return [hide, hide, show]
    elif(clustering_method == "kmeans"):
        return [show, show, hide]
    else:
        return [hide, hide, hide]

@app.callback(
    Output('embedding-graph', 'figure'),
    Input('embedding-method2', 'value'),
    Input('embedding-method-options2', 'value'),
    Input('vis-method', 'value'),
    Input('cluster_count', 'value'),
    Input('clustering-method2', 'value'),
    Input('clustering-index2', 'value'),
    Input('perplexity', 'value'),
    State('embedding-graph', 'figure'),
    prevent_initial_call = False)
def update_scatter_graph(embedding_method, embedding_method_option, pcaOrTsne, cluster_count, clustering_method, clustering_index,perplexity, old_graph):
    
    embeddings_dict = get_embeddings(embedding_method, embedding_method_option, index_value_key[clustering_index], clustering_method)
    if(embeddings_dict != {}):
        embedding_keys = list(embeddings_dict.keys())
        embeddings = list(embeddings_dict.values())

        df = pd.DataFrame.from_dict(embeddings_dict, orient='index')
        suitable_params_ = scoreCalculator.get_optimized_scores(embedding_method, embedding_method_option, index_value_key[clustering_index], clustering_method)

        if(type(suitable_params_) == list):
            suitable_params = suitable_params_[cluster_count-2]
        else:
            suitable_params = suitable_params_

        clustering_res = None
        p = {}
        if(clustering_method == 'hierarthical_scores'):
            p['affinity'] = affinity_vals[suitable_params['affinity'][0]]
            p['linkage'] = linkage_vals[suitable_params['linkage'][0]]
            print("pppppp: ", p)
            clustering_res = AgglomerativeClustering(n_clusters=cluster_count).fit(embeddings)
        elif clustering_method == 'kmeans_scores':
            p['init'] = init_vals[suitable_params['init'][0]]
            p['algorithm'] = algorithm_vals[suitable_params['algorithm'][0]]
            p['n_init'] = int(suitable_params['n_init'][0])
            p['max_iter'] = int(suitable_params['max_iter'][0])
            clustering_res = KMeans(n_clusters=cluster_count, **p).fit(embeddings)
        elif clustering_method == 'dbscan_scores':
            if suitable_params != None:
                if('eps' in suitable_params):
                    p['eps'] = suitable_params['eps'][0]
                if 'min_samples' in suitable_params:
                    p['min_samples'] = suitable_params['min_samples'][0]
            clustering_res = DBSCAN(**p).fit(embeddings)
        else:
            raise Exception("No such clustering method")

        labels = clustering_res.labels_.tolist()

        df.columns = ["embeddings" + str(i) for i in range(len(embeddings[0]))]
        arr = []
        if(pcaOrTsne == "pca"):
            arr = PCA(n_components=2).fit_transform(df)
        elif(pcaOrTsne == "tsne"):
            arr = TSNE(perplexity=perplexity).fit_transform(df)
        data_lst = []
        i = 0
        for el in arr:
            data_as_dict = {}
            data_as_dict["x"] = el[0]
            data_as_dict["y"] = el[1]
            # veriyi discrete gösterebilmek için
            data_as_dict["cluster"] = str(labels[i])
            data_as_dict["paper_name"] = embedding_keys[i]
            i += 1
            data_lst.append(data_as_dict)
        fig = px.scatter(pd.DataFrame(data_lst), x='x', y='y', color = 'cluster', hover_data=["paper_name"], opacity=0.75, template=draft_template)
        return fig
    else:
        return px.scatter()
    
def get_summary_around_word(word, paper_name, data):
    result_lst = []
    all_occurences = [m.start() for m in re.finditer(word, data)]
    for position in all_occurences:
        summary =  "..." + data[max(0, position - 200):position+200] + "..."
        result_lst.append(html.Li(summary))
    return result_lst

def init_filepaths():
    path = initial_files_path + '/documents'
    pwd = os.getcwd()
    os.chdir(path)
    filepaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.txt'];
    os.chdir(pwd)
    return filepaths

@app.callback(
    [Output("dynamic-list2", "children")],
    [State("my-dynamic-dropdown", "value")],
    [Input({'type': 'paper-item2', 'index': ALL}, 'n_clicks')],
    prevent_initial_call = True
)
def toggle_third_modal(word, _):
    if word is None or _ is None:
        raise PreventUpdate
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    context = json.loads(context)
    filepaths = init_filepaths()
    file_data = None
    for filename in filepaths:
        with open(filename, "r") as fd:
            paper_name = fd.readline().rstrip()
            if context["index"] == paper_name:
                file_data = fd.read()
    if(context != None and context["type"] == "paper-item2"):
        summary_items = get_summary_around_word(word, context["index"], file_data)
        return [create_neighboring_model(word, summary_items)]
    else:
        raise PreventUpdate



@app.callback(
    Output("my-dynamic-dropdown", "options"),
    Input("my-dynamic-dropdown", "search_value"),
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    return [o for o in word_dict if search_value in o["label"]]

def create_neighboring_model(paper_name, neighbor_papers):
    neighbor_paper_items = []
    for neighbor_paper in neighbor_papers:
        neighbor_paper_items.append(html.Li(neighbor_paper))

    return dbc.Modal([
            dbc.ModalHeader(id = 'modal-header3', style={'backgroundColor':corporate_colors['dark-green']}, children=html.Div(paper_name)),
            dbc.ModalBody(id =  'modal-body3', style={'backgroundColor': corporate_colors['light-grey']}, children=neighbor_paper_items),
            dbc.ModalFooter(
                dbc.Button("Close", id= "close3", className="ml-auto")
                , style={'backgroundColor': corporate_colors['dark-green']}
            )
        ],
        is_open = True,
        id={'type': "modal3", 'index': paper_name},
        scrollable=True
    )

@app.callback(
    Output("modal-body2", "children"), 
    Output("modal-header2", "children"), 
    Output("modal2", "is_open"),
    Input("my-dynamic-dropdown", "value"),
    prevent_initial_call = True
)
def update_options(word):
    token_papers = tokenizer.get_token_papers_with_bigram_words()
    if(word):
        token_papers_str = []
        for token_paper in token_papers[word]:
            token_papers_str.append(html.Li(id={'type': 'paper-item2', 'index': token_paper}, 
                                                children=token_paper))
        return [token_papers_str, word + ": " + str(len(token_papers_str)), True]
    return ["","", False, ""]

def get_method_options_by_method(embedding_method):
    return [{'label': method_option_desc[key], 'value': key} for key in available_methods[embedding_method]]

@app.callback(
    Output('embedding-method-options', 'options'),
    Input('embedding-method', 'value'),
    prevent_initial_call=False
)
def update_emb_met_opts(embedding_method):
    return get_method_options_by_method(embedding_method)


@app.callback(
    Output('embedding-method-options2', 'options'),
    Input('embedding-method2', 'value'),
    prevent_initial_call=False
)
def update_emb_met_opts(embedding_method):
    return get_method_options_by_method(embedding_method)


def calculate_correlation_coefficients(scores_dict):
    correlation_coeffs_lst = []
    for key1, key2 in combinations(scores_dict, 2):
        print("key1: ", key1, "key2: ", key2)
        scores1 = scores_dict[key1]
        print("scores1: ", scores1)
        scores2 = scores_dict[key2]
        print("scores2: ", scores2)
        correlation_coeffs_lst.append({'embedding method': str(key1) + " vs. " + str(key2), 'coeff': pearsonr(scores1, scores2)[0]})
    print(correlation_coeffs_lst)
    return correlation_coeffs_lst


kmeans_params = {
    'init': {
        'id': 'init', 
        'name': "Init mode", 
        'info': "Method for initialization", 
        'values': ['k-means++', 'random'], 
        'defaultValue': 'random'
    }, 'algorithm': {
        'id': 'algorithm', 
        'name': "algorithm", 
        'info': "Centering the data or not", 
        'values': ['auto', 'full', 'elkan'], 
        'defaultValue': 'auto'
    }, 'n_init': {
        'id': 'n_init',
        'name': 'Init points number',
        'info': 'Number of time the k-means algorithm will be run with different centroid seeds',
        '_min': 10,
        '_max': 50,
        'step': 2,
        'defaultValue': 10
    }, 'max_iterint': {
        'id': 'max_iterint',
        'name': 'Max number of iterations',
        'info': 'Maximum number of iterations',
        '_min': 100,
        '_max': 1000,
        'step': 100,
        'defaultValue': 300
    }
}

agglomerative_params = {
    'affinity': {
        'id': 'affinity', 
        'name': "affinity", 
        'info': "Metric used to compute the linkage", 
        'values': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], 
        'defaultValue': 'random'
    }, 'linkage': {
        'id': 'linkage', 
        'name': "linkage", 
        'info': "Metric used to compute the linkage", 
        'values': ['ward', 'complete', 'average', 'single'], 
        'defaultValue': 'random'
    }
}



@app.callback(
    Output('affinity', 'value'),
    Input('linkage', 'value'),
    State('affinity', 'value'),
    prevent_initial_call=False
)
def ward_method_works_only_with_euclidian_dist(linkage, prev_state):
    if linkage == 'ward':
        return 'euclidean'
    else:
        return prev_state



@app.callback(
    Output('best-embedding-results', 'figure'),
    Output('correlation-coef-of-embeddings-graph', 'figure'),
    Input('clustering-method', 'value'),
    Input('clustering-index', 'value'),
    prevent_initial_call=False
)
def update_clustering_graph(clustering_method, clustering_index):
    (data, scores_dict) = scoreCalculator.get_scores_for_graph(clustering_index, clustering_method)
    correlation_coefs = calculate_correlation_coefficients(scores_dict)
    df = pd.DataFrame(data)
    coeff_graph = px.line(template=draft_template)
    clustering_metrics_graph = None
    if(len(scores_dict)>0):
        print("correlation_coefs: ", correlation_coefs)
        # coeff_graph = px.line(pd.DataFrame(correlation_coefs), x = "embedding method", y = "coeff", template=draft_template)
        clustering_metrics_graph = px.line(df, x='cluster counts', y='scores', color='embedding method', template=draft_template)
    else:
        clustering_metrics_graph = px.scatter(df, x='cluster counts', y='scores', hover_data=["embedding method"], opacity=0.75, template=draft_template)
        # coeff_graph = px.line(template=draft_template)
    return [clustering_metrics_graph, coeff_graph]

@app.callback(
    Output('loading-4', 'figure'),
    Input('upload-files', 'contents'),
    prevent_initial_call=True
)
def upload_text_files(content):
    fileUploader = FileUploader()
    fileUploader.upload_file(content)
    fileUploader.initialize_initial_files_to_train()
    return True;

@app.callback(
    Output('loading-5', 'figure'),
    Output('doc_not_exists_alert', 'is_open'),
    Output('doc_not_exists_alert', 'children'),
    Output('doc_not_exists_alert', 'color'),
    Input('upload-annotation-file', 'contents'),
    prevent_initial_call=True
)
def upload_annotation_file(content):
    try:
        FileUploader().upload_annotation_file(content)
    except Exception as err:
        alert_message = html.Div([str(err)])
        return [True, True, alert_message, 'danger']
    return [True, True, 'Documents uploaded', 'info']

@app.callback(
    Output('loading-3', 'children'),
    Output('recalculation_result', 'children'),
    Input('recalculate_clusters', 'n_clicks')  
)
def re_calculate_scores(recalculate_clusters):
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if(context == 'recalculate_clusters'):
        scoreCalculator.recalculate_scores()
        return [True,html.Div('Recalculation finished')]
    return [False, html.Div('')]
