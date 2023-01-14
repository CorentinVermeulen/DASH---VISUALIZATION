import dash
from dash import Output, Input, dcc, html, dash_table

import plotly.express as px

import numpy as np
import pandas as pd
import math
from PIL import Image

import sklearn.decomposition as skd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

### DATA ----------------------
data = pd.read_csv('files/data/transcriptomics_data.csv')

X = data.iloc[:, :-2]
y = data['cell_type']
n = len(data)
genes = list(data.columns[:-2])
colors = {}
# -------------------------------

###### NOT REACTIVES COMPONENTS  #########

### PCA -------------------------
pca = skd.PCA(n_components=3)
U = pca.fit_transform(X)
# -------------------------------

# CORR MATRIX ------------------
corr = X.corr()
fig_corr = px.imshow(corr)
# ------------------------------

# CORR STACK
corr_stack = pd.DataFrame(corr.stack())  # Stack, it will be easier to use later for corr filtering
corr_stack = corr_stack.reset_index(level=[0, 1]).rename(columns={'level_0': 'G1', 'level_1': 'G2', 0: 'corr'})
corr_stack['abs_corr'] = np.abs(corr_stack['corr'])
corr_stack = corr_stack.sort_values(by=['G1', 'abs_corr'], ascending=False)
corr_stack = corr_stack[corr_stack['G1'] != corr_stack['G2']]
# ------------------------------

## HIERARCHICAL CLUSTER --------
#linkage_data = np.genfromtxt('files/data/linkage_data.csv', delimiter=',') # See big_computation.py file
#fig_hc = dendrogram(linkage_data, truncate_mode='lastp', no_labels=True)

fig_hc = Image.open('assets/fig_hc.png')

#image_filename = 'files/assets/fig_hc.png'
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#fig_hc = ff.create_dendrogram(X)

# ------------------------------

# t-SNE -----------------------
tsne_data = np.genfromtxt('files/data/tsne_data.csv', delimiter=',')  # See big_computation.py file
# ------------------------------


#### USER UI ###################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    dcc.Tabs([

        ### TAB 1
        dcc.Tab(label='Basic exploration', children=[
            html.H1('Basic data exploration'),
            ### TAB1 SCATTER PLOT
            html.H4("Scatter plot and marginal distribution"),
            html.Div([
                html.Div([
                    dcc.Graph(id='tab1_fig'),

            ### TAB1 BOXPLOTS
                    html.H4('Distribution of the value of genes'),
                    html.Div([
                        html.Label('Select genes to boxplot'),
                        dcc.Dropdown(genes,
                                     value=['gene_37'],
                                     id='tab1_box_xvar',
                                     multi=True,
                                     searchable=True,
                                     search_value='',
                                     placeholder="Select Genes"
                                     ),
                        dcc.Graph(id='tab1_boxplot')
                    ]),
                ], className="ten columns"),
                html.Div(children=[
                    html.Label('X variable'),
                    dcc.Dropdown(options=list(data.columns)[:-1],
                                 value='cell_type',
                                 id='tab1_x',
                                 searchable=True,
                                 search_value='',
                                 placeholder="Select X",
                                 clearable=False
                                 ),
                    html.Label('Y variable'),
                    dcc.Dropdown(options=genes,
                                 value='gene_166',
                                 id='tab1_y',
                                 searchable=True,
                                 search_value='',
                                 placeholder="Select Y",
                                 clearable=False
                                 ),
                    html.Label('Color'),
                    dcc.Dropdown(options=['all'] + list(np.unique(data["cell_type"])),
                                 value='all',
                                 id='tab1_col',
                                 multi=True,
                                 searchable=True,
                                 search_value='',
                                 placeholder="Select Color"
                                 ),
                    html.Br(),
                    html.H4('Summary statistics'),
                    html.Div(id='tab1_stat_table')
                ], className="two columns")

            ], className="row"),

            ### TAB1 CORRELATION BETWEEN GENES
            html.Div([
                html.Div([
                    html.H4('Correlation Matrix'),
                    dcc.Graph(id='fig_mean', figure=fig_corr),
                ], className='six columns'),
                html.Div([
                    html.H4("List with most correlated genes for selected gene"),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(options=['all'] + genes,
                                     value='all',
                                     id='tab1_gene_corr',
                                     multi=False,
                                     searchable=True,
                                     search_value='',
                                     placeholder="Select Gene to see most correlated ones",
                                     clearable=False
                                     )
                        ], className='three column'),
                        html.Div([
                            dcc.Dropdown(options= genes,
                                         value=None,
                                         id='tab1_gene_corr_2',
                                         multi=False,
                                         searchable=True,
                                         search_value='',
                                         placeholder="Select Second Gene to see correlation",
                                         clearable=True
                                         )
                        ], className='three column')
                   ],className='row'),
                    html.Div(id='tab1_gene_tab')
                ], className='six columns')
            ], className='row')
        ]),

        ### TAB 2
        dcc.Tab(label='Clustering', children=[

            ## TAB 2 HIERARCHICAL AND KMEANS CLUSTERING
            html.H1('Clustering'),
            html.Div([
                html.Div([
                    html.H2('Hierachical clustering'),
                    html.Label('Dendrogram'),
                    #dcc.Graph(id='fig_hc', figure=fig_hc)
                    html.Div(html.Img(src=app.get_asset_url('fig_hc.png'), style={'width':'100%'}))

                ], className='four columns'),
                html.Div([
                    html.H2('K-Means clustering'),
                    html.Label('Slider for number of cluster in Kmeans'),
                    dcc.Slider(id='slider_kmeans',
                               min=1,
                               max=128,
                               value=3,
                               tooltip={"placement": "bottom", "always_visible": True}),
                    dcc.Graph(id='tab2_kmeans')
                ], className='eight columns')
            ], className='row'),

            ## DENSITY BASED CLUSTERING
            html.Div([
                html.H3('Density based clustering: K-nearest neighbors'),

                html.Label('Input for K'),
                dcc.Input(id='input_knn',
                          type='number',
                          placeholder='Enter value for k',
                          value=round(math.sqrt(n))),
                dcc.Graph(id='tab2_knn')
            ])
        ]),

        ### TAB 3
        dcc.Tab(label='Dimension Reduction', children=[
            html.H1('Dimension Reduction Visualisation '),
            html.H6('Select data in one plot to see them in the other plot'),
            ## TAB 3 PCA
            html.Div([
                html.H3('Linear dimension reduction'),
                html.H4('PCA'),
                html.Div([
                    html.Label("Select color for points"),
                    dcc.Dropdown(options=list(data.columns)[:-1],
                                 value='cell_type',
                                 id='tab3_pca_dd',
                                 searchable=True,
                                 search_value='',
                                 placeholder="Select Color",
                                 clearable=False
                                 ),
                    html.Br(),
                    dcc.Graph(id='fig_pca')
                ])
            ]),

            ## TAB 3 T-SNE
            html.Div([
                html.H3('Non-linear dimension reduction'),
                html.H4('t-SNE'),
                html.Div([
                    html.Label("Select color for points"),
                    dcc.Dropdown(options=list(data.columns)[:-1],
                                 value='cell_type',
                                 id='tab3_tsne_dd',
                                 searchable=True,
                                 search_value='',
                                 placeholder="Select Color",
                                 clearable=False
                                 ),
                    html.Br(),
                    dcc.Graph(id='tab3_tsne')
                ])
            ])
        ])
    ])

])


@app.callback(
    Output('tab1_fig', 'figure'),
    Input('tab1_x', 'value'),
    Input('tab1_y', 'value'),
    Input('tab1_col', 'value')
)
def update_graph(x, y, col):
    if 'all' not in col:
        data_filtered = data.loc[data["cell_type"].isin(col)]
        data_filtered["cell_type_cat"] = data_filtered["cell_type"].astype('category')
        n = len(data_filtered)
        fig_r = px.scatter(data_filtered,
                           x=x, y=y, color='cell_type_cat',
                           marginal_x='box', marginal_y='box',
                           title=f'(n={n:_})')
        fig_r.update_layout(height=500)

    elif 'all' in col or col is None:
        n = len(data)
        fig_r = px.scatter(data, x=x, y=y, color='cell_type',
                           marginal_x='box', marginal_y='box',
                           title=f'(n={n:_})')
        fig_r.update_layout(height=500)
    return fig_r


@app.callback(
    Output('tab1_stat_table', 'children'),
    Input('tab1_x', 'value'),
    Input('tab1_y', 'value'),
    Input('tab1_col', 'value')
)
def update_table(x, y, col):
    if 'all' not in col:
        data_filtered = data.loc[data["cell_type"].isin(col)]
        data_filtered['cell_type_gp'] = data_filtered['cell_type'].astype('str')
        df = data_filtered[[x, y, 'cell_type_gp']]

        df_stat = pd.DataFrame({'N Obs.': df['cell_type_gp'].value_counts(),
                                'Mean x': df.groupby(['cell_type_gp']).mean().iloc[:, 0],
                                'Mean y': df.groupby(['cell_type_gp']).mean().iloc[:, 1],
                                'Median x': df.groupby(['cell_type_gp']).median().iloc[:, 0],
                                'Median y': df.groupby(['cell_type_gp']).median().iloc[:, 0],
                                'Min x': df.groupby(['cell_type_gp']).min().iloc[:, 0],
                                'Min y': df.groupby(['cell_type_gp']).min().iloc[:, 1],
                                'Max x': df.groupby(['cell_type_gp']).max().iloc[:, 0],
                                'Max y': df.groupby(['cell_type_gp']).max().iloc[:, 1],
                                'Std. x': df.groupby(['cell_type_gp']).std().iloc[:, 0],
                                'Std. y': df.groupby(['cell_type_gp']).std().iloc[:, 1],
                                })

    else:
        df = data[[x, y]]
        df_stat = pd.DataFrame({'N Obs.': [len(df[x])],
                                f'Mean x': [df[x].mean()],
                                f'Mean y': [df[y].mean()],
                                f'Median x': [df[x].median()],
                                f'Median y': [df[y].median()],
                                f'Min x': [df[x].min()],
                                f'Min y': [df[y].min()],
                                f'Max x': [df[x].max()],
                                f'Max y': [df[y].max()],
                                f'Std. x': [df[x].std()],
                                f'Std. y': [df[y].std()],
                                }, index=['all'])

    dff = df_stat.transpose().round(3)
    dff.insert(0, 'Stat', dff.index)

    res = dash_table.DataTable(
        id="table",
        columns=[{"id": name, "name": name} for name in dff.columns.values],
        data=dff.to_dict("rows")
    )
    return res


@app.callback(
    Output('tab1_boxplot', 'figure'),
    Input('tab1_box_xvar', 'value')
)
def update_boxplots(xvars):
    title = "Distribution of the value of genes"
    labels = {'x': 'Genes value', 'y': 'Gene'}

    fig = px.box(data, x=xvars, labels=labels)
    return fig


@app.callback(
    Output('tab1_gene_tab', 'children'),
    Input('tab1_gene_corr', 'value'),
    Input('tab1_gene_corr_2', 'value')
)
def update_corr_table(gene_to_corr, gene_to_corr2):
    print(gene_to_corr2)
    if gene_to_corr2 is None:
        if 'all' != gene_to_corr:
            df = corr_stack.loc[corr_stack["G1"] == gene_to_corr]
            dff = df.iloc[:100, 1:3]

        else:
            df = corr_stack.sort_values(['abs_corr'], ascending=False)
            dff = df.iloc[:100, :3]
    else:
        if 'all' != gene_to_corr:
            df = corr_stack.loc[corr_stack["G1"] == gene_to_corr]
            df = df.loc[df["G2"] == gene_to_corr2]
            dff = df.iloc[:,:3]

        else:
            df = corr_stack.loc[corr_stack["G2"] == gene_to_corr2].sort_values(['abs_corr'], ascending=False)
            df = df.sort_values(['abs_corr'], ascending=False)
            dff = df.iloc[:100, :3]


    res = dash_table.DataTable(
        id="corr_table",
        columns=[{"id": name, "name": name} for name in dff.columns.values],
        data=dff.to_dict("rows"),
        page_action='none',
        style_table={'height': '400px', 'overflowY': 'auto'}
    )
    return res


@app.callback(
    Output('tab2_kmeans', 'figure'),
    Input('slider_kmeans', 'value')
)
def update_kmeans(k):
    res = KMeans(n_clusters=k, random_state=0).fit(X)
    labs = res.labels_
    U = pca.fit_transform(X)
    fig = px.scatter(None, x=U[:, 0], y=U[:, 1],
                     color=labs,
                     labels={'sort': True, 'x': 'X axis of PCA', 'y': 'Y axis of PCA', 'color': 'Clusters'},
                     title='K-means plotted on PCA first components')
    fig.update_layout(height=700)
    return fig


@app.callback(
    Output('tab2_knn', 'figure'),
    Input('input_knn', 'value')
)
def update_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)

    fig_knn = px.scatter(None, x=U[:, 0], y=U[:, 1],
                         color=y_pred,
                         title='KNN prediction on PCA plot',
                         labels={
                             'x': 'First PCA component',
                             'y': 'Second PCA component',
                             'color': 'KNN label'
                         })
    fig_knn.update_layout(height=700)
    return fig_knn


"""@app.callback(
    Output('fig_pca', 'figure'),
    Input('tab3_pca_dd', 'value')
)
def update_pca(color):
    fig_pca = px.scatter(None, x=U[:, 0],
                         y=U[:, 1],
                         color=data[color],
                         title='2D PCA plot ({0} %) variance explained'.format(
                             round(100 * np.sum(pca.explained_variance_ratio_[:-1]), 2)),
                         labels={
                             'x': 'First component',
                             'y': 'Second component',
                             'color': color
                         })

    fig_pca.update_layout(height=700)
    return fig_pca"""

"""@app.callback(
    Output('tab3_tsne', 'figure'),
    Input('tab3_tsne_dd', 'value')
)
def update_pca(color):
    fig_tsne = px.scatter(None, x=tsne_data[:, 0],
                          y=tsne_data[:, 1],
                          color=data[color],
                          title='t-SNE representation',
                          labels={
                              'x': 'First component',
                              'y': 'Second component',
                              'color': color
                          })
    fig_tsne.update_layout(height=700)
    return fig_tsne"""


def get_figure(df, x_col, y_col, color, selectedpoints, selectedpoints_local, title):
    if selectedpoints_local and selectedpoints_local['range']:
        ranges = selectedpoints_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
    else:
        selection_bounds = {'x0': np.min(df[x_col]), 'x1': np.max(df[x_col]),
                            'y0': np.min(df[y_col]), 'y1': np.max(df[y_col])}

    fig = px.scatter(df, x=df[x_col],
                     y=df[y_col],
                     color=color,
                     title=title,
                     labels={
                         'x': 'First component',
                         'y': 'Second component',
                         'color': color
                     })

    fig.update_traces(selectedpoints=selectedpoints,
                      customdata=df.index,
                      mode='markers', unselected={'marker': {'opacity': 0.1 , 'color': 'rgba(131, 131, 131, 0.19)'}})

    fig.update_layout(dragmode='select', hovermode=False, height=700)

    fig.add_shape(dict({'type': 'rect',
                        'line': {'width': 1, 'dash': 'dot', 'color': 'darkgrey'}},
                       **selection_bounds))
    return fig


df_pca_tsne = pd.DataFrame({'PCA1': U[:, 0],
                            'PCA2': U[:, 1],
                            'TSNE1': tsne_data[:, 0],
                            'TSNE2': tsne_data[:, 1]})
df_pca_tsne = df_pca_tsne.join(data)


@app.callback(
    Output('fig_pca', 'figure'),
    Output('tab3_tsne', 'figure'),
    Input('fig_pca', 'selectedData'),
    Input('tab3_tsne', 'selectedData'),
    Input('tab3_pca_dd', 'value'),
    Input('tab3_tsne_dd', 'value')
)
def update_pca_tsne_cross(selection1, selection2, color_pca, color_tsne):
    selectedpoints = df_pca_tsne.index
    for selected_data in [selection1, selection2]:
        if selected_data and selected_data['points']:
            selectedpoints = np.intersect1d(selectedpoints,
                                            [p['customdata'] for p in selected_data['points']])

    title_pca = '2D PCA plot ({0} %) variance explained'.format(round(100 * np.sum(pca.explained_variance_ratio_[:-1]), 2))
    title_tsne = 't-SNE representation'

    return [get_figure(df_pca_tsne, "PCA1", "PCA2", color_pca, selectedpoints, selection1, title_pca),
            get_figure(df_pca_tsne, "TSNE1", "TSNE2", color_tsne, selectedpoints, selection2, title_tsne)]


app.run_server(debug=False, port=8054) # Debug is on False to avoid alert from cross filtering
