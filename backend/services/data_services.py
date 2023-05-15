import csv

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from configuration.config import settings
from services.visualization_service import download_results_file

def visualize_data(data_path, n_components=2, algorithm='tsne'):
    dimensions = [[] for _ in range(n_components)]
    transformed_data = None
    x, y = read_data(data_path)
    if algorithm == 'tsne':
        tsne = TSNE(n_components=n_components, random_state=1)
        transformed_data = tsne.fit_transform(x)
    elif algorithm == 'pca':
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(x)
    for t_data in transformed_data:
        for i in range(n_components):
            dimensions[i].append(float(t_data[i]))

    return dimensions

def get_coloring(coloring_result_filepath):
    # coloring_result_filepath = download_results_file(coloring_filename)
    coloring_result_file = open(coloring_result_filepath, 'r')
    coloring_csv_file = csv.DictReader(coloring_result_file)

    coloring_data = {}
    for row in coloring_csv_file:
        regressorName, dataPointNumber = '', ''
        for k, v in row.items():
            if k == '':
                v_split = v.split(settings.HEADER_SEPARATOR)
                regressorName, dataPointNumber = v_split[0], v_split[1][1:]
                continue

            metricName, colorValue = k, v
            if metricName not in coloring_data:
                coloring_data[metricName] = {}
            if regressorName not in coloring_data[metricName]:
                coloring_data[metricName][regressorName] = {}
            coloring_data[metricName][regressorName][dataPointNumber] = float(colorValue)
    return coloring_data

def read_data(data_path):
    df = pd.read_csv(data_path)
    x, y = df.iloc[:-3], df.iloc[:, -1:]
    return x, y
