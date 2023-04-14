import pandas as pd
from sklearn.manifold import TSNE

def visualize_data(data_path, algorithm='tsne'):
    if algorithm == 'tsne':
        tsne = TSNE(n_components=2, random_state=1)
        x, y = read_data(data_path)
        transformed_data = tsne.fit_transform(x)
        dim1, dim2 = [], []
        for t_data in transformed_data:
            dim1.append(float(t_data[0]))
            dim2.append(float(t_data[1]))
        return dim1, dim2

def read_data(data_path):
    df = pd.read_csv(data_path)
    x, y = df.iloc[:-3], df.iloc[:, -1:]
    return x, y
