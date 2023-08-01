import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd

# utils:
def make_text_fittable(text, n_words_per_line=10):
    text = text.split('<br>')
    T = ""
    for t in text:
        add = t.split(' ')
        add = [' '.join(add[i:i+n_words_per_line]) for i in range(0, len(add), n_words_per_line)]
        add = '<br>'.join(add)
        T += add + '<br>'
    return T 

class EmbeddingPlotter:
    def __init__(self, data_path, n_clusters=10):
        self.n_clusters = n_clusters
        self.data_path = data_path
        self.titles = list(np.load(data_path, allow_pickle=True)['embeddings'].keys())
        self.embeddings = np.vstack([np.load(data_path, allow_pickle=True)['embeddings'][title] for title in self.titles])
        self.summaries = [np.load(data_path, allow_pickle=True)['summaries'][title] for title in self.titles]
        self.author = [np.load(data_path, allow_pickle=True)['author'][title] for title in self.titles]
        self.pdf_link = [np.load(data_path, allow_pickle=True)['pdf_link'][title] for title in self.titles]

    def create_umap(self, n_components=2):
        umap = UMAP(n_components=n_components).fit_transform(self.embeddings)
        return umap

    def create_clusters(self):
        kmeans = KMeans(n_clusters=self.n_clusters).fit(self.embeddings)
        return kmeans.labels_

    def plot(self):
        umap = self.create_umap()
        clusters = self.create_clusters()
        titles = self.titles
        summaries = self.summaries
        authors = self.author
        pdf_links = self.pdf_link

        df = pd.DataFrame({
            'x': umap[:, 0],
            'y': umap[:, 1],
            'cluster': clusters,
            'author': authors,
            'pdf_link': pdf_links,
            'title': titles,
            'summary': [self.text_to_html(x) for x in summaries]
        })

        df['hover_data'] = df.apply(lambda row: f"<b>{row['author'], row['title'][:50]}</b>:<br>{row['summary']}", axis=1)

        fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['hover_data'])
        fig.show()

    def text_to_html(self, text):
        text = text.replace('\n', '<br>')
        text = make_text_fittable(text)
        text = text.replace('problem:' , '<b>problem:</b>')
        text = text.replace('solution:' , '<b>solution:</b>')
        text = text.replace('results:' , '<b>results:</b>')
        return text

if __name__ == '__main__':
    plotter = EmbeddingPlotter('assets/data_matth.npy')
    plotter.plot()
