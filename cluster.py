import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
import plotly.express as px
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
    def __init__(self, embeddings_path, titles_path, summaries_path, n_clusters=10):
        self.embeddings_path = embeddings_path
        self.titles_path = titles_path
        self.n_clusters = n_clusters
        self.summaries_path = summaries_path

    def create_umap(self, n_components=2):
        embeddings = np.load(self.embeddings_path)
        umap = UMAP(n_components=n_components).fit_transform(embeddings)
        return umap

    def create_clusters(self):
        embeddings = np.load(self.embeddings_path)
        kmeans = KMeans(n_clusters=self.n_clusters).fit(embeddings)
        return kmeans.labels_

    def plot(self):
        umap = self.create_umap()
        clusters = self.create_clusters()
        titles = np.load(self.titles_path)
        summaries = np.load(self.summaries_path)

        df = pd.DataFrame({
            'x': umap[:, 0],
            'y': umap[:, 1],
            'cluster': clusters,
            'title': titles,
            'summary': [self.text_to_html(x) for x in summaries]
        })

        df['hover_data'] = df.apply(lambda row: f"<b>{row['title']}</b>:<br>{row['summary']}", axis=1)

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
    plotter = EmbeddingPlotter('assets/embeddings.npy', 'assets/titles.npy', 'assets/summaries.npy')
    plotter.plot()
