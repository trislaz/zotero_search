import sys
import numpy as np
import openai

class EmbeddingMatcher:
    def __init__(self, embeddings_path, titles_path):
        self.embeddings_path = embeddings_path
        self.titles_path = titles_path

    def embed(self, prompt):
        e = openai.Embedding.create(input=prompt, model="text-embedding-ada-002")
        return np.stack(e['data'][0]['embedding'])

    def find_top_matches(self, prompt, N):
        e_prompt = self.embed(prompt)
        E_bibli = np.load(self.embeddings_path)
        titles = np.load(self.titles_path)

        scores = np.dot(E_bibli, e_prompt) / (np.linalg.norm(E_bibli, axis=1) * np.linalg.norm(e_prompt))
        sorted_idx = np.argsort(scores)[::-1]

        return [(titles[idx], scores[idx]) for idx in sorted_idx[:N]]

    def print_top_matches(self, prompt, N):
        matches = self.find_top_matches(prompt, N)
        for o, (title, score) in enumerate(matches):
           print(f'{o}. {title} - Score: {score}\n')

if __name__ == '__main__':
    matcher = EmbeddingMatcher('assets/embeddings.npy', 'assets/titles.npy')
    matcher.print_top_matches(sys.argv[1], 5)

