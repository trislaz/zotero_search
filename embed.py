import openai
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os
import time

class EmbeddingsCreator:
    model_emb = "text-embedding-ada-002"
    model_sum = "gpt-3.5-turbo"
    price_per_token = {"text-embedding-ada-002": 0.0001 / 1000,
            "gpt-3.5-turbo_in": 0.0015 / 1000,
            "gpt-3.5-turbo_out": 0.002 / 1000}

    def __init__(self, csv_path, outpath, summarize=False):
        self.csv_path = csv_path
        self.outpath = outpath
        self.summarize = summarize
        if summarize:
            self.summaries_path = os.path.dirname(embeddings_path) + f"/{os.path.basename(csv_path).split('.')[0]}_summaries.npy"

    def get_embedding(self, abstract):
        e = openai.Embedding.create(input=abstract, model=self.model_emb)
        return np.stack(e['data'][0]['embedding'])

    def get_summary(self, abstract):
        system_sum = "Write a summary of the following text such as in this example: \
            'problem: WSI classification overfitts quickly\n solution: Use SSL at the tile level\n \
            results: improves over SoTA on LUAD_LUSC classification by 0.1 AUC point.' Include numerical results. Only answer with the summary."
        messages = [{"role": "system", "content": system_sum}]
        messages.append({"role": "user", "content": abstract})
        e = openai.ChatCompletion.create(
                model=self.model_sum,
                messages=messages,
                temperature=0.0,
                )
        summary = e.choices[0]["message"]["content"]
        return summary

    def clean(self, df):
        df = df.drop_duplicates().set_index('Title')
        df = df[df['Abstract Note'].apply(lambda x: isinstance(x, str))]
        return df

    def process_data(self):
        df = self.clean(pd.read_csv(self.csv_path))
        abstracts = df['Abstract Note'].to_dict()

        E, S = {}, {}
        data = {"embeddings":{},"summaries":{}, "titles":{}, "author":{}, "pdf_link":{}}
        for title, abstract in tqdm(abstracts.items()):
            try:
                data['embeddings'][title] = self.get_embedding(abstract)
                data['summaries'][title] = self.get_summary(abstract) if self.summarize else None
                data['titles'][title] = title
                data['author'][title] = df.loc[title]['Author'].split(';')[0]
                data['pdf_link'][title] = df.loc[title]['File Attachments']
            except Exception as e:
                if "The server is overloaded or not ready yet" in str(e):
                    print('Too many requests, please wait...')
                    time.sleep(60)
                    E[title] = self.get_embedding(abstract)
                    S[title] = self.get_summary(abstract) if self.summarize else title
                    continue
                else:
                    print(f'Error for {title}: {e}')
                    continue

        price_em = sum([self.estimate_price(abstract, task="embedd") for abstract in abstracts.values()])
        price_sum = sum([self.estimate_price(abstract, task="summarize", summary=summary)\
                for abstract, summary in zip(abstracts.values(), S)]) if self.summarize else 0
        print(f'Estimated price for embedding: {price_em} $')
        print(f'Estimated price for summarizing: {price_sum} $')

        np.save(self.outpath, data, allow_pickle=True)

    def estimate_price(self, abstract, task, summary=None):
        """
        In English, 1 word ~ 1.3 tokens (https://gptforwork.com/tools/openai-chatgpt-api-pricing-calculator)
        """
        if task == "embedd":
            model = self.model_emb
            num_tokens = len(abstract.split(" ")) * 1.3
            return self.price_per_token[model] * num_tokens
        elif task == "summarize":
            model = self.model_sum
            num_tokens_in = len(abstract.split(" ")) * 1.3
            num_tokens_out = len(summary.split(" ")) * 1.3
            price_in = self.price_per_token[model + "_in"] * num_tokens_in
            price_out = self.price_per_token[model + "_out"] * num_tokens_out
            return price_in + price_out

    def main(self):
        if Path(self.embeddings_path).exists():
            user_input = input('Embeddings already exist. Do you want to continue this step ? (y/n) ')
            if user_input != 'y':
                sys.exit(0)
        self.process_data()

if __name__ == '__main__':
    creator = EmbeddingsCreator('assets/whole_zotero_library.csv', 'assets/data_tri.npy', summarize=True)
    creator.main()
