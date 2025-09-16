import requests
from llama_index.core.schema import BaseNode, MetadataMode
from typing import List

class RerankModel:
    def __init__(self, api_key: str,
                       api_base: str,
                       rerank_model_name: str):
        self.api_key = api_key
        self.api_base = api_base + '/rerank'
        self.rerank_model_name = rerank_model_name

    def rerank(self, documents: List[BaseNode],
                     query: str,
                     top_n: int = 5,
                     instruction: str = '请根据问题对于给出的相关文档进行语义相关性排序') -> List[BaseNode]:
        docs = [doc.get_content(MetadataMode.LLM) for doc in documents]
        payload = {
            'model': self.rerank_model_name,
            'query': query,
            'documents': docs,
            'top_n': top_n,
            'instruction': instruction
        }

        headers = {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }

        resp = requests.post(self.api_base, headers=headers, json=payload)

        results = resp.json()['results']
        new_docs = [documents[result['index']] for result in results]

        return new_docs