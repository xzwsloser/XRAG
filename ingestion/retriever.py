from XRAG.ingestion.vector_store import VectorStore
from XRAG.ingestion.hype_generator import HyPEGenerator
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
from abc import ABC, abstractmethod
from typing import List

class Retriever(ABC):
    vector_store: VectorStore
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @abstractmethod
    def retrieve(self, query: str, limit: int) -> List[BaseNode]:
        pass

class RetrieWithHyPE(Retriever):
    hype_generator: HyPEGenerator
    def __init__(self, vector_store: VectorStore,
                       hype_generator: HyPEGenerator):
        super().__init__(vector_store)
        self.hype_generator = hype_generator

    def retrieve(self, query: str, limit: int=5, use_hype: bool=False) -> List[BaseNode]:
        retrieved_nodes = []
        if use_hype:
            top_chunk = self.vector_store.get_relevant_nodes(query, limit=1)[0][0]
            chunk_str = top_chunk.get_content(MetadataMode.LLM)
            # print('-'*30)
            # print('top chunk: ')
            # print(chunk_str)
            # print('-'*30)

            virtual_doc_text = self.hype_generator.generate(query, chunk_str)
            retrieved_nodes = self.vector_store.get_relevant_nodes(virtual_doc_text, limit=limit)[0]
            virtual_doc_node = TextNode(
                text=virtual_doc_text,
                metadata={
                   'id': 1,
                   'doc_id': 'hype_documents',
                   'type': 'text',
                   'img_path': '',
                   'distance': 0.99
                }
            )
            retrieved_nodes.append(virtual_doc_node)
        else:
            retrieved_nodes = self.vector_store.get_relevant_nodes(query, limit=limit)[0]
        return retrieved_nodes