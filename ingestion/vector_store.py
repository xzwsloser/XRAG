from importlib.metadata import metadata

from XRAG.model.model_manager import  model_factory
from abc import ABC, abstractmethod
from llama_index.core.schema import  BaseNode
from pymilvus import MilvusClient, DataType
from typing import List, Union
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode

class VectorStore(ABC):
    embedding: OpenAILikeEmbedding | OpenAIEmbedding
    def __init__(self, embedding=None):
        if embedding is None:
            self.embedding = model_factory.get_embedding()
        else:
            self.embedding = embedding

    @abstractmethod
    def store_nodes(self, nodes: List[BaseNode]):
        pass

    @abstractmethod
    def clear_collection(self):
        pass

    @abstractmethod
    def get_relevant_nodes(self, query: Union[str, List[str]], limit: int, filter_expr: str='id is not null') -> List[List[BaseNode]]:
        pass

class MilvusVectorStore(VectorStore):
    client: MilvusClient
    dim: int
    collection_name: str
    def __init__(self, database='rag_base_db', embedding=None,
                 url: str='http://localhost:19530',
                 dim=4096,
                 collection_name='doc_vector_store'):
        super().__init__(embedding)
        self.client = MilvusClient(url)
        self.dim = dim
        self.collection_name = collection_name
        dbs = self.client.list_databases()
        if database == '':
            database = 'rag_base_db'
        if database not in dbs:
            print(f'create db: {database}')
            self.client.create_database(db_name=database)
        self.client.use_database(db_name=database)
        if not self.client.has_collection(collection_name=self.collection_name):
            self._create_collection()

    def _create_collection(self):
        print(f'create collection {self.collection_name}')
        schema = self.client.create_schema(
            auto_id=True
        )

        schema.add_field(
            field_name='id',
            datatype=DataType.INT64,
            is_primary=True
        )

        schema.add_field(
            field_name='doc_id',
            datatype=DataType.VARCHAR,
            max_length=100
        )

        schema.add_field(
            field_name='text',
            datatype=DataType.VARCHAR,
            max_length=65535
        )

        schema.add_field(
            field_name='embedding',
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dim
        )

        schema.add_field(
            field_name='type',
            datatype=DataType.VARCHAR,
            max_length=20,
        )

        schema.add_field(
            field_name='img_path',
            datatype=DataType.VARCHAR,
            max_length=200,
            default_value=''
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name='embedding',
            index_name='embedding_index',
            index_type='HNSW',
            metric_type='COSINE'
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            dimension=self.dim
        )

    def store_nodes(self, nodes: List[BaseNode]):
        print(f'store data to collection: {self.collection_name}')
        # 1. Embedding
        embed_nodes = self.embedding.get_text_embedding_batch([node.text for node in nodes])
        for node, embed_node in zip(nodes, embed_nodes):
            node.embedding = embed_node
        # 2. store
        for idx, node in enumerate(nodes):
            data = [
                {
                    'id': None,
                    'doc_id': node.ref_doc_id,
                    'text': node.text,
                    'embedding': node.embedding,
                    'type': node.metadata['type'],
                    'img_path': node.metadata['img_path']
                }
            ]
            self.client.upsert(collection_name=self.collection_name, data=data)

    def clear_collection(self):
        print(f'delete collection {self.collection_name}')
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

    def get_relevant_nodes(self, query: Union[str, List[str]], limit: int, filter_expr: str = 'id is not null') -> List[List[BaseNode]]:
        if isinstance(query, str):
            query_vector = [self.embedding.get_text_embedding(query)]
        else:
            query_vector = self.embedding.get_text_embedding_batch(query)

        resp = self.client.search(
            collection_name=self.collection_name,
            data=query_vector,
            filter=filter_expr,
            limit=limit,
            output_fields=['id', 'doc_id', 'text', 'type', 'img_path'],
            search_params={'metric_type': 'COSINE'}
        )
        retrieved_node = []
        for sample_nodes in resp:
            cur_sample_nodes = []
            for sample in sample_nodes:
                cur_node = TextNode(
                    text=sample['entity']['text'],
                    metadata={
                        'id': sample['entity']['id'],
                        'doc_id': sample['entity']['doc_id'],
                        'type': sample['entity']['type'],
                        'img_path': sample['entity']['img_path'],
                        'distance': sample['distance']
                    }
                )
                cur_sample_nodes.append(cur_node)
            retrieved_node.append(cur_sample_nodes)

        return retrieved_node