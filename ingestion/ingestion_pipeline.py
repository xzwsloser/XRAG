from XRAG.ingestion.file_loader import CustomPDFLoader
from XRAG.ingestion.splitter import CustomSplitter
from XRAG.ingestion.vector_store import VectorStore, MilvusVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from typing import Union

class IngestionPipeline:
    def __init__(self, file_path: str,
                       parse_image: bool = False,
                       parse_table: bool = False,
                       chunk_size: int = 512,
                       chunk_overlap: int = 100,
                       is_chinese: bool = True,
                       vector_store_type: str = 'milvus',
                       database: str = 'rag_database',
                       embedding: Union[OpenAIEmbedding, OpenAILikeEmbedding] = None,
                       url: str = 'http://localhost:19530',
                       dim: int = 4096,
                       collection_name: str = 'doc_vector_store'):
        self.loader = CustomPDFLoader(
            file_path=file_path,
            parse_image=parse_image,
            parse_table=parse_table
        )

        self.text_splitter = CustomSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_chinese=is_chinese,
        )

        if vector_store_type == 'milvus':
            self.vector_store = MilvusVectorStore(
                database=database,
                embedding=embedding,
                url=url,
                dim=dim,
                collection_name=collection_name
            )
    def run(self) -> VectorStore:
        documents = self.loader.load()
        nodes = self.text_splitter.split_documents(documents)
        self.vector_store.store_nodes(nodes)
        return self.vector_store
