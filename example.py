from XRAG.model.model_manager import model_factory
from XRAG.ingestion.retriever import Retriever, RetrieWithHyPE
from XRAG.ingestion.hype_generator import HyPEGenerator
from ingestion.file_loader import CustomPDFLoader
from ingestion.splitter import  CustomSplitter
from ingestion.vector_store import  MilvusVectorStore
from chatbot.query_engine import  RetrievalQueryEngine, MultiQueryRetrievalEngine
from ingestion.ingestion_pipeline import IngestionPipeline
from llama_index.core.schema import MetadataMode, BaseNode
from typing import List
from eval.rag_eval import RAGEvaluationSystem
import time

def print_nodes(nodes: List[BaseNode]):
    for idx, node in enumerate(nodes):
        print(f'chunk {idx}: ')
        print(node.get_content(MetadataMode.LLM))
        print('-'*20)

if __name__ == '__main__':
    # 1. ingestion
    is_query = True
    is_eval = False
    is_rerank = False
    multi_query_engine = True
    database = 'rag_database_with_info_v1'
    collection_name = 'doc_vector_store_v2'
    chunk_list = [
        (1024, 100)
    ]
    if is_query:
        # start_time = time.time()
        # pipeline = IngestionPipeline(file_path='./data/技术报告.pdf',
        #                              parse_image=True,
        #                              parse_table=True,
        #                              chunk_size=chunk_list[0][0],
        #                              chunk_overlap=chunk_list[0][1],
        #                              database=database,
        #                              collection_name=collection_name)
        # vector_store = pipeline.run()
        # end_time = time.time()
        # print('the time ingestion step cost: {:.4f} s'.format(end_time - start_time))
        # 2. query
        vector_store = MilvusVectorStore(database=database, collection_name=collection_name)
        hype_generator = HyPEGenerator()
        retriever = RetrieWithHyPE(vector_store, hype_generator)
        # query_engine = RetrievalQueryEngine(vector_store)
        if multi_query_engine:
            query_engine = MultiQueryRetrievalEngine(knowledge_database=retriever,
                                                    rerank_model=model_factory.get_rerank_model())
        else:
            query_engine = RetrievalQueryEngine(knowledge_database=retriever,
                                            rerank_model=model_factory.get_rerank_model())
        while True:
            query = input('请输入你的问题: ')
            if query == 'exit':
                break
            print(f'your query is: {query}')
            start_time = time.time()
            resp,_ = query_engine.chat(query, limit=5, rerank_limit=5)
            end_time = time.time()
            print('-'*10 + 'Answer'+ '-'*10)
            print(resp)
            print('-'*20)
            print('-'*10)
            print('the time query step cost: {:.4f} s'.format(end_time - start_time))
            print('-'*10)
    elif is_eval:
        vector_store = MilvusVectorStore(database=database, collection_name=collection_name)
        # query_engine = MultiQueryRetrievalEngine(vector_store,
        #                                         rerank_model=model_factory.get_rerank_model())
        hype_generator = HyPEGenerator()
        retriever = RetrieWithHyPE(vector_store, hype_generator)
        if multi_query_engine:
            query_engine = MultiQueryRetrievalEngine(knowledge_database=retriever,
                                                    rerank_model=model_factory.get_rerank_model())
        else:
            query_engine = RetrievalQueryEngine(knowledge_database=retriever,
                                            rerank_model=model_factory.get_rerank_model())
        rag_eval_system = RAGEvaluationSystem()
        eval_result = rag_eval_system.eval(query_engine, eval_dataset_len=20)
        print('-'*10 + 'Eval' + '-'*10)
        print(eval_result)
        print('-'*20)
    elif is_rerank:
        vector_store = MilvusVectorStore(database=database, collection_name=collection_name)
        query = '请问 EasyRAG 框架是什么样的?'
        nodes = vector_store.get_relevant_nodes(query=query, limit=5)[0]
        print_nodes(nodes)
        rerank_model = model_factory.get_rerank_model()
        new_nodes = rerank_model.rerank(documents=nodes, query=query, top_n=5)
        print('-'*30)
        print_nodes(new_nodes)