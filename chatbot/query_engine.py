from XRAG.model.model_manager import  model_factory
from XRAG.ingestion.retriever import Retriever
from XRAG.model.rerank_model import RerankModel
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import  BaseNode
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
from llama_index.core.prompts import ChatPromptTemplate
from XRAG.prompt.prompt import  (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    MULTI_QUERY_SYSTEM_PROMPT,
    MULTI_QUERY_USER_PROMPT
)
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.schema import MetadataMode
import ast

class QueryEngine(ABC):
    llm: Union[OpenAILike, OpenAI]
    rerank_model: RerankModel
    prompt_template: str
    def __init__(self, llm: Union[OpenAILike, OpenAI] = None,
                       rerank_model: RerankModel = None):
        if llm is None:
            self.llm = model_factory.get_llm()
        else:
            self.llm = llm
        self.rerank_model=rerank_model

    @abstractmethod
    def chat(self, query: str, limit: int) -> Tuple[ChatResponse, List[BaseNode]]:
        pass

class RetrievalQueryEngine(QueryEngine):
    knowledge_database: Retriever
    def __init__(self, knowledge_database: Retriever, llm: Union[OpenAILike, OpenAI] = None,
                        rerank_model: RerankModel = None):
        super().__init__(llm, rerank_model)
        self.knowledge_database = knowledge_database

    def chat(self, query: str, limit: int=5, rerank_limit: int=-1) -> Tuple[ChatResponse, List[BaseNode]]:
        # retrieve
        content_info = ''
        retrieved_nodes = []
        relevant_documents  = []
        if self.knowledge_database is not None:
            retrieved_nodes = self.knowledge_database.retrieve(
                query, limit=limit, use_hype=True
            )

            relevant_documents = retrieved_nodes

            if self.rerank_model is not None:
                if rerank_limit < 0:
                    rerank_limit=limit
                rerank_retrieved_nodes = self.rerank_model.rerank(documents=retrieved_nodes,
                                                                  query=query,
                                                                  top_n=rerank_limit)
                relevant_documents = rerank_retrieved_nodes

            print('-'*30)
            print('relevant docs: ')
            for idx, relevant_node in enumerate(relevant_documents):
                cur_node_info = f'''
                chunk {idx}: 
                {relevant_node.get_content(MetadataMode.LLM)} 
                \n
                '''
                content_info += cur_node_info
                print(relevant_node)
                print('-'*20)
            print('-'*30)
        prompt_template = [
            ChatMessage(role='system', content=RAG_SYSTEM_PROMPT),
            ChatMessage(role='user', content=RAG_USER_PROMPT_TEMPLATE)
        ]
        chat_message_prompt = ChatPromptTemplate(message_templates=prompt_template)
        messages = chat_message_prompt.format_messages(content_str=content_info, query_str=query)

        resp = self.llm.chat(messages)
        return resp, relevant_documents

class MultiQueryRetrievalEngine(QueryEngine):
    knowledge_database: Retriever
    def __init__(self, knowledge_database: Retriever, llm: Union[OpenAILike, OpenAI] = None,
                 rerank_model: RerankModel = None):
        super().__init__(llm, rerank_model)
        self.knowledge_database = knowledge_database

    def chat(self, query: str, limit: int=5, rerank_limit: int =-1, multi_query_number: int=3) -> Tuple[ChatResponse, List[BaseNode]]:
        # 1. get multi query
        multi_query_generate_prompt_template = [
            ChatMessage(role='system', content=MULTI_QUERY_SYSTEM_PROMPT),
            ChatMessage(role='user', content=MULTI_QUERY_USER_PROMPT)
        ]

        multi_query_prompt_template = ChatPromptTemplate.from_messages(multi_query_generate_prompt_template)
        multi_query_generate_prompt = multi_query_prompt_template.format_messages(query_str=query,
                                                                                  query_number=multi_query_number)
        multi_query_resp = self.llm.chat(multi_query_generate_prompt)
        multi_query_list_str = multi_query_resp.message.blocks[0].text
        # print('-'*20)
        # print('multi query: ')
        # print(multi_query_list_str)
        # print('-'*20)
        multi_query = ast.literal_eval(multi_query_list_str)
        # 2. retrieve
        retrieved_documents = []
        # origin query retrieved
        origin_retrieve_documents = self.knowledge_database.retrieve(query, limit=limit, use_hype=True)
        # multi query retrieved
        # multi_retrieve_documents_set = self.knowledge_database.retrieve(multi_query, limit=limit)
        multi_retrieve_documents_set = []
        for cur_query in multi_query:
            documents = self.knowledge_database.retrieve(cur_query, limit=limit)
            multi_retrieve_documents_set.append(documents)
        # map: Node-Id -> Node
        node_id_to_node = {}
        # map: Node-Id -> Number
        f = lambda node: node.metadata['id']
        freq_to_node = {}
        for cur_query_nodes in multi_retrieve_documents_set:
            for cur_node in cur_query_nodes:
                if f(cur_node) not in node_id_to_node:
                    node_id_to_node[f(cur_node)] = cur_node
                if f(cur_node) not in freq_to_node:
                    freq_to_node[f(cur_node)] = 1
                else:
                    freq_to_node[f(cur_node)] += 1

        sorted_freq_nodes = sorted(freq_to_node.items(), key=lambda x: x[1], reverse=True)

        multi_query_final_nodes = []

        for idx in range(limit):
            node_id = sorted_freq_nodes[idx][0]
            multi_query_final_nodes.append(
                node_id_to_node[node_id]
            )

        # final_relevant_nodes = list(set(multi_query_final_nodes + origin_retrieve_documents))
        all_nodes = multi_query_final_nodes + origin_retrieve_documents
        node_dict = {}
        for node in all_nodes:
            if f(node) not in node_dict:
                node_dict[f(node)] = node

        final_relevant_nodes = list(node_dict.values())
        # print('-'*20)
        # print(f'final_relevant_nodes len {len(final_relevant_nodes)}')
        # print('-'*20)

        sorted_relevant_nodes = sorted(final_relevant_nodes, key=lambda node: node.metadata['distance'], reverse=True)
        # rerank
        if self.rerank_model is not None:
            if rerank_limit < 0:
                rerank_limit = limit
            else:
                rerank_limit = rerank_limit if limit > rerank_limit else limit
            rerank_relevant_nodes = self.rerank_model.rerank(documents=sorted_relevant_nodes,
                                                             query=query,
                                                             top_n=rerank_limit)
            retrieved_documents = rerank_relevant_nodes
        else:
            retrieved_documents = sorted_relevant_nodes[:limit]
        # print('-'*30)
        # print('relevant docs: ')
        content_info = ''
        for idx, relevant_node in enumerate(retrieved_documents):
            cur_node_info = f'''
            chunk {idx}: 
            {relevant_node.get_content(MetadataMode.LLM)} 
            \n
            '''
            content_info += cur_node_info
            # print(relevant_node)
            # print('-'*20)
        # print('-'*30)
        prompt_template = [
            ChatMessage(role='system', content=RAG_SYSTEM_PROMPT),
            ChatMessage(role='user', content=RAG_USER_PROMPT_TEMPLATE)
        ]
        chat_message_prompt = ChatPromptTemplate(message_templates=prompt_template)
        messages = chat_message_prompt.format_messages(content_str=content_info, query_str=query)

        resp = self.llm.chat(messages)
        return resp , retrieved_documents