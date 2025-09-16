import json
from typing import List, Dict, Union
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from XRAG.model.model_manager import  model_factory
from XRAG.chatbot.query_engine import QueryEngine
from llama_index.core.schema import MetadataMode
from ragas import EvaluationDataset
from ragas import evaluate
import random
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,  # RAG系统答案和参考答案相似度
    AnswerRelevancy,  # RAG系统的答案和问题的相似度
    ContextRecall,  # RAG系统答案在上下文出现频率
    ContextPrecision # 参考答案在上下文出现频率
)
from datetime import datetime

class RAGEvaluationSystem:
    reference_dataset_path: str
    eval_dataset: List[Dict]
    eval_llm: Union[OpenAILike, OpenAI]
    eval_embedding: Union[OpenAIEmbedding, OpenAILikeEmbedding]
    def __init__(self, reference_dataset_path: str='./config/eval_reference.json',
                       eval_llm: Union[OpenAILike, OpenAI] = None,
                       eval_embedding: Union[OpenAIEmbedding, OpenAILikeEmbedding] = None):
        self.reference_dataset_path = reference_dataset_path
        self._load_dataset()
        if eval_llm is None:
            self.eval_llm = model_factory.get_llm()
        else:
            self.eval_llm = eval_llm
        if eval_embedding is None:
            self.eval_embedding = model_factory.get_embedding()
        else:
            self.eval_embedding = eval_embedding
    def _load_dataset(self):
        with open(self.reference_dataset_path, 'r') as f:
            eval_dataset_str = f.read()
        self.eval_dataset = json.loads(eval_dataset_str)
    def eval(self, rag_system: QueryEngine, eval_dataset_len: int=10, limit: int=5):
        eval_dataset = self.eval_dataset
        if len(self.eval_dataset) > eval_dataset_len:
            eval_dataset = random.sample(self.eval_dataset, eval_dataset_len)

        data_to_eval = []
        data_to_save = []
        for reference_pair in eval_dataset:
            query = reference_pair['query']
            reference_answer = reference_pair['answer']
            print('-'*20)
            print('query: ')
            print(query)
            print('-'*20)
            print('reference: ')
            print(reference_answer)
            print('-'*20)
            resp, relevant_nodes = rag_system.chat(query, limit)
            relevant_contents = [node.get_content(MetadataMode.LLM) for node in relevant_nodes]
            print('-'*30)
            print(resp)
            print('-'*30)
            data_to_eval.append(
                   {
                        "user_input": query,
                        "retrieved_contexts": relevant_contents,
                        "response": str(resp),
                        "reference": reference_answer,
                   }
            )
            data_to_save.append({'query': query, 'answer': str(resp)})

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        file_name = f'rag_system_{current_time}.json'
        with open(f'./record/{file_name}', 'w') as f:
            f.write('[\n')
            for pair in data_to_save:
                f.write(json.dumps(pair) + ',\n')
            f.write(']')

        dataset = EvaluationDataset.from_list(data_to_eval)
        eval_llm = LlamaIndexLLMWrapper(self.eval_llm)
        eval_embedding = LlamaIndexEmbeddingsWrapper(self.eval_embedding)
        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(llm=eval_llm),
                AnswerRelevancy(llm=eval_llm),
                ContextPrecision(llm=eval_llm),
                ContextRecall(llm=eval_llm)
            ],
            llm=eval_llm,
            embeddings=eval_embedding
        )
        return result