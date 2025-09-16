from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.prompts import ChatMessage, ChatPromptTemplate
from XRAG.ingestion.vector_store import VectorStore
from XRAG.model.model_manager import model_factory
from typing import Union
from XRAG.prompt.prompt import (
    HYPE_GENERATE_SYSTEM_PROMPT,
    HYPE_GENERATE_USER_PROMPT
)

class HyPEGenerator:
    llm: Union[OpenAI, OpenAILike]
    def __init__(self, llm: Union[OpenAI, OpenAILike] = None):
        if llm is None:
            self.llm = model_factory.get_llm()
        else:
            self.llm = llm
    # No chunk
    def generate(self, query: str, chunk_str: str) -> str:
        hype_generate_template = [
            ChatMessage(role='system', content=HYPE_GENERATE_SYSTEM_PROMPT),
            ChatMessage(role='user', content=HYPE_GENERATE_USER_PROMPT)
        ]

        hype_generate_prompt = ChatPromptTemplate.from_messages(message_templates=hype_generate_template)
        hype_generate_chat_prompt = hype_generate_prompt.format_messages(query_str=query, content_str=chunk_str)
        resp = self.llm.chat(hype_generate_chat_prompt)
        virtual_document = resp.message.blocks[0].text
        # print('-'*30)
        # print('virtual document text: ')
        # print(virtual_document)
        # print('-'*30)
        return virtual_document