from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, Document
from typing import List
import jieba

class CustomSplitter:
    def __init__(self, chunk_size: int=1024, chunk_overlap: int=100, is_chinese: bool = True):
        if is_chinese:
            self.splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="",
                tokenizer=lambda text: list(jieba.cut(text))
            )
        else:
            self.splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
    def split_documents(self, documents: List[Document]):
        return self.splitter.get_nodes_from_documents(documents)