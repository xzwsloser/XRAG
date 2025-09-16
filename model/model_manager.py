from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from XRAG.model.rerank_model import RerankModel
import json
from pathlib import Path

class ModelFactory:
    def __init__(self, config_path: str = '', provider: str = 'SiliconCloud'):
        if config_path == '':
            BASE_DIR = Path(__file__).resolve().parent
            config_path = (BASE_DIR.parent / 'config' / 'config.json').resolve()
        self.config = config_path
        model_config = self._load_config()
        special_config = model_config[provider]
        api_key = special_config['LLM_API_KEY']
        base_url = special_config['LLM_BASE_URL']
        self.llm = OpenAILike(
            api_key=api_key,
            api_base=base_url,
            model=special_config['CHAT_MODEL_NAME'],
            is_chat_model=True,
            is_function_calling_model=True
        )

        self.embed_model = OpenAILikeEmbedding(
            api_key=api_key,
            api_base=base_url,
            model_name=special_config['EMBED_MODEL_NAME']
        )

        self.vision_model = OpenAILike(
            api_key=api_key,
            api_base=base_url,
            model=special_config['VLM_MODEL_NAME'],
            is_chat_model=True
        )

        self.rerank_model = RerankModel(
            api_key=api_key,
            api_base=base_url,
            rerank_model_name=special_config['RERANK_MODEL_NAME']
        )

    def _load_config(self):
        with open(self.config, 'r') as f:
            config_info = f.read()
        return json.loads(config_info)

    def get_llm(self):
        return self.llm

    def get_embedding(self):
        return self.embed_model

    def get_vision_model(self):
        return self.vision_model

    def get_rerank_model(self) -> RerankModel:
        return self.rerank_model

# config_path = '../config/config.json'
model_factory = ModelFactory()