from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

from .utils.logger_utils import setup_logger

LOGGER_NAME = "MODEL_SERVICE_LOGGER"
# GENERATION ENV VARIABLES
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", 'http://0.0.0.0:8000/v1')
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN", 'no-need')
MODEL_NAME = os.getenv('MODEL_NAME', "meta-llama/Llama-3.2-3B-Instruct")
# EMBED ENV VARIABLES
OPENAI_EMBED_BASE_URL = os.getenv("OPENAI_EMBED_BASE_URL", 'http://0.0.0.0:8001/v1')
OPENAI_EMBED_TOKEN = os.getenv("OPENAI_EMBED_TOKEN", 'no-need')
EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', "Alibaba-NLP/gte-Qwen2-1.5B-instruct")

STOP_AFTER_ATTEMPT = int(os.getenv("STOP_AFTER_ATTEMPT", 5))
WAIT_BETWEEN_RETRIES = int(os.getenv("WAIT_BETWEEN_RETRIES", 2))


class ModelService:
    def __init__(self):
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)
        self.client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_TOKEN,
        )
        self.embed_client = OpenAI(
            base_url=OPENAI_EMBED_BASE_URL,
            api_key=OPENAI_EMBED_TOKEN,
        )

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    def query(self, prompt:str) -> str:
        self.logger.info(f'Querying model {MODEL_NAME}')
        completion = self.client.chat.completions.create(
          model=MODEL_NAME,
          messages=[
            {"role": "user", "content": prompt}
          ]
        )
        return completion.choices[0].message.content

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    def embed(self, text_to_embed:str) -> list:
        self.logger.info(f'Embedding text using {EMBED_MODEL_NAME}')
        response = self.embed_client.embeddings.create(
            input=text_to_embed,
            model=EMBED_MODEL_NAME,
        )

        return response.data[0].embedding

