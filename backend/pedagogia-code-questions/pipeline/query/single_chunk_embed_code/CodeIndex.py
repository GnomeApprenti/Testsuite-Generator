import logging
import uuid
from datetime import datetime

import os
import chromadb


from .utils.logger_utils import setup_logger
from .Node import ChunkNode
from .ModelService import ModelService

LOGGER_NAME = 'CODE_INDEX_LOGGER'
BACKEND = str(os.getenv("BACKEND", 'azure'))
STOP_AFTER_ATTEMPT = int(os.getenv("STOP_AFTER_ATTEMPT", 5))
WAIT_BETWEEN_RETRIES = int(os.getenv("WAIT_BETWEEN_RETRIES", 2))
MODEL_ID = os.getenv("MODEL_ID")
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2048))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.2))
TOP_P = float(os.getenv('TOP_P', 0.95))
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
STOP = None
EMBEDDING_MODEL_URL = os.getenv('EMBEDDING_MODEL_URL')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
EMBEDDING_MODEL_API_KEY = os.getenv('EMBEDDING_MODEL_API_KEY', "no_need")
EMBEDDING_NUMBER_DIMENSIONS = int(os.getenv('EMBEDDING_NUMBER_DIMENSIONS', 1536))

CHROMADB_HOST = os.getenv('CHROMADB_HOST', "localhost")
CHROMADB_PORT = int(os.getenv('CHROMADB_PORT', "8008"))
CHROMADB_PROTOCOL = os.getenv('CHROMADB_PROTOCOL', "http")
ALPHA_SEARCH_VALUE = os.getenv('ALPHA_SEARCH_VALUE', "0.8")


class CodeIndex:
    def __init__(self, nodes, use_embed:bool = True):
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)
        self.model_service = ModelService()
        self.chromadb_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        self.chromadb_client.heartbeat()
        self.collection_name = str(uuid.uuid4())
        self.collection = self.chromadb_client.create_collection(
            name=self.collection_name,
            metadata={
                'created': datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
            }
        )
        for node in nodes:
            if node.node_type == 'chunk':
                self.index_node(node, use_embed=use_embed)



    def index_node(self, node: ChunkNode, use_embed:bool=True) -> ChunkNode:
        self.logger.info(f'Indexing node : {node.id}')
        document = node.content
        if node.embedding is None or not use_embed:
            embedding = self.model_service.embed(text_to_embed=node.get_field_to_embed())
            node.embedding = embedding
        else:
            embedding = node.embedding
        self.collection.add(
            documents=[document],
            embeddings=[embedding],
            metadatas=[node.dict()],
            ids=[node.id],
        )
        self.logger.info(f'Indexed node : {node.id}')
        return node


    def query(self, query: str, n_results:int) -> list:
        """
        Chromadb will do a vector search on all the regular fields, and a semantic search on the auto-embedding field
        and combine the results using Rank Fusion to arrive at a fusion score that is used to rank the hits.
        K = rank of document in keyword search
        S = rank of document in semantic search

        rank_fusion_score = 0.7 * K + 0.3 * S
        The 0.7 and 0.3 values can be changed using the alpha parameter.
        alpha corresponds to the weight of the vector search ranking

        """
        embedding = self.model_service.embed(text_to_embed=query)
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
            )
            return results

        except Exception as e:
            self.logger.error(f'Failed to query: {e}', exc_info=True)
            raise e


    def __del__(self):
        self.logger.info(f'Dropping code index collection')
        self.chromadb_client.delete_collection(name=self.collection_name)