from .RepoKnowledgeGraph import RepoKnowledgeGraph
from .ModelService import ModelService

from pathlib import Path
import os.path


def run(repo_dir:str, data_dir:str='~/pedagogia-code-questions/data'):
    model_service = ModelService()
    repo_name = repo_dir.split('/')[-1]
    group_dir = os.path.join(data_dir, repo_name)
    Path(group_dir).mkdir(parents=True, exist_ok=True)
    knowledge_graph_path = os.path.join(group_dir,'knowledge_graph.json')
    knowledge_graph = RepoKnowledgeGraph.from_path(repo_dir, index_nodes=False)
    knowledge_graph.print_tree()
    knowledge_graph.save_graph_to_file(filepath=knowledge_graph_path)