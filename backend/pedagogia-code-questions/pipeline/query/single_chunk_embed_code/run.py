from .RepoKnowledgeGraph import RepoKnowledgeGraph
from .ModelService import ModelService
import json
import os.path


def run(repo_dir:str, data_dir:str='~/pedagogia-code-questions/data'):
    model_service = ModelService()
    repo_name = repo_dir.split('/')[-1]
    group_dir = os.path.join(data_dir, repo_name)
    knowledge_graph_path = os.path.join(group_dir, 'knowledge_graph.json')
    knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(knowledge_graph_path, use_embed=False)
    knowledge_graph.print_tree()

    with open(f'{group_dir}/dataset_single_chunk.json', 'r') as f:
        question_dataset = json.load(f)

    dataset = []

    for element in question_dataset:
        node = element['chunk']
        question = element['generated_question']
        results = knowledge_graph.code_index.query(query=question, n_results=10)
        dataset.append({
            'chunk': node,
            'generated_question': question,
            'target_response': element['target_response'],
            'results': results,
        })

    with open(f'{group_dir}/result_single_chunk_embed_chunk.json', 'w') as outfile:
        json.dump(dataset, outfile)
