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

    with open(f'{group_dir}/dataset_noisy_question_single_chunk_embed_chunk.json', 'r') as f:
        question_dataset = json.load(f)

    dataset = []

    for element in question_dataset:
        node = element['target_chunk']
        question = element['generated_question']
        baseline_result = element['baseline_result']
        noisy_question_1 = element['noisy_question_1']
        noisy_results_1 = knowledge_graph.code_index.query(query=noisy_question_1, n_results=10)
        noisy_question_2 = element['noisy_question_2']
        noisy_results_2 = knowledge_graph.code_index.query(query=noisy_question_2, n_results=10)
        noisy_question_3 = element['noisy_question_3']
        noisy_results_3 = knowledge_graph.code_index.query(query=noisy_question_3, n_results=10)
        noisy_question_4 = element['noisy_question_4']
        noisy_results_4 = knowledge_graph.code_index.query(query=noisy_question_4, n_results=10)
        dataset.append({
            'target_chunk': node,
            'generated_question': question,
            'target_response': element['target_response'],
            'baseline_result': baseline_result,
            'noisy_question_1': noisy_question_1,
            'noisy_results_1': noisy_results_1,
            'noisy_question_2': noisy_question_2,
            'noisy_results_2': noisy_results_2,
            'noisy_question_3': noisy_question_3,
            'noisy_results_3': noisy_results_3,
            'noisy_question_4': noisy_question_4,
            'noisy_results_4': noisy_results_4,

        })

    with open(f'{group_dir}/result_noisy_question_single_chunk_embed_chunk.json', 'w') as outfile:
        json.dump(dataset, outfile)
