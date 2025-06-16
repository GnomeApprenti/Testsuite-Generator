from .RepoKnowledgeGraph import RepoKnowledgeGraph
from .ModelService import ModelService
import json
from pathlib import Path
import os.path


def run(repo_dir:str, data_dir:str='~/pedagogia-code-questions/data'):
    model_service = ModelService()

    group_name = repo_dir.split('/')[-1]
    group_dir = os.path.join(data_dir, group_name)
    if not os.path.exists(f'{group_dir}/dataset_single_chunk.json'):
        Path(group_dir).mkdir(parents=True, exist_ok=True)
        knowledge_graph_path = os.path.join(group_dir,'knowledge_graph.json')
        knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(filepath=knowledge_graph_path, use_embed=False)

        node_list = list(knowledge_graph)

        dataset = []

        for node in node_list:

            if not node.node_type == 'chunk':  # Accessing the data
                continue
            else:
                print(f'Generating question for {node.id}')
                question = model_service.query(
                    f"Based on the following code:\n{node.content}\n\nWrite a question that tests understanding of the behavior, logic, or purpose of specific functions, classes, or variables in the code. Always refer to them by name. Avoid phrases like 'the provided code' or 'this snippet'. Only write the question.")
                target_response = model_service.query(
                    f'Answer this question: {question} using this chunk of code: {node.content}')
                # results = knowledge_graph.code_index.query(query=question, n_results=5)
                dataset.append({
                    'chunk': node.dict(),
                    'generated_question': question,
                    'target_response': target_response,
                })


        with open(f'{group_dir}/dataset_single_chunk.json', 'w') as outfile:
            json.dump(dataset, outfile)