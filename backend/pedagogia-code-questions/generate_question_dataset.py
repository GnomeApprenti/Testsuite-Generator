from RepoKnowledgeGraph import RepoKnowledgeGraph
from ModelService import ModelService
import json
from pathlib import Path
import os.path

model_service = ModelService()

for repo_dir in ['/mnt/workdir/jperez/data/tiger-2026/epita-ing-assistants-yaka-tiger-2026-paris-55',
                 '/mnt/workdir/jperez/data/tiger-2026/epita-ing-assistants-yaka-tiger-2026-paris-106',
                 '/mnt/workdir/jperez/data/tiger-2026/epita-ing-assistants-yaka-tiger-2026-toulouse-19',
                 '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-lucas.llombart',
                 '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-juliette.meyniel',
                 '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-yanis.belami']:
    group_name = repo_dir.split('/')[-1]
    group_dir = f'data/{group_name}'
    Path(group_dir).mkdir(parents=True, exist_ok=True)
    knowledge_graph_path = f'{group_dir}/knowledge_graph.json'
    if not os.path.exists(knowledge_graph_path):
        source_file_dir = os.path.join(repo_dir, 'src')
        if os.path.exists(source_file_dir):
            repo_dir = source_file_dir
        knowledge_graph = RepoKnowledgeGraph.from_path(repo_dir, index_nodes=False)
        knowledge_graph.print_tree()
        knowledge_graph.save_graph_to_file(filepath=knowledge_graph_path)
    else:
        knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(filepath=knowledge_graph_path, use_embed=True)

    node_list = list(knowledge_graph)

    # data_info = {
    #    'generation_date_time': datetime.now().isoformat(),
    #   'repo_dir': repo_dir,
    #    'generation_strategy': '1 question for 1 chunk, on all chunks'
    # }
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
    # data_info['dataset'] = dataset

    with open(f'{group_dir}/dataset_single_chunk.json', 'w') as outfile:
        json.dump(dataset, outfile)
        # json.dump(data_info, outfile)
