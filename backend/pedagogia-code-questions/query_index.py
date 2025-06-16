from RepoKnowledgeGraph import RepoKnowledgeGraph

import json

#TODO need to remember to change which node element is being embedded
for repo_dir in ['/mnt/workdir/jperez/data/tiger-2026/epita-ing-assistants-yaka-tiger-2026-paris-55',
                 '/mnt/workdir/jperez/data/tiger-2026/epita-ing-assistants-yaka-tiger-2026-paris-106',
                 '/mnt/workdir/jperez/data/tiger-2026/epita-ing-assistants-yaka-tiger-2026-toulouse-19',
                 '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-lucas.llombart',
                 '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-juliette.meyniel',
                 '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-yanis.belami']:
    group_name = repo_dir.split('/')[-1]
    group_dir = f'data/{group_name}'
    knowledge_graph_path = f'{group_dir}/knowledge_graph.json'
    knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(knowledge_graph_path, use_embed=False)
    knowledge_graph.print_tree()
#knowledge_graph.save_graph_to_file('knowledge_graph.json')

    with open(f'{group_dir}/dataset_single_chunk.json', 'r') as f:
        question_dataset = json.load(f)

    dataset = []

    for element in question_dataset:
        node = element['chunk']
        question = element['generated_question']
        results = knowledge_graph.code_index.query(query=question, n_results=10)
        dataset.append({
            'chunk': element['chunk'],
            'generated_question': question,
            'target_response': element['target_response'],
            'results': results,
        })


    with open(f'{group_dir}/result_single_chunk_embed_description.json', 'w') as outfile:
        json.dump(dataset, outfile)

