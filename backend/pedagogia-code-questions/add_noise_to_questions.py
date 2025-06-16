from RepoKnowledgeGraph import RepoKnowledgeGraph
from ModelService import ModelService
import json
import os.path

model_service = ModelService()

for group_dir in ['data/epita-ing-assistants-acu-42sh-2025-lyon-20',
                  'data/epita-ing-assistants-acu-42sh-2025-paris-50',
                  'data/epita-ing-assistants-acu-42sh-2025-strasbourg-7', ]:

    target_file_name = 'result_noisy_question_single_chunk_embed_description.json'

    if not os.path.exists(os.path.join(group_dir, target_file_name)):
        knowledge_graph_path = f'{group_dir}/knowledge_graph.json'
        knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(knowledge_graph_path, use_embed=False)
        knowledge_graph.print_tree()
        # knowledge_graph.save_graph_to_file('knowledge_graph.json')

        with open(f'{group_dir}/dataset_single_chunk.json', 'r') as f:
            question_dataset = json.load(f)

        dataset = []

        for element in question_dataset:
            node = element['chunk']
            question = element['generated_question']
            baseline_result = knowledge_graph.code_index.query(query=question, n_results=10)

            noisy_prompt_1 = f"""You are a helpful assistant modifying questions about code to make them slightly more vague while keeping them realistic. 
            Transform the following question to make it less specific, more ambiguous, or slightly incomplete, as a human might phrase it imprecisely. 
            Do not completely change the topic. Keep it a plausible query about code. 
            Question: {question}
            Return ONLY the modified question."""

            noisy_prompt_2 = f"""Rewrite this coding-related question to make it more general, by removing specific variable names, functions, or APIs.
            Keep it understandable and plausible, but make it less precise.
            Original: {question}
            Only return the modified question."""

            noisy_prompt_3 = f"""Make this programming-related question more vague by omitting a key detail such as a data type, method name, or context.
            It should still resemble a real question, just slightly unclear.
            Original: {question}
            Only return the modified question."""

            noisy_prompt_4 = f"""Rewrite this technical programming question using less precise, more everyday language. 
            Introduce some ambiguity but keep it recognizable as a real question.
            Original: {question}
            Only return the modified question."""



            noisy_question_1 = model_service.query(noisy_prompt_1)
            noisy_results_1 = knowledge_graph.code_index.query(query=noisy_question_1, n_results=10)
            noisy_question_2 = model_service.query(noisy_prompt_2)
            noisy_results_2 = knowledge_graph.code_index.query(query=noisy_question_2, n_results=10)
            noisy_question_3 = model_service.query(noisy_prompt_3)
            noisy_results_3 = knowledge_graph.code_index.query(query=noisy_question_3, n_results=10)
            noisy_question_4 = model_service.query(noisy_prompt_4)
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

        with open(os.path.join(group_dir, target_file_name), 'w') as outfile:
            json.dump(dataset, outfile)
    else:
        print(f'Skipping {group_dir}')
