from .RepoKnowledgeGraph import RepoKnowledgeGraph
from .ModelService import ModelService
import json
import os.path


def run(repo_dir:str, data_dir:str='~/pedagogia-code-questions/data'):
    model_service = ModelService()
    repo_name = repo_dir.split('/')[-1]
    group_dir = os.path.join(data_dir, repo_name)
    knowledge_graph_path = os.path.join(group_dir, 'knowledge_graph.json')
    knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(knowledge_graph_path, use_embed=False, index_nodes=False)
    knowledge_graph.print_tree()

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
        noisy_question_2 = model_service.query(noisy_prompt_2)
        noisy_question_3 = model_service.query(noisy_prompt_3)
        noisy_question_4 = model_service.query(noisy_prompt_4)
        dataset.append({
            'target_chunk': node,
            'generated_question': question,
            'target_response': element['target_response'],
            'baseline_result': baseline_result,
            'noisy_question_1': noisy_question_1,
            'noisy_question_2': noisy_question_2,
            'noisy_question_3': noisy_question_3,
            'noisy_question_4': noisy_question_4,

        })

    with open(f'{group_dir}/dataset_noisy_question_single_chunk_embed_chunk.json', 'w') as outfile:
        json.dump(dataset, outfile)
