from .RepoKnowledgeGraph import RepoKnowledgeGraph
from .ModelService import ModelService

from pathlib import Path
import os.path
import json 
import pandas as pd 

OPENAI_TOKEN = os.getenv("OPENAI_TOKEN", 'no-need')

MODEL_NAMES = ['meta-llama/Llama-3.2-3B-Instruct','Qwen/Qwen3-1.7B', 'google/gemma-3-4b-it', 'microsoft/Phi-4-mini-reasoning']

base_model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tasks = [{'task_name': 'Summarizing', 'task_prompt': 'Summarize this {language} code chunk in a few sentences: {content}'}, 
             {'task_name': 'Reviewing', 'task_prompt': 'You are reviewing a piece of code. Your goal is to deeply understand it and identify any issues or improvements related to correctness, clarity, performance, or maintainability. Here is the code: {content}'}, 
             {'task_name': 'Completion', 'task_prompt': 'You are completing a piece of code. Continue it in a way that is consistent, correct, and idiomatic. Here is the code: {content}'}]


def run(repo_dir:str, data_dir:str='~/pedagogia-code-questions/data'):
    group_name = repo_dir.split('/')[-1]
    group_dir = os.path.join(data_dir, group_name)
    several_model_dataset_path = f'{group_dir}/dataset_several_model_description_single_chunk.json'
    knowledge_graph_path = os.path.join(group_dir,'knowledge_graph.json')
    with open(knowledge_graph_path, 'r') as f: 
        knowledge_graph_dict = json.load(f)
    
    with open(several_model_dataset_path, 'r') as f: 
        several_model_question_dataset = json.load(f)

    
    with open(f'{group_dir}/result_single_chunk_embed_chunk.json', 'r') as f:
        chunk_embed_results_dict = json.load(f)

    chunk_embed_results_extracted = {element['chunk']['id']: element['results'] for element in chunk_embed_results_dict }
    
    df = pd.DataFrame.from_dict(several_model_question_dataset, orient='index')

    
    result_several_models_path = os.path.join(group_dir,'results_several_model_description.json')
    if not os.path.exists(result_several_models_path):
        new_question_dataset = {}
    else: 
        with open(result_several_models_path, 'r') as f: 
            new_question_dataset = json.load(f)
    for model_name in MODEL_NAMES: 
        for task_dict in tasks: 
            task_name = task_dict['task_name']
            for verbalisation in [f'description_{model_name}_{task_name}', f'thought_{model_name}_{task_name}', f'description_thought_{model_name}_{task_name}']: 
                if verbalisation in df.columns: 
                    new_knowledge_graph = {'edges': knowledge_graph_dict['edges'].copy()}
                    new_nodes = []
                    for node in knowledge_graph_dict['nodes']: 
                        try: 
                            new_node = node.copy()
                            if node['class'] == 'ChunkNode':
                                chunk_id = node['id']
                                print(chunk_id)
                                new_description = several_model_question_dataset[chunk_id][verbalisation]
                                new_node['data']['description'] = new_description
                            new_nodes.append(new_node)
                        except KeyError as e: 
                            print(f'Could not find key: {e}')
                    new_knowledge_graph['nodes'] = new_nodes

                    knowledge_graph = RepoKnowledgeGraph.from_dict(new_knowledge_graph, use_embed=False)

                    for chunk_id, element in several_model_question_dataset.items(): 
                        question = element['generated_question']
                        
                        if chunk_id in new_question_dataset: 
                            new_element = new_question_dataset[chunk_id].copy()
                        else: 
                            new_element = element.copy()
                            new_element[f'result_ChunkEmbed_no_model_NoTask'] = chunk_embed_results_extracted[chunk_id]

                        # Always update the verbalization (fix for the issue)
                        if verbalisation in several_model_question_dataset[chunk_id]: 
                            new_element[verbalisation] = several_model_question_dataset[chunk_id][verbalisation]
                        
                        # Add the query result
                        new_element[f'result_{verbalisation}'] = knowledge_graph.code_index.query(query=question, n_results=10)
                        new_question_dataset[chunk_id] = new_element

    
    with open(result_several_models_path, 'w') as outfile: 
        json.dump(new_question_dataset, outfile)


     
if __name__ == '__main__': 
    run(repo_dir='/mnt/workdir/lelkoussy/pedagogia-code-questions/data/epita-ing-assistants-acu-42sh-2025-lyon-20', data_dir='/mnt/workdir/lelkoussy/pedagogia-code-questions/data')
    # Open knowledge graph JSON, modify in json, then create knowledge graph object, then query

    # Need to save the dataset such that for each chunk + question 
    # We have chunk, target_question
