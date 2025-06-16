from .RepoKnowledgeGraph import RepoKnowledgeGraph
from .ModelService import ModelService

from pathlib import Path
import os.path
import json 
import re

def extract_thought_and_conclusion(llm_output: str):
    """
    Extracts the thought within <think>...</think> and everything after it as the conclusion.
    
    If no <think> tags are present, returns ("", full string).
    
    Parameters:
        llm_output (str): The string output from a reasoning LLM.
    
    Returns:
        tuple: (thought: str, conclusion: str)
    """
    match = re.search(r'<think>(.*?)</think>', llm_output, re.DOTALL)
    
    if match:
        thought = match.group(1).strip()
        # Everything after the closing </think> tag
        end_idx = match.end()
        conclusion = llm_output[end_idx:].strip()
    else:
        thought = ""
        conclusion = llm_output.strip()
    
    return thought, conclusion


OPENAI_TOKEN = os.getenv("OPENAI_TOKEN", 'no-need')

MODEL_NAMES = ['meta-llama/Llama-3.2-3B-Instruct','Qwen/Qwen3-1.7B', 'microsoft/Phi-4-mini-reasoning']
current_model_name = 'Qwen/Qwen3-1.7B'
base_model_name = 'meta-llama/Llama-3.2-3B-Instruct'

def run(repo_dir:str, data_dir:str='~/pedagogia-code-questions/data'):
    model_service = ModelService()
    group_name = repo_dir.split('/')[-1]
    group_dir = os.path.join(data_dir, group_name)
    knowledge_graph_path = os.path.join(group_dir,'knowledge_graph.json')
    with open(knowledge_graph_path, 'r') as f: 
        knowledge_graph_dict = json.load(f)
    
    question_dataset_path = f'{group_dir}/dataset_single_chunk.json'
    with open(question_dataset_path, 'r') as f: 
        question_dataset = json.load(f)
        
    several_model_dataset_path = f'{group_dir}/dataset_several_model_description_single_chunk.json'
    if not os.path.exists(several_model_dataset_path):

        several_model_dataset = {element['chunk']['id']: {'chunk': element['chunk'],
                                'generated_question': element['generated_question'],
                                'target_response': element['target_response'], 
                                 f'description_{base_model_name}_Summarizing': element['chunk']['description']} for element in question_dataset} 

    else: 
        with open(several_model_dataset_path, 'r') as f: 
            several_model_dataset = json.load(f)

    # TODO: add different tasks 
    tasks = [{'task_name': 'Summarizing', 'task_prompt': 'Summarize this {language} code chunk in a few sentences: {content}'}, 
             {'task_name': 'Reviewing', 'task_prompt': 'You are reviewing a piece of code. Your goal is to deeply understand it and identify any issues or improvements related to correctness, clarity, performance, or maintainability. Here is the code: {content}'}, 
    {'task_name': 'Completion', 'task_prompt': 'You are completing a piece of code. Continue it in a way that is consistent, correct, and idiomatic. Here is the code: {content}'}
]

     
    for node in knowledge_graph_dict['nodes']: 
        if node['class'] == 'ChunkNode': 
            node_data = node['data']
            chunk_id = node_data['id']
            
            for task_dict in tasks:
                task_name = task_dict['task_name'] 
                task_prompt = task_dict['task_prompt'].format_map(node_data)
                model_output = model_service.query(task_prompt, model_name=current_model_name)
                thought, conclusion = extract_thought_and_conclusion(model_output)
                several_model_dataset[chunk_id][f'description_{current_model_name}_{task_name}'] = conclusion
                if len(thought): 
                    several_model_dataset[chunk_id][f'thought_{current_model_name}_{task_name}'] = thought
                    several_model_dataset[chunk_id][f'description_thought_{current_model_name}_{task_name}'] = model_output


    
    with open(several_model_dataset_path, 'w') as outfile: 
        json.dump(several_model_dataset, outfile)


