from .RepoKnowledgeGraph import RepoKnowledgeGraph
from .ModelService import ModelService

from pathlib import Path
import os.path
import json 
import pandas as pd 
import torch 

# Disable automatic device mapping
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# If you want to force CPU usage:
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Clear GPU memory if using GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()


EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'Salesforce/SFR-Embedding-Code-400M_R')

def run(repo_dir:str, data_dir:str='~/pedagogia-code-questions/data'):
    group_name = repo_dir.split('/')[-1]
    group_dir = os.path.join(data_dir, group_name)
    several_model_dataset_path = f'{group_dir}/dataset_several_model_description_single_chunk.json'
    knowledge_graph_path = os.path.join(group_dir,'knowledge_graph.json')
    model_service = ModelService()
    with open(knowledge_graph_path, 'r') as f: 
        knowledge_graph_dict = json.load(f)
    
    with open(several_model_dataset_path, 'r') as f: 
        several_model_question_dataset = json.load(f)

    result_several_models_path = os.path.join(group_dir,'results_several_model_description.json')
    if not os.path.exists(result_several_models_path):
        new_question_dataset = {}
    else: 
        with open(result_several_models_path, 'r') as f: 
            new_question_dataset = json.load(f)


    knowledge_graph = RepoKnowledgeGraph.from_dict(knowledge_graph_dict, model_service=model_service, use_embed=False)

    for chunk_id, element in several_model_question_dataset.items(): 
        question = element['generated_question']
        
        if chunk_id in new_question_dataset: 
            new_element = new_question_dataset[chunk_id].copy()
        else: 
            new_element = element.copy()
        
        if f'result_ChunkEmbed_{EMBEDDING_MODEL_NAME}_NoTask' not in new_element: 
            # Add the query result
            new_element[f'result_ChunkEmbed_{EMBEDDING_MODEL_NAME}_NoTask'] = knowledge_graph.code_index.query(query=question, n_results=10)
        new_question_dataset[chunk_id] = new_element

    
    with open(result_several_models_path, 'w') as outfile: 
        json.dump(new_question_dataset, outfile)


     
if __name__ == "__main__":
    run(repo_dir='/mnt/workdir/lelkoussy/pedagogia-code-questions/data/epita-ing-assistants-yaka-js-workshop-2025-lucas.llombart', data_dir='/mnt/workdir/lelkoussy/pedagogia-code-questions/data')

    # Open knowledge graph JSON, modify in json, then create knowledge graph object, then query

    # Need to save the dataset such that for each chunk + question 
    # We have chunk, target_question
