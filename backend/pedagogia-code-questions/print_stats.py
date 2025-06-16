import pandas as pd
import json
import os.path
import argparse
import numpy as np 
from sklearn.metrics import ndcg_score


def extract_found_and_order(row, results_column_name='results', chunk_column_name='chunk'):
    # Handle different chunk column structures
    if chunk_column_name in row:
        if isinstance(row[chunk_column_name], dict):
            target_id = row[chunk_column_name]['id']
        else:
            target_id = row[chunk_column_name]
    else:
        # Fallback for different column names
        if 'target_chunk' in row:
            if isinstance(row['target_chunk'], dict):
                target_id = row['target_chunk']['id']
            else:
                target_id = row['target_chunk']
        else:
            raise KeyError(f"No chunk column found. Available columns: {list(row.keys())}")

    id_list = row[results_column_name]['ids'][0]  # The nested list of IDs
    if target_id in id_list:
        return pd.Series({'found': True, 'order': id_list.index(target_id)})
    else:
        return pd.Series({'found': False, 'order': None})


def compute_rank_stats(dataframe):
    stats = {}
    found_percentage = dataframe['found'].mean() * 100
    stats['found_percentage'] = found_percentage

    found_df = dataframe[dataframe['found']]
    rank_counts = found_df['order'].value_counts().sort_index()
    rank_percentages = (rank_counts / len(dataframe)) * 100

    cumulative = 0
    for rank in range(10):  # top 1 to 10
        pct = rank_percentages.get(rank, 0)
        cumulative += pct
        stats[f'top_{rank + 1}'] = cumulative

    return stats

def compute_ndcg_stats_sklearn(dataframe, k=10):
    """
    Compute NDCG@k scores using scikit-learn's ndcg_score.
    
    Parameters:
    dataframe: pandas DataFrame with columns 'found' (bool) and 'order' (int, 0-based ranking)
    k: maximum k value for NDCG@k calculations
    
    Returns:
    dict: NDCG@k scores for k=1 to k
    """
    ndcg_stats = {}
    
    # Prepare data for sklearn
    # y_true: relevance scores (1 for found items, 0 for not found)
    y_true = dataframe['found'].astype(int).values.reshape(1, -1)
    
    # y_score: inverse of order (higher scores for better rankings)
    # Items not found get score 0, found items get score based on inverse order
    max_order = dataframe['order'].max() + 1
    y_score = np.where(dataframe['found'], 
                      max_order - dataframe['order'], 
                      0).reshape(1, -1)
    
    # Calculate NDCG@k for each k from 1 to k
    for rank in range(1, k + 1):
        ndcg_at_k = ndcg_score(y_true, y_score, k=rank)
        ndcg_stats[f'NDCG@{rank}'] = ndcg_at_k
    
    return ndcg_stats

def print_combined_stats(stats_chunk, stats_description, stats_noisy_chunk=None, stats_noisy_desc=None):
    print('=' * 140)
    print('RETRIEVAL PERFORMANCE COMPARISON')
    print('=' * 140)
    
    # Determine which datasets we have
    datasets = []
    if stats_chunk is not None:
        datasets.append(('Chunk Embed', stats_chunk))
    if stats_description is not None:
        datasets.append(('Desc Embed', stats_description))
    if stats_noisy_chunk is not None:
        datasets.append(('Noisy Q Chunk', stats_noisy_chunk))
    if stats_noisy_desc is not None:
        datasets.append(('Noisy Q Desc', stats_noisy_desc))
    
    if not datasets:
        print("No datasets available for comparison.")
        return
    
    # Print Found Percentage Summary
    print("\nFOUND PERCENTAGE SUMMARY")
    print('-' * 70)
    for name, stats in datasets:
        print(f"{name:<20}: {stats.get('found_percentage', 0):>8.2f}%")
    
    # Create the main comparison table
    print(f"\nDETAILED PERFORMANCE BY RANK (k)")
    print('=' * 140)
    
    # Build header
    header_parts = [f"{'k':<3}"]
    for dataset_name, _ in datasets:
        header_parts.extend([
            f"{'Top-k':>8}",
            f"{'NDCG@k':>8}"
        ])
    
    # Create separator line for dataset groups
    separator_parts = [f"{'':=<3}"]
    dataset_separator_parts = [f"{'':=<3}"]
    for i, (dataset_name, _) in enumerate(datasets):
        dataset_separator_parts.append(f"{dataset_name:=^17}")
        separator_parts.extend([f"{'':=<8}", f"{'':=<8}"])
        if i < len(datasets) - 1:
            separator_parts.append(f"{'':=<1}")
            dataset_separator_parts.append(f"{'':=<1}")
    
    print("".join(dataset_separator_parts))
    print("".join(header_parts))
    print("".join(separator_parts))
    
    # Print data rows
    for k in range(1, 11):
        row_parts = [f"{k:<3}"]
        for dataset_name, stats in datasets:
            top_k_val = stats.get(f'top_{k}', 0)
            ndcg_val = stats.get(f'NDCG@{k}', 0)
            row_parts.extend([
                f"{top_k_val:>7.2f}%",
                f"{ndcg_val:>8.4f}"
            ])
            if dataset_name != datasets[-1][0]:  # Add separator except for last dataset
                row_parts.append(f"{'':>1}")
        print("".join(row_parts))
    
    print('=' * 140)
    print("\nLEGEND:")
    print("• Top-k: Cumulative percentage of queries where target chunk is found in top k results")
    print("• NDCG@k: Normalized Discounted Cumulative Gain at rank k (higher is better)")
    print("• Noisy Q: Results with artificially noisy questions to test robustness")
    print('=' * 140)
    print()


def print_stats(directory: str):
    knowledge_graph_path = os.path.join(directory, 'knowledge_graph.json')
    with open(knowledge_graph_path, 'r') as f:
        knowledge_graph_dict = json.load(f)

    num_file_nodes = sum(1 for n in knowledge_graph_dict['nodes'] if n['class'] == 'FileNode')
    num_chunk_nodes = sum(1 for n in knowledge_graph_dict['nodes'] if n['class'] == 'ChunkNode')

    print('----------------------------------------------------------------------------------------------------------------------------------------')
    print(f'Generated knowledge graph : {knowledge_graph_path}')
    print('----------------------------------------------------------------------------------------------------------------------------------------')
    print(f"Number of files: {num_file_nodes}")
    print(f'Number of chunks : {num_chunk_nodes}')
    print(f'Number of edges : {len(knowledge_graph_dict["edges"])}')
    print('----------------------------------------------------------------------------------------------------------------------------------------')
    print('')

    generated_dataset_path = os.path.join(directory, 'dataset_single_chunk.json')
    with open(generated_dataset_path, 'r') as f:
        generated_dataset = json.load(f)
    df_generated_dataset = pd.DataFrame.from_dict(generated_dataset)

    print('----------------------------------------------------------------------------------------------------------------------------------------')
    print(f'Generated dataset : {generated_dataset_path}')
    print('----------------------------------------------------------------------------------------------------------------------------------------')
    print(f"Number of questions: {df_generated_dataset.shape[0]}")
    print('----------------------------------------------------------------------------------------------------------------------------------------')
    print('')

    # Initialize all stats variables to None
    stats_chunk = None
    stats_desc = None
    stats_noisy_chunk = None
    stats_noisy_desc = None

    # Chunk embedding results
    try:
        single_chunk_embed_result_path = os.path.join(directory, 'result_single_chunk_embed_chunk.json')
        with open(single_chunk_embed_result_path, 'r') as f:
            result_chunk = json.load(f)
        df_chunk = pd.DataFrame.from_dict(result_chunk)
        df_chunk[['found', 'order']] = df_chunk.apply(extract_found_and_order, axis=1)
        stats_chunk = compute_rank_stats(df_chunk)
        stats_chunk.update(compute_ndcg_stats_sklearn(df_chunk))

        print(f"✓ Loaded chunk embedding results: {single_chunk_embed_result_path}")
    except FileNotFoundError:
        print(f"✗ Chunk embedding results not found: result_single_chunk_embed_chunk.json")
    except Exception as e:
        print(f"✗ Error loading chunk embedding results: {e}")

    # Description embedding results
    try:
        single_description_embed_result_path = os.path.join(directory, 'result_single_chunk_embed_description.json')
        with open(single_description_embed_result_path, 'r') as f:
            result_desc = json.load(f)
        df_desc = pd.DataFrame.from_dict(result_desc)
        df_desc[['found', 'order']] = df_desc.apply(extract_found_and_order, axis=1)
        stats_desc = compute_rank_stats(df_desc)
        stats_desc.update(compute_ndcg_stats_sklearn(df_desc))
        print(f"✓ Loaded description embedding results: {single_description_embed_result_path}")
    except FileNotFoundError:
        print(f"✗ Description embedding results not found: result_single_chunk_embed_description.json")
    except Exception as e:
        print(f"✗ Error loading description embedding results: {e}")

    # Noisy question description embedding results
    try:
        noise_question_embed_description_result_path = os.path.join(directory,
                                                                    'result_noisy_question_single_chunk_embed_description.json')
        with open(noise_question_embed_description_result_path, 'r') as f:
            result_noisy_desc = json.load(f)
        df_noisy_desc = pd.DataFrame.from_dict(result_noisy_desc)

        # Use the noisy_results column and target_chunk column
        df_noisy_desc[['found', 'order']] = df_noisy_desc.apply(
            extract_found_and_order, axis=1,
            results_column_name='noisy_results_1',
            chunk_column_name='target_chunk'
        )

        stats_noisy_desc = compute_rank_stats(df_noisy_desc)
        stats_noisy_desc.update(compute_ndcg_stats_sklearn(df_noisy_desc))
        print(f"✓ Loaded noisy description embedding results: {noise_question_embed_description_result_path}")
    except FileNotFoundError:
        print(
            f"✗ Noisy description embedding results not found: result_noisy_question_single_chunk_embed_description.json")
    except KeyError as e:
        print(f"✗ Column issue in noisy description embedding results: {e}")
    except Exception as e:
        print(f"✗ Error loading noisy description embedding results: {e}")

    # Noisy question chunk embedding results
    try:
        noise_question_embed_chunk_result_path = os.path.join(directory,
                                                              'result_noisy_question_single_chunk_embed_chunk.json')
        with open(noise_question_embed_chunk_result_path, 'r') as f:
            result_noisy_chunk = json.load(f)
        df_noisy_chunk = pd.DataFrame.from_dict(result_noisy_chunk)


        # Use the noisy_results column and appropriate chunk column
        df_noisy_chunk[['found', 'order']] = df_noisy_chunk.apply(
            extract_found_and_order, axis=1,
            results_column_name='noisy_results_1',
            chunk_column_name='target_chunk'
        )

        stats_noisy_chunk = compute_rank_stats(df_noisy_chunk)
        stats_noisy_chunk.update(compute_ndcg_stats_sklearn(df_noisy_chunk))
        
        print(f"✓ Loaded noisy chunk embedding results: {noise_question_embed_chunk_result_path}")
    except FileNotFoundError:
        print(f"✗ Noisy chunk embedding results not found: result_noisy_question_single_chunk_embed_chunk.json")
    except KeyError as e:
        print(f"✗ Column issue in noisy chunk embedding results: {e}")
    except Exception as e:
        print(f"✗ Error loading noisy chunk embedding results: {e}")

    print('')

    # Print combined stats only if we have at least one set of results
    if any([stats_chunk, stats_desc, stats_noisy_chunk, stats_noisy_desc]):
        print_combined_stats(stats_chunk, stats_desc, stats_noisy_chunk, stats_noisy_desc)
    else:
        print("No results found to display statistics.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze retrieval performance stats.')
    parser.add_argument('directory', type=str, help='Directory containing dataset and result files')
    args = parser.parse_args()

    print_stats(args.directory)