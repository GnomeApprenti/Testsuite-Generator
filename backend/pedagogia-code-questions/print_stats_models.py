import pandas as pd
import json
import os.path
import argparse
import numpy as np 
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def extract_found_and_order_single(row, results_column, chunk_column_name='chunk'):
    """
    Extract found and order for a single results column
    """
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
    
    # Check if the results column exists and has the expected structure
    if results_column not in row or pd.isna(row[results_column]):
        return {'found': False, 'order': None}
    
    try:
        id_list = row[results_column]['ids'][0]  # The nested list of IDs
        if target_id in id_list:
            return {'found': True, 'order': id_list.index(target_id)}
        else:
            return {'found': False, 'order': None}
    except (KeyError, TypeError, IndexError):
        # Handle cases where the structure is unexpected
        return {'found': False, 'order': None}


def compute_description_length_stats(dataframe, description_column):
    """
    Compute average length statistics for description columns
    """
    if description_column not in dataframe.columns:
        return {'avg_description_length': 0.0}
    
    # Filter out NaN values and compute lengths
    valid_descriptions = dataframe[description_column].dropna()
    
    if len(valid_descriptions) == 0:
        return {'avg_description_length': 0.0}
    
    # Calculate lengths for each description
    lengths = valid_descriptions.apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    avg_length = lengths.mean()
    
    return {'avg_description_length': avg_length}


def extract_all_result_column_names(df, chunk_column_name='chunk') -> list:
    """
    Process all result columns dynamically and add found/order columns
    """
    # Find all columns that start with 'result_'
    result_columns = [col for col in df.columns if col.startswith('result_')]
    
    print(f"Found {len(result_columns)} result columns:")
    for col in result_columns:
        print(f"  - {col}")
    
    # Process each result column
    for result_col in result_columns:
        # Extract the suffix (everything after 'result_')
        suffix = result_col[7:]  # Remove 'result_' prefix
        
        # Create column names for found and order
        found_col = f'found_{suffix}'
        order_col = f'order_{suffix}'
        
        print(f"Processing {result_col} -> {found_col}, {order_col}")
        
        # Apply the extraction function to each row
        results = df.apply(
            lambda row: extract_found_and_order_single(row, result_col, chunk_column_name), 
            axis=1
        )
        
        # Extract found and order values from the results
        df[found_col] = [r['found'] for r in results]
        df[order_col] = [r['order'] for r in results]
    
    return result_columns


def compute_rank_stats(dataframe, found_col, order_col):
    """
    Compute rank statistics for a specific found/order column pair
    """
    stats = {}
    found_percentage = dataframe[found_col].mean() * 100
    stats['found_percentage'] = found_percentage

    found_df = dataframe[dataframe[found_col]]
    if len(found_df) > 0:
        rank_counts = found_df[order_col].value_counts().sort_index()
        rank_percentages = (rank_counts / len(dataframe)) * 100

        cumulative = 0
        for rank in range(10):  # top 1 to 10
            pct = rank_percentages.get(rank, 0)
            cumulative += pct
            stats[f'top_{rank + 1}'] = cumulative
    else:
        # No items found, all top_k values are 0
        for rank in range(10):
            stats[f'top_{rank + 1}'] = 0.0

    return stats


def compute_ndcg_stats_sklearn(dataframe, found_col, order_col, k=10):
    """
    Compute NDCG@k scores using scikit-learn's ndcg_score.
    
    For retrieval evaluation, each row represents a separate query, and we need to
    compute NDCG per query, then average across all queries.
    
    Parameters:
    dataframe: pandas DataFrame 
    found_col: name of the 'found' column (bool)
    order_col: name of the 'order' column (int, 0-based ranking)
    k: maximum k value for NDCG@k calculations
    
    Returns:
    dict: NDCG@k scores for k=1 to k
    """
    ndcg_stats = {}
    
    # For retrieval, each row is a separate query
    # We need to compute NDCG for each query individually, then average
    
    num_queries = len(dataframe)
    if num_queries == 0:
        for rank in range(1, k + 1):
            ndcg_stats[f'NDCG@{rank}'] = 0.0
        return ndcg_stats
    
    # Calculate NDCG@k for each k from 1 to k
    for rank in range(1, k + 1):
        ndcg_scores = []
        
        for idx, row in dataframe.iterrows():
            # For each query (row), we simulate a ranking list
            found = row[found_col]
            order = row[order_col]
            
            if found and pd.notna(order):
                # Create a simulated ranking list of size k
                # The target item is at position 'order', others are irrelevant
                y_true = np.zeros(max(rank, int(order) + 1))
                y_true[int(order)] = 1  # Target item has relevance 1
                
                # Scores: items at earlier positions get higher scores
                y_score = np.arange(len(y_true), 0, -1, dtype=float)
                
                # Reshape for sklearn (expects 2D arrays)
                y_true = y_true.reshape(1, -1)
                y_score = y_score.reshape(1, -1)
                
                try:
                    ndcg = ndcg_score(y_true, y_score, k=rank)
                    ndcg_scores.append(ndcg)
                except ValueError:
                    # If NDCG can't be computed, treat as 0
                    ndcg_scores.append(0.0)
            else:
                # Target not found in top results, NDCG = 0
                ndcg_scores.append(0.0)
        
        # Average NDCG across all queries
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        ndcg_stats[f'NDCG@{rank}'] = avg_ndcg
    
    return ndcg_stats


def print_individual_model_stats(model_name, stats):
    """
    Print detailed stats for a single model
    """
    print('=' * 80)
    print(f'MODEL: {model_name}')
    print('=' * 80)
    print(f"Found Percentage: {stats.get('found_percentage', 0):.2f}%")
    print()
    
    print(f"{'Rank (k)':<8} {'Top-k %':<10} {'NDCG@k':<10}")
    print('-' * 30)
    
    for k in range(1, 11):
        top_k_val = stats.get(f'top_{k}', 0)
        ndcg_val = stats.get(f'NDCG@{k}', 0)
        print(f"{k:<8} {top_k_val:<9.2f}% {ndcg_val:<10.4f}")
    
    print()


def print_summary_comparison(all_stats):
    """
    Print a summary comparison table of key metrics
    """
    if not all_stats:
        return
        
    print('=' * 120)
    print('SUMMARY COMPARISON - KEY METRICS')
    print('=' * 120)
    
    print(f"{'Model':<40} {'Found %':<10} {'Top-1 %':<10} {'Top-5 %':<10} {'Top-10 %':<10} {'NDCG@10':<10}")
    print('-' * 120)
    
    # Sort models by found percentage for better readability
    sorted_models = sorted(all_stats.items(), key=lambda x: x[1].get('found_percentage', 0), reverse=True)
    
    for model_name, stats in sorted_models:
        found_pct = stats.get('found_percentage', 0)
        top_1 = stats.get('top_1', 0)
        top_5 = stats.get('top_5', 0)
        top_10 = stats.get('top_10', 0)
        ndcg_10 = stats.get('NDCG@10', 0)
        
        print(f"{model_name[:39]:<40} {found_pct:<9.2f}% {top_1:<9.2f}% {top_5:<9.2f}% {top_10:<9.2f}% {ndcg_10:<10.4f}")
    
    print('=' * 120)
    print()


def group_models_by_base_model(all_stats):
    """
    Group models by their base model name for organized display
    """
    groups = {}
    
    for model_name, stats in all_stats.items():
        # Extract base model name (everything after the embedding type prefix)
        if '] ' in model_name:
            base_model = model_name.split('] ')[1]
        else:
            base_model = model_name
            
        if base_model not in groups:
            groups[base_model] = []
        groups[base_model].append((model_name, stats))
    
    return groups


def print_grouped_stats(all_stats):
    """
    Print stats grouped by base model
    """

    # Step 1: Flatten the JSON into rows
    rows = []
    for model, tasks in all_stats.items():
        for task, embeddings in tasks.items():
            for embedding, metrics in embeddings.items():
                for metric_name, value in metrics.items():
                    rows.append({
                        "Model": model,
                        "Task": task,
                        "Embedding": embedding.strip(),
                        "Metric": metric_name,
                        "Value": float(value)
                    })

    # Step 2: Create a DataFrame
    df = pd.DataFrame(rows)

    # Step 3: Separate out metrics for better visualization
    ndcg_df = df[df['Metric'].str.startswith('NDCG@')].copy()
    topk_df = df[df['Metric'].str.startswith('top_')].copy()
    found_df = df[df['Metric'] == 'found_percentage'].copy()
    ndcg_10_df = df[df['Metric'].str.startswith('NDCG@10')].copy()
    desc_length_df = df[df['Metric'] == 'avg_description_length'].copy()

    # Step 4: Pivot tables
    # NDCG table
    ndcg_pivot = ndcg_df.pivot_table(index=["Task", "Metric"], columns=["Model", "Embedding"], values="Value")

    # Top-k table
    topk_pivot = topk_df.pivot_table(index=["Task", "Metric"], columns=["Model", "Embedding"], values="Value")

    # Found % table
    found_pivot = found_df.pivot_table(index=["Task"], columns=["Model", "Embedding"], values="Value")

    ndcg_10_pivot = ndcg_10_df.pivot_table(index=["Task"], columns=["Model", "Embedding"], values="Value")
    
    # Description length table
    desc_length_pivot = desc_length_df.pivot_table(index=["Task"], columns=["Model", "Embedding"], values="Value")

    # Optional: round for better readability
    ndcg_pivot = ndcg_pivot.round(3)
    topk_pivot = topk_pivot.round(2)
    found_pivot = found_pivot.round(2)
    ndcg_10_pivot = ndcg_10_pivot.round(3)
    desc_length_pivot = desc_length_pivot.round(1)

    # Display sample output
    print("🔹 NDCG Scores:")
    print(ndcg_pivot.to_string())
    print("\n🔹 Top-k Accuracy:")
    print(topk_pivot.to_string())
    print("\n🔹🔹🔹🔹 Summaries 🔹🔹🔹🔹")
    print("\n🔹 top-10 ")
    print(found_pivot.to_string())
    print('')
    print("\n🔹 Average NDCG@10")
    print(ndcg_10_pivot.to_string())
    print('')
    print("\n🔹 Average Description Length (characters)")
    print(desc_length_pivot.to_string())


def create_line_charts(all_stats, output_dir):
    """
    Create line charts for metrics across different k values
    """
    print("\n🔹 Creating visualization charts...")
    
    # Prepare data for visualization
    viz_data = []
    for model, tasks in all_stats.items():
        for task, embeddings in tasks.items():
            for embedding, metrics in embeddings.items():
                # Extract Top-k metrics
                for k in range(1, 11):
                    top_k_val = metrics.get(f'top_{k}', 0)
                    viz_data.append({
                        'Model': model,
                        'Task': task,
                        'Embedding': embedding.strip(),
                        'k': k,
                        'Top_k': top_k_val,
                        'Metric_Type': 'Top-k Accuracy (%)'
                    })
                
                # Extract NDCG metrics
                for k in range(1, 11):
                    ndcg_val = metrics.get(f'NDCG@{k}', 0)
                    viz_data.append({
                        'Model': model,
                        'Task': task,
                        'Embedding': embedding.strip(),
                        'k': k,
                        'NDCG': ndcg_val,
                        'Metric_Type': 'NDCG'
                    })
    
    df_viz = pd.DataFrame(viz_data)
    
    if df_viz.empty:
        print("No data available for visualization.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get unique tasks and create subplots
    unique_tasks = df_viz['Task'].unique()
    n_tasks = len(unique_tasks)
    
    # Create figure with subplots for each task
    fig = plt.figure(figsize=(20, 6 * n_tasks))
    gs = GridSpec(n_tasks, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, task in enumerate(unique_tasks):
        task_data = df_viz[df_viz['Task'] == task]
        
        # Top-k Accuracy chart
        ax1 = fig.add_subplot(gs[i, 0])
        
        # Group by Model and Embedding for line plotting
        for (model, embedding), group in task_data.groupby(['Model', 'Embedding']):
            top_k_data = group[group['Metric_Type'] == 'Top-k Accuracy (%)']
            if not top_k_data.empty:
                label = f"{model} - {embedding}"
                ax1.plot(top_k_data['k'], top_k_data['Top_k'], 
                        marker='o', linewidth=2, markersize=4, label=label)
        
        ax1.set_xlabel('k (Rank)', fontsize=12)
        ax1.set_ylabel('Top-k Accuracy (%)', fontsize=12)
        ax1.set_title(f'Top-k Accuracy - {task}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.set_xticks(range(1, 11))
        
        # NDCG chart
        ax2 = fig.add_subplot(gs[i, 1])
        
        for (model, embedding), group in task_data.groupby(['Model', 'Embedding']):
            ndcg_data = group[group['Metric_Type'] == 'NDCG']
            if not ndcg_data.empty:
                label = f"{model} - {embedding}"
                ax2.plot(ndcg_data['k'], ndcg_data['NDCG'], 
                        marker='s', linewidth=2, markersize=4, label=label)
        
        ax2.set_xlabel('k (Rank)', fontsize=12)
        ax2.set_ylabel('NDCG@k', fontsize=12)
        ax2.set_title(f'NDCG@k - {task}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.set_xticks(range(1, 11))
    
    plt.tight_layout()
    
    # Save the plot
    chart_path = os.path.join(output_dir, 'retrieval_performance_charts.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Charts saved to: {chart_path}")
    
    # Create separate charts for each metric type (for better readability)
    create_separate_metric_charts(df_viz, output_dir)
    
    plt.show()


def create_separate_metric_charts(df_viz, output_dir):
    """
    Create separate charts for each metric type and task combination,
    and add comparison charts for the best-performing model+embedding per task.
    """
    unique_tasks = df_viz['Task'].unique()
    
    # === TOP-K ACCURACY ===
    fig, axes = plt.subplots(1, len(unique_tasks), figsize=(8 * len(unique_tasks), 6))
    if len(unique_tasks) == 1:
        axes = [axes]
        
    best_topk_combos = []

    for i, task in enumerate(unique_tasks):
        task_data = df_viz[(df_viz['Task'] == task) & (df_viz['Metric_Type'] == 'Top-k Accuracy (%)')]
        
        for (model, embedding), group in task_data.groupby(['Model', 'Embedding']):
            label = f"{model}\n{embedding}"
            axes[i].plot(group['k'], group['Top_k'], 
                         marker='o', linewidth=3, markersize=6, label=label)
        
        axes[i].set_xlabel('k (Rank)', fontsize=12)
        axes[i].set_ylabel('Top-k Accuracy (%)', fontsize=12)
        axes[i].set_title(f'Top-k Accuracy\n{task}', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10)
        axes[i].set_xticks(range(1, 11))
        axes[i].set_ylim(0, 100)

        # Find best performing combination at max k
        max_k = task_data['k'].max()
        best_row = task_data[task_data['k'] == max_k].sort_values(by='Top_k', ascending=False).iloc[0]
        best_group = task_data[(task_data['Model'] == best_row['Model']) & 
                               (task_data['Embedding'] == best_row['Embedding'])]
        best_topk_combos.append((task, best_row['Model'], best_row['Embedding'], best_group))

    plt.tight_layout()
    topk_path = os.path.join(output_dir, 'top_k_accuracy_charts.png')
    plt.savefig(topk_path, dpi=300, bbox_inches='tight')
    print(f"📊 Top-k charts saved to: {topk_path}")
    
    # === NDCG ===
    fig, axes = plt.subplots(1, len(unique_tasks), figsize=(8 * len(unique_tasks), 6))
    if len(unique_tasks) == 1:
        axes = [axes]
    
    best_ndcg_combos = []

    for i, task in enumerate(unique_tasks):
        task_data = df_viz[(df_viz['Task'] == task) & (df_viz['Metric_Type'] == 'NDCG')]
        
        for (model, embedding), group in task_data.groupby(['Model', 'Embedding']):
            label = f"{model}\n{embedding}"
            axes[i].plot(group['k'], group['NDCG'], 
                         marker='s', linewidth=3, markersize=6, label=label)
        
        axes[i].set_xlabel('k (Rank)', fontsize=12)
        axes[i].set_ylabel('NDCG@k', fontsize=12)
        axes[i].set_title(f'NDCG@k\n{task}', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10)
        axes[i].set_xticks(range(1, 11))
        axes[i].set_ylim(0, 1)

        # Best NDCG performer
        max_k = task_data['k'].max()
        best_row = task_data[task_data['k'] == max_k].sort_values(by='NDCG', ascending=False).iloc[0]
        best_group = task_data[(task_data['Model'] == best_row['Model']) & 
                               (task_data['Embedding'] == best_row['Embedding'])]
        best_ndcg_combos.append((task, best_row['Model'], best_row['Embedding'], best_group))

    plt.tight_layout()
    ndcg_path = os.path.join(output_dir, 'ndcg_charts.png')
    plt.savefig(ndcg_path, dpi=300, bbox_inches='tight')
    print(f"📊 NDCG charts saved to: {ndcg_path}")
    
    # === Best-of-Best Comparison Charts ===
    # TOP-K Accuracy Best Comparison
    plt.figure(figsize=(10, 6))
    for task, model, embedding, group in best_topk_combos:
        label = f"{task}: {model}\n{embedding}"
        plt.plot(group['k'], group['Top_k'], marker='o', linewidth=3, markersize=6, label=label)
    plt.xlabel('k (Rank)', fontsize=12)
    plt.ylabel('Top-k Accuracy (%)', fontsize=12)
    plt.title('Best Top-k Accuracy per Task', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(range(1, 11))
    plt.ylim(0, 100)
    plt.tight_layout()
    comp_topk_path = os.path.join(output_dir, 'best_top_k_comparison.png')
    plt.savefig(comp_topk_path, dpi=300, bbox_inches='tight')
    print(f"📈 Comparison chart (Top-k) saved to: {comp_topk_path}")

    # NDCG Best Comparison
    plt.figure(figsize=(10, 6))
    for task, model, embedding, group in best_ndcg_combos:
        label = f"{task}: {model}\n{embedding}"
        plt.plot(group['k'], group['NDCG'], marker='s', linewidth=3, markersize=6, label=label)
    plt.xlabel('k (Rank)', fontsize=12)
    plt.ylabel('NDCG@k', fontsize=12)
    plt.title('Best NDCG@k per Task', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(range(1, 11))
    plt.ylim(0, 1)
    plt.tight_layout()
    comp_ndcg_path = os.path.join(output_dir, 'best_ndcg_comparison.png')
    plt.savefig(comp_ndcg_path, dpi=300, bbox_inches='tight')
    print(f"📈 Comparison chart (NDCG) saved to: {comp_ndcg_path}")



def create_heatmaps(all_stats, output_dir):
    """
    Create heatmaps for key metrics comparison
    """
    print("\n🔹 Creating heatmap visualizations...")
    
    # Prepare data for heatmaps
    heatmap_data = []
    for model, tasks in all_stats.items():
        for task, embeddings in tasks.items():
            for embedding, metrics in embeddings.items():
                heatmap_data.append({
                    'Model': model,
                    'Task': task,
                    'Embedding': embedding.strip(),
                    'Found_Percentage': metrics.get('found_percentage', 0),
                    'Top_1': metrics.get('top_1', 0),
                    'Top_5': metrics.get('top_5', 0),
                    'Top_10': metrics.get('top_10', 0),
                    'NDCG_5': metrics.get('NDCG@5', 0),
                    'NDCG_10': metrics.get('NDCG@10', 0)
                })
    
    df_heatmap = pd.DataFrame(heatmap_data)
    
    if df_heatmap.empty:
        print("No data available for heatmap visualization.")
        return
    
    # Create a combined identifier for better visualization
    df_heatmap['Model_Task_Embedding'] = (df_heatmap['Model'] + '\n' + 
                                         df_heatmap['Task'] + '\n' + 
                                         df_heatmap['Embedding'])
    
    # Create heatmap for key metrics
    metrics_to_plot = ['Found_Percentage', 'Top_1', 'Top_5', 'Top_10', 'NDCG_5', 'NDCG_10']
    heatmap_matrix = df_heatmap.set_index('Model_Task_Embedding')[metrics_to_plot]
    
    plt.figure(figsize=(12, max(8, len(heatmap_matrix) * 0.5)))
    sns.heatmap(heatmap_matrix, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Performance Score'})
    plt.title('Performance Heatmap: Key Metrics Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Model - Task - Embedding', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'performance_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"📊 Heatmap saved to: {heatmap_path}")
    plt.show()


def extract_model_name_from_suffix(suffix:str):
    """
    Extract a readable model name from the result column suffix
    """
    # Identify the embedding type prefix
    embedding_type = ""
    remaining_suffix = suffix

    if suffix.startswith('description_thought_'):
        embedding_type = "[Desc+Thought]"
        remaining_suffix = suffix[20:]  # Remove 'description_thought_' prefix
    elif suffix.startswith('description_'):
        embedding_type = "[Description]"
        remaining_suffix = suffix[12:]  # Remove 'description_' prefix
    elif suffix.startswith('thought_'):
        embedding_type = "[Thought]"
        remaining_suffix = suffix[8:]  # Remove 'thought_' prefix
    else:
        embedding_type = suffix.split('_')[0]
        remaining_suffix = suffix.replace(embedding_type, '')
        embedding_type = f'[{embedding_type}]'

    task_name = remaining_suffix.split('_')[-1]
    model_name = remaining_suffix.replace('_' +task_name, '')

    # Replace common patterns to make model names shorter and more readable
    model_name = model_name.replace('meta-llama/', 'Meta-')
    model_name = model_name.replace('Llama-3.2-3B-Instruct', 'Llama3.2-3B')
    model_name = model_name.replace('Qwen/Qwen3-1.7B', 'Qwen3-1.7B')
    model_name = model_name.replace('google/gemma-3', 'Gemma-3')
    model_name = model_name.replace('microsoft/Phi-4', 'Phi-4')
    model_name = model_name.replace('_no_model', 'NoModel')
    model_name = model_name.replace('_Salesforce/SFR-Embedding-Code', 'SalesForce')
    task_name = task_name.replace('_Summarizing', ' (Sum)')
    task_name = task_name.replace('_Reviewing', ' (Rev)')
    task_name = task_name.replace('_Completion', ' (Comp)')

    return embedding_type, task_name, model_name


def find_corresponding_description_column(result_suffix, dataframe_columns):
    """
    Find the description column that corresponds to a result column
    """
    # The result column has format: result_{embedding_type}_{model}_{task}
    # The description column should have format: {embedding_type}_{model}_{task}
    # So we just need to remove the 'result_' prefix
    description_col = result_suffix
    
    if description_col in dataframe_columns:
        return description_col
    
    # If direct match doesn't work, try some common variations
    # Join the last two parts with underscore to create a proper suffix string
    suffix_parts = result_suffix.split('_')
    if len(suffix_parts) >= 2:
        suffix_to_match = '_'.join(suffix_parts[-2:])
        possible_columns = [col for col in dataframe_columns if col.endswith(suffix_to_match)]
        if possible_columns:
            return possible_columns[0]
    
    return None


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

    # Load the multi-model results
    try:
        results_path = os.path.join(directory, 'results_several_model_description.json')
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        df_results = pd.DataFrame.from_dict(results_data, orient='index')
        
        print(f"✓ Loaded multi-model results: {results_path}")
        print(f"  Dataset shape: {df_results.shape}")
        print(f"  Columns: {list(df_results.columns)}")
        print('')
        
        # Process all result columns to create found/order columns
        result_columns = extract_all_result_column_names(df_results, chunk_column_name='chunk')
        
        if not result_columns:
            print("✗ No result columns found in the dataset.")
            return
            
        print('')
        
        # Compute stats for each model
        all_stats = {}

        for result_col in result_columns:
            suffix = result_col[7:]  # Remove 'result_' prefix
            found_col = f'found_{suffix}'
            order_col = f'order_{suffix}'

            # Create a readable model name
            embedding_type, task_name, model_name = extract_model_name_from_suffix(suffix)

            print(f"Computing stats for {model_name}...")

            # Compute basic rank stats
            stats = compute_rank_stats(df_results, found_col, order_col)

            # Compute NDCG stats
            ndcg_stats = compute_ndcg_stats_sklearn(df_results, found_col, order_col)
            stats.update(ndcg_stats)
            
            # Find and compute description length stats
            description_col = find_corresponding_description_column(suffix, df_results.columns)
            if description_col:
                desc_stats = compute_description_length_stats(df_results, description_col)
                stats.update(desc_stats)
                print(f"  Added description length stats for column: {description_col}")
            else:
                print(f"  Warning: Could not find description column for {suffix}")
                stats['avg_description_length'] = 0.0
            
            if model_name not in all_stats:
                all_stats[model_name] = {
                    task_name: {
                        embedding_type: stats
                    }
                }
            else:
                if task_name not in all_stats[model_name]:
                    all_stats[model_name][task_name] = {embedding_type: stats}
                else:
                    all_stats[model_name][task_name][embedding_type] = stats
        
        print('')
        
        # Print combined comparison
        if all_stats:
            print_grouped_stats(all_stats)
            
            # Create visualizations
            print("\n" + "="*80)
            print("CREATING VISUALIZATIONS")
            print("="*80)
            
            # Create line charts for metrics across k values
            create_line_charts(all_stats, directory)
            
            # Create heatmaps for key metrics comparison
            create_heatmaps(all_stats, directory)
            
            print(f"\n📊 All visualizations saved in directory: {directory}")
            
        else:
            print("No statistics computed.")
            
    except FileNotFoundError:
        print(f"✗ Multi-model results not found: results_several_model_description.json")
    except Exception as e:
        print(f"✗ Error loading multi-model results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze retrieval performance stats for multiple models.')
    parser.add_argument('directory', type=str, help='Directory containing dataset and result files')
    args = parser.parse_args()

    print_stats(args.directory)