import generate.knowledge_graph.run
import generate.single_chunk_single_question.run
import  generate.single_chunk_single_question_noisy_questions.run
import generate.knowledge_graph_several_models.run
import query.different_model_description_test.run 
import query.single_chunk_embed_code.run
import query.single_chunk_embed_summary.run
import  query.add_noise_questions_embed_summary.run
import  query.add_noise_questions_embed_chunk.run
import query.different_model_different_embedder.run


import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

# Configure logging for parallel execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


data_dir = os.path.expanduser('/mnt/workdir/lelkoussy/pedagogia-code-questions/data')

pipeline_steps = [
    {'step_name': 'generate knowledge graph', 'step_produces': 'knowledge_graph.json',
     'run_path': generate.knowledge_graph.run},
    {'step_name': 'generate single_chunk_single_question dataset',
     'step_produces': 'dataset_single_chunk.json',
     'run_path': generate.single_chunk_single_question.run}, 
     {'step_name': 'generate several model description', 
      'step_produces': 'dataset_several_model_description_single_chunk.json', 
      'run_path': generate.knowledge_graph_several_models.run
      },
          {'step_name': 'query single chunk embed code',
     'step_produces': "result_single_chunk_embed_chunk.json",
     'run_path': query.single_chunk_embed_code.run },
     
      {'step_name': 'query several model descriptions', 
       'step_produces': 'results_several_model_description.json', 
       'run_path':  query.different_model_description_test.run }, 
       {
        'step_name': 'baseline embedding using different embedder', 
        'step_produces': 'resuls_FAKE_FILE.json', 
        'run_path': query.different_model_different_embedder.run, 
       }
    
]

"""
      {'step_name': 'generate dataset_noisy_question_single_chunk_embed_chunk', 
      'run_path': generate.single_chunk_single_question_noisy_questions.run, 
      'step_produces': 'dataset_noisy_question_single_chunk_embed_chunk.json'},


    {'step_name': 'query single chunk embed summary',
     'step_produces': 'result_single_chunk_embed_description.json',
     'run_path': query.single_chunk_embed_summary.run},
    {
        'step_name': 'Add noise to questions single chunk embed description',
        'step_produces': 'result_noisy_question_single_chunk_embed_description.json',
        'run_path': query.add_noise_questions_embed_summary.run,
    },
{
        'step_name': 'Add noise to questions single chunk embed chunk',
        'step_produces': 'result_noisy_question_single_chunk_embed_chunk.json',
        'run_path': query.add_noise_questions_embed_chunk.run,
    }
    """
"""



        """

repo_dirs = ['/mnt/workdir/jperez/data/42sh-2025/epita-ing-assistants-acu-42sh-2025-lyon-20',
             '/mnt/workdir/jperez/data/42sh-2025/epita-ing-assistants-acu-42sh-2025-paris-50',
             '/mnt/workdir/jperez/data/42sh-2025/epita-ing-assistants-acu-42sh-2025-strasbourg-7',
            '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-lucas.llombart',
             '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-juliette.meyniel',
             '/mnt/workdir/jperez/data/js-workshop-2025/epita-ing-assistants-yaka-js-workshop-2025-yanis.belami',
             '/mnt/workdir/jperez/data/piscine-2025/epita-ing-assistants-acu-piscine-2025-yacine.boureghda', 
             '/mnt/workdir/jperez/data/piscine-2025/epita-ing-assistants-acu-piscine-2025-zoe.breniaux', 
             '/mnt/workdir/jperez/data/piscine-2025/epita-ing-assistants-acu-piscine-2025-virgile1.hermant'
             ]


def process_repository(repo_dir):
    """
    Process all pipeline steps for a single repository sequentially.
    Since steps have dependencies, they must run in order.
    """
    group_name = repo_dir.split('/')[-1]
    group_dir = os.path.join(data_dir, group_name)


    logger.info(f"Starting pipeline for repository: {group_name}")
    start_time = datetime.now()

    results = []

    for i, step in enumerate(pipeline_steps, 1):
        step_start_time = datetime.now()
        logger.info(f"[{group_name}] Step {i}/{len(pipeline_steps)}: {step['step_name']}")

        # Check if output already exists
        output_path = os.path.join(group_dir, step['step_produces'])
        if os.path.exists(output_path):
            logger.info(f"[{group_name}] Skipping step '{step['step_name']}' - output already exists")
            results.append({
                'repo': group_name,
                'step': step['step_name'],
                'status': 'SKIPPED',
                'message': 'Output already exists'
            })
            continue

        try:
            # Ensure output directory exists
            os.makedirs(group_dir, exist_ok=True)
            if i == 1:
                source_file_dir = os.path.join(repo_dir, 'src')
                if os.path.exists(source_file_dir):
                    repo_dir = source_file_dir

            # Run the step
            step['run_path'].run(repo_dir=repo_dir, data_dir=data_dir)

            step_duration = datetime.now() - step_start_time
            logger.info(f"[{group_name}] Completed step '{step['step_name']}' in {step_duration}")

            results.append({
                'repo': group_name,
                'step': step['step_name'],
                'status': 'COMPLETED',
                'duration': step_duration
            })

        except Exception as e:
            step_duration = datetime.now() - step_start_time
            logger.error(f"[{group_name}] ERROR in step '{step['step_name']}': {str(e)}")

            results.append({
                'repo': group_name,
                'step': step['step_name'],
                'status': 'ERROR',
                'error': str(e),
                'duration': step_duration
            })

            # Since steps have dependencies, stop processing this repo on error
            logger.warning(f"[{group_name}] Stopping pipeline due to error in step '{step['step_name']}'")
            break

    total_duration = datetime.now() - start_time
    logger.info(f"Finished pipeline for repository: {group_name} in {total_duration}")

    return {
        'repo': group_name,
        'total_duration': total_duration,
        'steps': results
    }


def run_parallel_pipeline(max_workers=None):
    """
    Run the pipeline across multiple repositories in parallel.
    Each repository processes its steps sequentially due to dependencies.
    """
    if max_workers is None:
        # Default to number of repositories or 4, whichever is smaller
        max_workers = min(len(repo_dirs), 4)

    logger.info(f"Starting parallel pipeline with {max_workers} workers")
    logger.info(f"Processing {len(repo_dirs)} repositories with {len(pipeline_steps)} steps each")

    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Pipeline") as executor:
        # Submit all repository processing tasks
        future_to_repo = {
            executor.submit(process_repository, repo_dir): repo_dir
            for repo_dir in repo_dirs
        }

        all_results = []
        completed_count = 0

        for future in as_completed(future_to_repo):
            repo_dir = future_to_repo[future]
            group_name = repo_dir.split('/')[-1]

            try:
                result = future.result()
                all_results.append(result)
                completed_count += 1

                logger.info(f"Repository {group_name} completed ({completed_count}/{len(repo_dirs)})")

            except Exception as exc:
                logger.error(f"Repository {group_name} generated an exception: {exc}")
                all_results.append({
                    'repo': group_name,
                    'total_duration': None,
                    'steps': [],
                    'error': str(exc)
                })

    total_duration = datetime.now() - start_time
    logger.info(f"All repositories completed in {total_duration}")

    return all_results


def print_summary(results):
    """Print a summary of the pipeline execution"""
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)

    for result in results:
        repo_name = result['repo']
        if 'error' in result:
            logger.error(f"{repo_name}: FAILED - {result['error']}")
            continue

        total_duration = result.get('total_duration', 'Unknown')
        logger.info(f"\n{repo_name}: {total_duration}")

        for step_result in result['steps']:
            status = step_result['status']
            step_name = step_result['step']

            if status == 'COMPLETED':
                duration = step_result.get('duration', 'Unknown')
                logger.info(f"  ✓ {step_name}: {duration}")
            elif status == 'SKIPPED':
                logger.info(f"  - {step_name}: SKIPPED")
            elif status == 'ERROR':
                error = step_result.get('error', 'Unknown error')
                logger.error(f"  ✗ {step_name}: ERROR - {error}")


if __name__ == "__main__":
    # You can adjust max_workers based on your system capacity
    # For I/O bound tasks, you can typically use more workers than CPU cores
    results = run_parallel_pipeline(max_workers=1)

    print_summary(results)
