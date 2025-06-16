import os.path
import argparse
from RepoKnowledgeGraph import RepoKnowledgeGraph


def print_tree(directory: str):
    knowledge_graph_dict_path = os.path.join(directory, 'knowledge_graph.json')
    knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(knowledge_graph_dict_path, index_nodes=False)
    knowledge_graph.print_tree()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze retrieval performance stats.')
    parser.add_argument('directory', type=str, help='Directory containing dataset and result files')
    args = parser.parse_args()

    print_tree(args.directory)