import logging
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

from .ModelService import ModelService
from .Node import Node, DirectoryNode, FileNode, ChunkNode
from .CodeParser import CodeParser
from .EntityExtractor import extract_entities, get_language_from_filename
from .CodeIndex import CodeIndex
from .utils.logger_utils import setup_logger
from .utils.parsing_utils import read_directory_files_recursively
from .utils.path_utils import prepare_input_path

LOGGER_NAME = 'REPO_KNOWLEDGE_GRAPH_LOGGER'


# A RepoKnowledgeGraph is a weighted DAG based on a tree-structure with added edges
class RepoKnowledgeGraph:
    def __init__(self, path: str, index_nodes:bool=True):
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)
        self.code_parser = CodeParser()
        self.model_service = ModelService()
        self.nodes = {0: [],  # Level 0 contains root node (representing all the repo)
                      1: [],  # Level 1 contains file nodes
                      2: []}  # Level 2 contains chunk nodes
        # Create a directed graph
        self.graph = nx.DiGraph()
        prepared_path = prepare_input_path(path)
        self._initial_parse_path(prepared_path)
        self._build_relationships()
        if index_nodes:
            self.code_index = CodeIndex(list(self))



    def __iter__(self):
        # Yield only the 'data' attribute from each node
        return (node_data['data'] for _, node_data in self.graph.nodes(data=True))

    def __getitem__(self, node_id):
        return self.graph.nodes[node_id]['data']


    @classmethod
    def from_path(cls, path: str, index_nodes:bool=True):
        self = cls.__new__(cls)  # Create instance without calling __init__
        setup_logger(LOGGER_NAME)
        self.model_service = ModelService()
        self.logger = logging.getLogger(LOGGER_NAME)
        self.code_parser = CodeParser()
        self.nodes = {0: [], 1: [], 2: []}
        self.graph = nx.DiGraph()
        prepared_path = prepare_input_path(path)
        self._initial_parse_path(prepared_path)
        self._build_relationships()
        if index_nodes:
            self.code_index = CodeIndex(list(self))
        return self

    def _initial_parse_path(self, path: str):
        root_node = Node(id='root', name='root', node_type='root')
        self.graph.add_node('root', data=root_node, level=0)

        level1_node_contents = read_directory_files_recursively(path, skip_pattern=r"(?:\.log$|\.json$|(?:^|/)(?:\.git|\.idea|__pycache__|\.cache)(?:/|$)|(?:^|/)(?:changelog|ChangeLog)(?:\.[a-z0-9]+)?$|\.cache$)")

        for file_path in level1_node_contents:
            print(file_path)
            full_path = os.path.normpath(file_path)
            parts = full_path.split(os.sep)
            current_parent = 'root'
            path_accum = ''

            # Create intermediate directory nodes
            for part in parts[:-1]:  # Skip file itself
                path_accum = os.path.join(path_accum, part) if path_accum else part
                if path_accum not in self.graph:
                    dir_node = DirectoryNode(id=path_accum, name=part, path=path_accum)
                    self.graph.add_node(path_accum, data=dir_node, level=1)
                    self.graph.add_edge(current_parent, path_accum, relation='contains')
                current_parent = path_accum

            # Now handle the file itself
            try:
                language = get_language_from_filename(file_path)
            except KeyError as e:
                self.logger.error(f'Unable to get language from filename: {e}')
                self.logger.error('Returning raw language extension')
                language = file_path.split('.')[-1]
            parsed_content = self.code_parser.parse(file_name=file_path, file_content=level1_node_contents[file_path])

            file_declared_entities = {}
            file_called_entities = set()
            chunk_ids = []

            for i, chunk in enumerate(parsed_content):
                chunk_id = f'{file_path}_{i}'
                chunk_ids.append(chunk_id)

                declared_entities, called_entities = extract_entities(code=chunk, file_name=file_path)
                description = self.model_service.query(
                    f'Summarize this {language} code chunk in a few sentences: {chunk}')
                chunk_node = ChunkNode(
                    id=chunk_id,
                    name=chunk_id,
                    path=file_path,
                    content=chunk,
                    order_in_file=i,
                    called_entities=called_entities,
                    defined_entities=declared_entities,
                    language=language,
                    description=description,
                )
                embedding = self.model_service.embed(text_to_embed=chunk_node.get_field_to_embed())
                chunk_node.embedding = embedding

                self.graph.add_node(chunk_id, data=chunk_node, level=2)

                for entity_type in declared_entities:
                    file_declared_entities.setdefault(entity_type, []).extend(declared_entities[entity_type])

                file_called_entities.update(called_entities)

            file_node = FileNode(
                id=file_path,
                name=parts[-1],
                path=file_path,
                node_type='file',
                content=level1_node_contents[file_path],
                defined_entities=file_declared_entities,
                called_entities=list(file_called_entities),
                language=language,
            )

            self.graph.add_node(file_path, data=file_node, level=1)
            self.graph.add_edge(current_parent, file_path, relation='contains')

            for chunk_id in chunk_ids:
                self.graph.add_edge(file_path, chunk_id, relation='contains')

    def _build_relationships(self):
        # Get all nodes that are either files or chunks
        nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if 'data' in node_data and isinstance(node_data['data'], (FileNode, ChunkNode)):
                nodes.append((node_id, node_data['data']))

        for caller_id, caller_node in nodes:
            for callee_id, callee_node in nodes:
                if caller_id == callee_id:
                    continue  # Skip self-loops

                # Skip if a 'contains' relationship exists between the nodes in either direction
                if self.graph.has_edge(caller_id, callee_id) and self.graph.edges[caller_id, callee_id].get(
                        'relation') == 'contains':
                    continue
                if self.graph.has_edge(callee_id, caller_id) and self.graph.edges[callee_id, caller_id].get(
                        'relation') == 'contains':
                    continue

                # Flatten callee defined entities into a set
                callee_defined = {
                    entity
                    for entity_list in callee_node.defined_entities.values()
                    for entity in entity_list
                }

                # Check for intersection between caller's called entities and callee's defined entities
                intersection = set(caller_node.called_entities) & callee_defined
                if intersection:
                    self.graph.add_edge(caller_id, callee_id, relation='calls')
                    self.graph.add_edge(callee_id, caller_id, relation='called_by')

    def print_tree(self, max_depth=None, start_node_id='root', level=0, prefix=""):
        """
        Print the repository tree structure using the graph with 'contains' edges.

        Args:
            max_depth (int, optional): Maximum depth to print. None = unlimited.
            start_node_id (str): ID of the node to start from. Default is 'root'.
            level (int): Internal use only (used for recursion).
            prefix (str): Internal use only (used for formatting output).
        """
        if max_depth is not None and level > max_depth:
            return

        if start_node_id not in self.graph:
            self.logger.warning(f"Start node '{start_node_id}' not found in graph.")
            return


        try:
            node_data = self[start_node_id]
        except KeyError as e:
            self.logger.error(f"KeyError when accessing node {start_node_id}: {e}")
            self.logger.error(f"Available node attributes: {list(self.graph.nodes[start_node_id].keys())}")
            # Use a fallback approach if 'data' is missing
            if 'data' not in self.graph.nodes[start_node_id]:
                self.logger.warning(f"Node {start_node_id} has no 'data' attribute, using node itself")
                # Create a fallback node if 'data' is missing
                if start_node_id == 'root':
                    # Create a default root node
                    node_data = Node(id='root', name='root', node_type='root')
                    # Update the graph node with the fallback data
                    self.graph.nodes[start_node_id]['data'] = node_data
                else:
                    # Try to infer node type from ID or structure
                    name = start_node_id.split('/')[-1] if '/' in start_node_id else start_node_id
                    if '_' in start_node_id and start_node_id.split('_')[-1].isdigit():
                        # Looks like a chunk ID
                        node_data = ChunkNode(id=start_node_id, name=name, node_type='chunk')
                    elif '.' in name:
                        # Looks like a file
                        node_data = FileNode(id=start_node_id, name=name, node_type='file', path=start_node_id)
                    else:
                        # Fallback to directory or generic node
                        node_data = DirectoryNode(id=start_node_id, name=name, node_type='directory',
                                                  path=start_node_id)
                    # Update the graph node with the fallback data
                    self.graph.nodes[start_node_id]['data'] = node_data
            return

        # Choose icon based on node type
        if node_data.node_type == 'file':
            node_symbol = "📄"
        elif node_data.node_type == 'chunk':
            node_symbol = "📝"
        elif node_data.node_type == 'root':
            node_symbol = "📁"
        elif node_data.node_type == 'directory':
            node_symbol = "📂"
        else:
            node_symbol = "📦"

        if level == 0:
            print(f"{node_symbol} {node_data.name} ({node_data.node_type})")
        else:
            print(f"{prefix}└── {node_symbol} {node_data.name} ({node_data.node_type})")

        # Get children via 'contains' edges
        children = [
            child for child in self.graph.successors(start_node_id)
            if self.graph.edges[start_node_id, child].get('relation') == 'contains'
        ]

        child_count = len(children)
        for i, child_id in enumerate(children):
            is_last = i == child_count - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            self.print_tree(max_depth, start_node_id=child_id, level=level + 1, prefix=new_prefix)

    def save_graph_visualization(self, output_path: str = "repo_graph.png"):
        pos = nx.spring_layout(self.graph, seed=42)  # Consistent layout

        # Define edge colors by type
        edge_colors = []
        for u, v in self.graph.edges:
            relation = self.graph.edges[u, v].get('relation', '')
            if relation == 'contains':
                edge_colors.append('green')
            elif relation == 'calls':
                edge_colors.append('blue')
            else:
                edge_colors.append('black')  # Default fallback

        # Prepare node labels
        labels = {}
        for node_id, node_attrs in self.graph.nodes(data=True):
            if 'data' not in node_attrs:
                labels[node_id] = node_id  # Fallback
                continue

            node = node_attrs['data']
            if isinstance(node, (FileNode, ChunkNode)):
                defined_summary = ', '.join([
                    f"{k}:{len(v)}" for k, v in node.defined_entities.items()
                ])
                called_summary = ', '.join(node.called_entities[:3])  # Limit to 3 for display
                label = f"{node.name}\nCalls: {called_summary}\nDefs: {defined_summary}"
            else:
                label = node.name
            labels[node_id] = label

        plt.figure(figsize=(18, 12))
        nx.draw(
            self.graph,
            pos,
            labels=labels,
            with_labels=True,
            font_size=8,
            node_color='lightyellow',
            edge_color=edge_colors,
            node_size=2000,
            font_weight='bold',
        )
        plt.title("Repository Knowledge Graph", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, format='PNG')
        plt.close()
        self.logger.info(f"Graph saved to {output_path}")

    def to_dict(self):
        graph_data = {
            'nodes': [],
            'edges': []
        }

        for node_id, node_attrs in self.graph.nodes(data=True):
            if 'data' not in node_attrs:
                self.logger.warning(f"Node {node_id} has no 'data' attribute, skipping in serialization")
                continue

            node = node_attrs['data']
            node_dict = {
                'id': node.id or node_id,  # Use node_id as fallback if node.id is empty
                'class': node.__class__.__name__,
                'data': {
                    'id': node.id or node_id,
                    'name': node.name,
                    'node_type': node.node_type,
                    'description': node.description,
                    'defined_entities': node.defined_entities,
                    'called_entities': node.called_entities,
                }
            }

            # FileNode-specific
            if isinstance(node, FileNode):
                node_dict['data']['path'] = node.path
                node_dict['data']['content'] = node.content
                node_dict['data']['language'] = node.language if hasattr(node, 'language') else ''

            # ChunkNode-specific
            if isinstance(node, ChunkNode):
                node_dict['data']['order_in_file'] = node.order_in_file
                node_dict['data']['embedding'] = node.embedding

            graph_data['nodes'].append(node_dict)

        for u, v, attrs in self.graph.edges(data=True):
            graph_data['edges'].append({
                'source': u,
                'target': v,
                'relation': attrs.get('relation', '')
            })

        return graph_data

    @classmethod
    def from_dict(cls, data_dict, index_nodes:bool=True, use_embed:bool=True):
        self = cls.__new__(cls)  # bypass __init__
        setup_logger(LOGGER_NAME)
        self.model_service = ModelService()
        self.logger = logging.getLogger(LOGGER_NAME)
        self.code_parser = CodeParser()
        self.nodes = {0: [], 1: [], 2: []}
        self.graph = nx.DiGraph()

        node_classes = {
            'Node': Node,
            'FileNode': FileNode,
            'ChunkNode': ChunkNode,
            'DirectoryNode': DirectoryNode,
        }

        # Create a root node if not present in the data
        root_found = False
        for node_data in data_dict['nodes']:
            if node_data['id'] == 'root':
                root_found = True
                break

        if not root_found:
            self.logger.warning("Root node not found in the data, creating one")
            root_node = Node(id='root', name='root', node_type='root')
            self.graph.add_node('root', data=root_node, level=0)

        for node_data in data_dict['nodes']:
            cls_name = node_data['class']
            node_cls = node_classes.get(cls_name, Node)
            kwargs = node_data['data']

            # Ensure ID is properly set
            if not kwargs.get('id'):
                kwargs['id'] = node_data['id']

            # Provide default values where needed
            kwargs.setdefault('defined_entities', {})
            kwargs.setdefault('called_entities', [])
            if node_cls in (FileNode, ChunkNode):
                kwargs.setdefault('path', '')
                kwargs.setdefault('content', '')
                kwargs.setdefault('language', '')
            if node_cls == ChunkNode:
                kwargs.setdefault('order_in_file', 0)

            node_instance = node_cls(**kwargs)
            self.graph.add_node(node_data['id'], data=node_instance, level=self._infer_level(node_instance))

        for edge in data_dict['edges']:
            source = edge['source']
            target = edge['target']
            # Check nodes exist before adding edge
            if source in self.graph and target in self.graph:
                self.graph.add_edge(source, target, relation=edge.get('relation', ''))
            else:
                self.logger.warning(f"Cannot add edge {source} -> {target}, nodes don't exist")

        if index_nodes:
            self.code_index = CodeIndex(list(self), use_embed=use_embed)

        return self

    def _infer_level(self, node):
        """Infer the level of a node based on its type"""
        if node.node_type == 'root':
            return 0
        elif node.node_type in ('file', 'directory'):
            return 1
        elif node.node_type == 'chunk':
            return 2
        return 1  # Default level

    def save_graph_to_file(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_graph_from_file(cls, filepath: str, index_nodes= True, use_embed:bool = True):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, use_embed=use_embed, index_nodes=index_nodes)

    def get_neighbors(self, node_id):
        return [self.graph.nodes[neighbor_id]['data']
                for neighbor_id in self.graph.neighbors(node_id)
                if 'data' in self.graph.nodes[neighbor_id]]

    def get_previous_chunk(self, node: ChunkNode) -> ChunkNode:
        # Check if node is of type ChunkNode
        if not isinstance(node, ChunkNode):
            raise Exception(f'Cannot get previous chunk on node of type {type(node)}')

        if node.order_in_file == 0:
            raise Exception(f'Cannot get previous chunk for first node')

        file_path = node.path
        previous_chunk_id = f'{file_path}_{node.order_in_file - 1}'

        if previous_chunk_id not in self.graph:
            raise Exception(f'Previous chunk {previous_chunk_id} not found in graph')

        previous_chunk = self[previous_chunk_id]
        return previous_chunk


