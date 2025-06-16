import os
import subprocess
import json
from smolagents import Tool
from coverage import Coverage

class FileTraverserTool(Tool):
    """Tool to traverse directory and find Python files."""
    
    name = "file_traverser"
    description = "Parcourt un répertoire et retourne la liste des fichiers Python (.py) trouvés."
    inputs = {
        "directory": {"type": "string", "description": "Le chemin du répertoire à parcourir."}
    }
    output_type = "string"

    def forward(self, directory: str) -> str:
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

class CodeReaderTool(Tool):
    """Tool to read file contents."""
    
    name = "code_reader"
    description = "Lit un fichier et retourne son contenu."
    inputs = {
        "filepath": {"type": "string", "description": "Le chemin du fichier à lire."}
    }
    output_type = "string"

    def forward(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

class TestFileWriterTool(Tool):
    """Tool to write test code to a file."""
    
    name = "test_file_writer"
    description = "Écrit le code de tests généré dans un fichier .py."
    inputs = {
        "test_code": {"type": "string", "description": "Le code des tests à écrire."},
        "output_path": {"type": "string", "description": "Le chemin du fichier de tests à créer"}
    }
    output_type = "string"

    def forward(self, test_code: str, output_path: str) -> str:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(test_code)
        return f"Le fichier de tests a été écrit dans {output_path}"

class GraphNavigationTool(Tool):
    """Tool to navigate and query the repository knowledge graph."""
    
    name = "graph_navigator"
    description = "Navigate and query the repository knowledge graph to understand code structure, relationships, and dependencies."
    inputs = {
        "action": {
            "type": "string", 
            "description": "Action to perform: 'get_node', 'get_neighbors', 'get_file_chunks', 'get_calls', 'get_called_by', 'list_files', 'get_file_entities', 'search_entities'"
        },
        "node_id": {
            "type": "string", 
            "description": "Node ID to query (required for most actions except 'list_files')"
        },
        "entity_name": {
            "type": "string", 
            "description": "Entity name to search for (required for 'search_entities' action)"
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.graph_data = None
        self.nodes_by_id = {}
        self.edges = []
    
    def _load_graph(self):
        """Load the graph data from graph.json file."""
        if self.graph_data is None:
            try:
                with open('graph.json', 'r', encoding='utf-8') as f:
                    self.graph_data = json.load(f)
                
                # Create lookup dictionaries for efficient access
                self.nodes_by_id = {node['id']: node for node in self.graph_data['nodes']}
                self.edges = self.graph_data['edges']
                
            except FileNotFoundError:
                return "Error: graph.json file not found. Make sure the knowledge graph has been generated."
            except json.JSONDecodeError as e:
                return f"Error: Failed to parse graph.json: {e}"
        return None

    def forward(self, action: str, node_id: str = "", entity_name: str = "") -> str:
        # Load graph if not already loaded
        error = self._load_graph()
        if error:
            return error
        
        try:
            if action == "get_node":
                return self._get_node(node_id)
            elif action == "get_neighbors":
                return self._get_neighbors(node_id)
            elif action == "get_file_chunks":
                return self._get_file_chunks(node_id)
            elif action == "get_calls":
                return self._get_calls(node_id)
            elif action == "get_called_by":
                return self._get_called_by(node_id)
            elif action == "list_files":
                return self._list_files()
            elif action == "get_file_entities":
                return self._get_file_entities(node_id)
            elif action == "search_entities":
                return self._search_entities(entity_name)
            else:
                return f"Error: Unknown action '{action}'. Available actions: get_node, get_neighbors, get_file_chunks, get_calls, get_called_by, list_files, get_file_entities, search_entities"
        
        except Exception as e:
            return f"Error executing action '{action}': {str(e)}"
    
    def _get_node(self, node_id: str) -> str:
        """Get detailed information about a specific node."""
        if not node_id:
            return "Error: node_id is required for get_node action"
        
        if node_id not in self.nodes_by_id:
            return f"Error: Node '{node_id}' not found in graph"
        
        node = self.nodes_by_id[node_id]
        node_data = node['data']
        
        result = f"Node: {node_id}\n"
        result += f"Name: {node_data.get('name', 'N/A')}\n"
        result += f"Type: {node_data.get('node_type', 'N/A')}\n"
        result += f"Class: {node.get('class', 'N/A')}\n"
        
        if 'path' in node_data:
            result += f"Path: {node_data['path']}\n"
        
        if 'language' in node_data and node_data['language']:
            result += f"Language: {node_data['language']}\n"
        
        if 'description' in node_data and node_data['description']:
            result += f"Description: {node_data['description']}\n"
        
        # Show defined entities
        if 'defined_entities' in node_data and node_data['defined_entities']:
            result += "Defined entities:\n"
            for entity_type, entities in node_data['defined_entities'].items():
                if entities:
                    result += f"  {entity_type}: {', '.join(entities)}\n"
        
        # Show called entities
        if 'called_entities' in node_data and node_data['called_entities']:
            result += f"Called entities: {', '.join(node_data['called_entities'])}\n"
        
        return result
    
    def _get_neighbors(self, node_id: str) -> str:
        """Get all neighboring nodes (connected by any relation)."""
        if not node_id:
            return "Error: node_id is required for get_neighbors action"
        
        neighbors = set()
        
        # Find all edges where this node is source or target
        for edge in self.edges:
            if edge['source'] == node_id:
                neighbors.add((edge['target'], edge['relation'], 'outgoing'))
            elif edge['target'] == node_id:
                neighbors.add((edge['source'], edge['relation'], 'incoming'))
        
        if not neighbors:
            return f"No neighbors found for node '{node_id}'"
        
        result = f"Neighbors of {node_id}:\n"
        for neighbor_id, relation, direction in sorted(neighbors):
            neighbor_name = self.nodes_by_id.get(neighbor_id, {}).get('data', {}).get('name', neighbor_id)
            result += f"  {direction} '{relation}' -> {neighbor_name} ({neighbor_id})\n"
        
        return result
    
    def _get_file_chunks(self, file_id: str) -> str:
        """Get all chunks belonging to a specific file."""
        if not file_id:
            return "Error: node_id (file_id) is required for get_file_chunks action"
        
        chunks = []
        
        # Find all 'contains' edges from this file to chunks
        for edge in self.edges:
            if edge['source'] == file_id and edge['relation'] == 'contains':
                target_node = self.nodes_by_id.get(edge['target'])
                if target_node and target_node.get('class') == 'ChunkNode':
                    chunks.append((edge['target'], target_node['data']))
        
        if not chunks:
            return f"No chunks found for file '{file_id}'"
        
        # Sort chunks by order_in_file
        chunks.sort(key=lambda x: x[1].get('order_in_file', 0))
        
        result = f"Chunks in file {file_id}:\n"
        for chunk_id, chunk_data in chunks:
            result += f"  Chunk {chunk_data.get('order_in_file', 0)}: {chunk_id}\n"
            if chunk_data.get('description'):
                result += f"    Description: {chunk_data['description']}\n"
            
            # Show defined entities for this chunk
            if chunk_data.get('defined_entities'):
                entities_summary = []
                for entity_type, entities in chunk_data['defined_entities'].items():
                    if entities:
                        entities_summary.append(f"{entity_type}: {', '.join(entities)}")
                if entities_summary:
                    result += f"    Defines: {'; '.join(entities_summary)}\n"
        
        return result
    
    def _get_calls(self, node_id: str) -> str:
        """Get all nodes that this node calls."""
        if not node_id:
            return "Error: node_id is required for get_calls action"
        
        calls = []
        
        for edge in self.edges:
            if edge['source'] == node_id and edge['relation'] == 'calls':
                target_node = self.nodes_by_id.get(edge['target'])
                if target_node:
                    calls.append((edge['target'], target_node['data']))
        
        if not calls:
            return f"Node '{node_id}' doesn't call any other nodes"
        
        result = f"Nodes called by {node_id}:\n"
        for called_id, called_data in calls:
            result += f"  {called_data.get('name', called_id)} ({called_id})\n"
            if called_data.get('path') and called_data['path'] != called_id:
                result += f"    Path: {called_data['path']}\n"
        
        return result
    
    def _get_called_by(self, node_id: str) -> str:
        """Get all nodes that call this node."""
        if not node_id:
            return "Error: node_id is required for get_called_by action"
        
        called_by = []
        
        for edge in self.edges:
            if edge['target'] == node_id and edge['relation'] == 'calls':
                source_node = self.nodes_by_id.get(edge['source'])
                if source_node:
                    called_by.append((edge['source'], source_node['data']))
        
        if not called_by:
            return f"Node '{node_id}' is not called by any other nodes"
        
        result = f"Nodes that call {node_id}:\n"
        for caller_id, caller_data in called_by:
            result += f"  {caller_data.get('name', caller_id)} ({caller_id})\n"
            if caller_data.get('path') and caller_data['path'] != caller_id:
                result += f"    Path: {caller_data['path']}\n"
        
        return result
    
    def _list_files(self) -> str:
        """List all file nodes in the repository."""
        files = []
        
        for node_id, node in self.nodes_by_id.items():
            if node.get('class') == 'FileNode':
                files.append((node_id, node['data']))
        
        if not files:
            return "No files found in the repository"
        
        result = "Files in repository:\n"
        for file_id, file_data in sorted(files):
            result += f"  {file_data.get('name', file_id)} ({file_id})\n"
            if file_data.get('language'):
                result += f"    Language: {file_data['language']}\n"
        
        return result
    
    def _get_file_entities(self, file_id: str) -> str:
        """Get all entities defined and called in a specific file."""
        if not file_id:
            return "Error: node_id (file_id) is required for get_file_entities action"
        
        if file_id not in self.nodes_by_id:
            return f"Error: File '{file_id}' not found"
        
        file_node = self.nodes_by_id[file_id]
        if file_node.get('class') != 'FileNode':
            return f"Error: Node '{file_id}' is not a file node"
        
        file_data = file_node['data']
        
        result = f"Entities in file {file_data.get('name', file_id)}:\n"
        
        # Defined entities
        if file_data.get('defined_entities'):
            result += "Defined entities:\n"
            for entity_type, entities in file_data['defined_entities'].items():
                if entities:
                    result += f"  {entity_type}: {', '.join(entities)}\n"
        
        # Called entities
        if file_data.get('called_entities'):
            result += f"Called entities: {', '.join(file_data['called_entities'])}\n"
        
        return result
    
    def _search_entities(self, entity_name: str) -> str:
        """Search for nodes that define or call a specific entity."""
        if not entity_name:
            return "Error: entity_name is required for search_entities action"
        
        defining_nodes = []
        calling_nodes = []
        
        for node_id, node in self.nodes_by_id.items():
            node_data = node['data']
            
            # Check if this node defines the entity
            if node_data.get('defined_entities'):
                for entity_type, entities in node_data['defined_entities'].items():
                    if entity_name in entities:
                        defining_nodes.append((node_id, node_data, entity_type))
            
            # Check if this node calls the entity
            if node_data.get('called_entities') and entity_name in node_data['called_entities']:
                calling_nodes.append((node_id, node_data))
        
        if not defining_nodes and not calling_nodes:
            return f"Entity '{entity_name}' not found in the repository"
        
        result = f"Search results for entity '{entity_name}':\n"
        
        if defining_nodes:
            result += "Defined in:\n"
            for node_id, node_data, entity_type in defining_nodes:
                result += f"  {node_data.get('name', node_id)} ({node_id}) as {entity_type}\n"
        
        if calling_nodes:
            result += "Called by:\n"
            for node_id, node_data in calling_nodes:
                result += f"  {node_data.get('name', node_id)} ({node_id})\n"
        
        return result

class CoverageCalculatorTool(Tool):
    """Tool to calculate code coverage and generate HTML report."""
    
    name = "coverage_calculator"
    description = "Calcule la couverture de code avec coverage.py et génère un rapport HTML."
    inputs = {
        "tests_directory": {"type": "string", "description": "Le répertoire contenant les tests."},
        "source_directory": {"type": "string", "description": "Le répertoire des sources à analyser."}
    }
    output_type = "string"

    def forward(self, tests_directory: str, source_directory: str) -> str:
        cov = Coverage(source=[source_directory], branch=True)
        cov.start()
        result = subprocess.run(
            ["pytest", tests_directory, "--maxfail=1", "--disable-warnings"],
            capture_output=True,
            text=True
        )
        cov.stop()
        cov.save()
        report_output = subprocess.run(["coverage", "report", "-m"], capture_output=True, text=True)
        cov.html_report(directory="htmlcov")
        lines = report_output.stdout.splitlines()
        total_line = next((l for l in lines if l.strip().startswith("TOTAL")), "")
        return f"{total_line}\nRapport HTML : ./htmlcov/index.html"