import os
from smolagents import CodeAgent, OpenAIServerModel
from pedagogia_code_questions.RepoKnowledgeGraph import RepoKnowledgeGraph

from config import config
from tools import FileTraverserTool, CodeReaderTool, TestFileWriterTool, CoverageCalculatorTool, GraphNavigationTool

class TestSuiteGenerator:
    """Main class for generating test suites using AI agents."""
    
    def __init__(self):
        self.model = self._create_model()
        self.tools = self._create_tools()
    
    def _create_model(self):
        """Create and configure the AI model."""
        return OpenAIServerModel(
            model_id=config.model_config["model_id"],
            api_base=config.model_config["api_base"],
            api_key=config.model_config["api_key"],
        )
    
    def _create_tools(self):
        """Create and return the tools for the agent."""
        return [
            FileTraverserTool(),
            CodeReaderTool(),
            GraphNavigationTool(),
            TestFileWriterTool(),
            CoverageCalculatorTool()
        ]
    
    def _collect_python_files(self, path: str) -> list:
        """Collect all Python files in the given path."""
        python_files = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def _read_all_code(self, python_files: list) -> str:
        """Read and concatenate all Python files content."""
        code = ""
        for file in python_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    code += f.read() + "\n"
            except (UnicodeDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not read file {file}: {e}")
                continue
        return code
    
    def _create_knowledge_graph(self, path: str) -> str:
        """Create and save the repository knowledge graph."""
        try:
            graph = RepoKnowledgeGraph(path=path)
            graph.save_graph_to_file("graph.json")
            
            with open('graph.json', 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not create knowledge graph: {e}")
            return ""
    
    def _create_agent_task(self, path: str) -> str:
        """Create the task description for the agent."""
        return (
            "You are a test generation expert. Your task is to generate comprehensive unit tests for a Python project. "
            "Follow these steps systematically:\n\n"
            
            "1. **Explore the project structure**: Use the file_traverser tool to find all Python files in the " + path + "/ directory\n"
            
            "2. **Navigate the knowledge graph**: Use the graph_navigator tool to understand the project structure:\n"
            "   - Use action='list_files' to see all files in the repository\n"
            "   - For each important file, use action='get_file_entities' to understand what functions/classes it defines\n"
            "   - Use action='get_file_chunks' to see how files are organized into logical chunks\n"
            "   - Use action='get_calls' and 'get_called_by' to understand dependencies between components\n\n"
            
            "3. **Read the code**: Use the code_reader tool to read the content of the Python files you found\n\n"
            
            "4. **Analyze dependencies**: Use the graph_navigator tool to:\n"
            "   - Identify which files/functions call others (action='get_calls')\n"
            "   - Find what entities are defined in each file (action='get_file_entities')\n"
            "   - Search for specific entities across the codebase (action='search_entities')\n\n"
            
            "5. **Generate comprehensive tests**: Based on your analysis, create unit tests that:\n"
            "   - Test all public functions and methods\n"
            "   - Cover edge cases and error conditions\n"
            "   - Mock external dependencies properly\n"
            "   - Test integration between components\n"
            "   - Include setup and teardown as needed\n"
            "   - Use appropriate assertions and test data\n\n"
            
            "6. **Structure your tests**: Organize tests by:\n"
            "   - Creating separate test files for each main module\n"
            "   - Grouping related tests in test classes\n"
            "   - Using descriptive test names that explain what is being tested\n"
            "   - Including docstrings for complex test scenarios\n\n"
            
            "**Important guidelines:**\n"
            "- Use the graph_navigator tool extensively to understand the codebase structure before writing tests\n"
            "- Focus on testing the most critical and complex parts of the code first\n"
            "- Include both positive and negative test cases\n"
            "- Mock external dependencies and file I/O operations\n"
            "- Your final answer should contain the complete test code, ready to run with pytest\n"
            "- Do NOT try to execute the tests, just return the test code\n\n"
            
            "The repository knowledge graph is available in graph.json - use the graph_navigator tool to explore it, "
            "do NOT use the code_reader tool on graph.json directly as it's very large."
        )
    
    def generate(self, path: str) -> str:
        """
        Generate test suite for the given project path.
        
        Args:
            path (str): Path to the project directory
            
        Returns:
            str: Generated test code
        """
        try:
            # Collect Python files and read code (for debugging/logging)
            python_files = self._collect_python_files(path)
            code = self._read_all_code(python_files)
            
            # Create knowledge graph
            graph_str = self._create_knowledge_graph(path)
            
            # Create agent with enhanced tools including graph navigation
            agent_tools = [FileTraverserTool(), CodeReaderTool(), GraphNavigationTool()]
            agent = CodeAgent(
                tools=agent_tools, 
                model=self.model, 
                max_steps=10, 
                additional_authorized_imports=[
                    '*', 'open', 'pytest', 'os', 'sys', 
                    'subprocess', 'posixpath', 'unittest'
                ]
            )
            
            # Create and run the task
            task = self._create_agent_task(path)
            output = agent.run(task)
            
            # Save output to file
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(output)
            
            print(f"Generated test suite for {len(python_files)} Python files")
            return output
            
        except Exception as e:
            print(f"Error generating test suite: {e}")
            raise

# Global generator instance
generator = TestSuiteGenerator()