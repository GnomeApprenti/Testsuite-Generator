import re
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import logging

from .utils.logger_utils import setup_logger

LOGGER_NAME = "ENTITY_EXTRACTOR"
setup_logger(LOGGER_NAME)
logger = logging.getLogger(LOGGER_NAME)


@dataclass
class Entity:
   entity_type: str
   entity_name: str


class EntityExtractor(ABC):
    """Abstract base class for language-specific entity extractors"""

    @abstractmethod
    def extract_declared_entities(self, code: str) -> Dict[str, List[str]]:
        """
        Extract declared entities from code

        Args:
            code: The source code as a string

        Returns:
            Dict with entity types as keys and lists of entity names as values
        """
        pass

    @abstractmethod
    def extract_called_entities(self, code: str) -> List[str]:
        """
        Extract called entities from code

        Args:
            code: The source code as a string

        Returns:
            List of called entity names
        """
        pass


class PythonExtractor(EntityExtractor):
    """Entity extractor for Python code"""

    def extract_declared_entities(self, code: str) -> Dict[str, List[str]]:
        entities = {
            'classes': [],
            'functions': [],
            'variables': []
        }

        # Extract classes
        class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?:'
        entities['classes'] = re.findall(class_pattern, code)

        # Extract functions
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        entities['functions'] = re.findall(function_pattern, code)

        # Extract global variables (simple approach)
        # This is a simplified approach and might not catch all variables
        # or might include some false positives
        variable_pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=(?!=)'
        entities['variables'] = re.findall(variable_pattern, code, re.MULTILINE)

        return entities

    def extract_called_entities(self, code: str) -> List[str]:
        # Basic function calls (won't catch all complex cases)
        call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        calls = re.findall(call_pattern, code)

        # Filter out function declarations
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        declarations = set(re.findall(function_pattern, code))

        # Include method calls
        method_pattern = r'(?:\.|->)([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        method_calls = re.findall(method_pattern, code)

        # Combine and remove duplicates
        all_calls = [call for call in calls if
                     call not in declarations and call != 'if' and call != 'for' and call != 'while']
        all_calls.extend(method_calls)

        return list(set(all_calls))


class JavaScriptExtractor(EntityExtractor):
    """Entity extractor for JavaScript code"""

    def extract_declared_entities(self, code: str) -> Dict[str, List[str]]:
        entities = {
            'classes': [],
            'functions': [],
            'variables': []
        }

        # Extract classes
        class_pattern = r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        entities['classes'] = re.findall(class_pattern, code)

        # Extract function declarations (named functions)
        function_pattern = r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        entities['functions'].extend(re.findall(function_pattern, code))

        # Extract arrow functions with names
        arrow_func_pattern = r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:\([^)]*\)|[a-zA-Z_$][a-zA-Z0-9_$]*)\s*=>'
        entities['functions'].extend(re.findall(arrow_func_pattern, code))

        # Extract methods in classes or objects
        method_pattern = r'(?:^|\s+)([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*{'
        entities['functions'].extend(re.findall(method_pattern, code, re.MULTILINE))

        # Extract variables (var, let, const)
        variable_pattern = r'(?:var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?:=|;|,)'
        entities['variables'] = re.findall(variable_pattern, code)

        return entities

    def extract_called_entities(self, code: str) -> List[str]:
        # Function calls
        call_pattern = r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        calls = re.findall(call_pattern, code)

        # Method calls
        method_pattern = r'\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        method_calls = re.findall(method_pattern, code)

        # Filter out function declarations and control structures
        function_pattern = r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        declarations = set(re.findall(function_pattern, code))

        # Filter keywords and declarations
        keywords = {'if', 'for', 'while', 'switch', 'catch', 'function'}
        filtered_calls = [call for call in calls if call not in declarations and call not in keywords]

        # Combine calls
        all_calls = filtered_calls + method_calls

        return list(set(all_calls))


class JavaExtractor(EntityExtractor):
    """Entity extractor for Java code"""

    def extract_declared_entities(self, code: str) -> Dict[str, List[str]]:
        entities = {
            'classes': [],
            'functions': [],
            'variables': []
        }

        # Extract classes and interfaces
        class_pattern = r'(?:class|interface|enum)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        entities['classes'] = re.findall(class_pattern, code)

        # Extract methods
        # This pattern is simplified and might not catch all method declarations
        method_pattern = r'(?:public|private|protected|static|\s)+[a-zA-Z_$][a-zA-Z0-9_$]*\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*(?:{|throws)'
        entities['functions'] = re.findall(method_pattern, code)

        # Extract variables (simplified)
        # This won't catch all variable declarations, especially complex ones
        variable_pattern = r'(?:^|\s+)(?:(?:public|private|protected|static|final|\s)+)?(?:[a-zA-Z_$][a-zA-Z0-9_$]*(?:<[^>]*>)?)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?:=|;)'
        entities['variables'] = re.findall(variable_pattern, code, re.MULTILINE)

        return entities

    def extract_called_entities(self, code: str) -> List[str]:
        # Method calls
        call_pattern = r'(?<!new\s)([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        calls = re.findall(call_pattern, code)

        # Object method calls
        method_pattern = r'\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        method_calls = re.findall(method_pattern, code)

        # Filter out method declarations and control structures
        method_pattern = r'(?:public|private|protected|static|\s)+[a-zA-Z_$][a-zA-Z0-9_$]*\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        declarations = set(re.findall(method_pattern, code))

        # Filter keywords
        keywords = {'if', 'for', 'while', 'switch', 'catch'}
        filtered_calls = [call for call in calls if call not in declarations and call not in keywords]

        # Combine calls
        all_calls = filtered_calls + method_calls

        return list(set(all_calls))


class CExtractor(EntityExtractor):
    """Entity extractor for C code"""

    def extract_declared_entities(self, code: str) -> Dict[str, List[str]]:
        entities = {
            'structs': [],
            'functions': [],
            'variables': []
        }

        # Extract structs
        struct_pattern = r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        entities['structs'] = re.findall(struct_pattern, code)

        # Extract function declarations
        # This is simplified and might not catch all function declarations
        function_pattern = r'(?:^|\s+)(?:[a-zA-Z_][a-zA-Z0-9_]*\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;]*\)\s*(?:{|$)'
        potential_functions = re.findall(function_pattern, code, re.MULTILINE)

        # Filter out control structures
        keywords = {'if', 'for', 'while', 'switch', 'return'}
        entities['functions'] = [f for f in potential_functions if f not in keywords]

        # Extract global variables (simplified)
        variable_pattern = r'(?:^|\s+)(?:(?:const|static|extern|\s)+)?(?:[a-zA-Z_][a-zA-Z0-9_]*(?:\s*\*)*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|;|\[)'
        entities['variables'] = re.findall(variable_pattern, code, re.MULTILINE)

        return entities

    def extract_called_entities(self, code: str) -> List[str]:
        # Function calls
        call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        calls = re.findall(call_pattern, code)

        # Filter out function declarations and control structures
        keywords = {'if', 'for', 'while', 'switch', 'return'}
        function_pattern = r'(?:^|\s+)(?:[a-zA-Z_][a-zA-Z0-9_]*\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        declarations = set(re.findall(function_pattern, code, re.MULTILINE))

        # Filter keywords and declarations
        filtered_calls = [call for call in calls if call not in declarations and call not in keywords]

        return list(set(filtered_calls))


class CPlusPlusExtractor(CExtractor):
    """Entity extractor for C++ code, extends C extractor"""

    def extract_declared_entities(self, code: str) -> Dict[str, List[str]]:
        entities = super().extract_declared_entities(code)

        # Rename structs to classes and add C++ classes
        entities['classes'] = entities.pop('structs')
        class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        entities['classes'].extend(re.findall(class_pattern, code))

        # Add namespace entities
        namespace_pattern = r'namespace\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        entities['namespaces'] = re.findall(namespace_pattern, code)

        # Extract templates
        template_pattern = r'template\s*<[^>]*>\s*(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        entities['classes'].extend(re.findall(template_pattern, code))

        return entities

    def extract_called_entities(self, code: str) -> List[str]:
        # Get base C function calls
        calls = super().extract_called_entities(code)

        # Add C++ specific method calls and operator overloads
        method_pattern = r'(?:->|\.)(~?[a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        method_calls = re.findall(method_pattern, code)

        # Add template calls
        template_call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)<[^>]*>\s*\('
        template_calls = re.findall(template_call_pattern, code)

        # Combine all calls
        all_calls = calls + method_calls + template_calls

        return list(set(all_calls))


class GoExtractor(EntityExtractor):
    """Entity extractor for Go code"""

    def extract_declared_entities(self, code: str) -> Dict[str, List[str]]:
        entities = {
            'structs': [],
            'interfaces': [],
            'functions': [],
            'variables': []
        }

        # Extract structs
        struct_pattern = r'type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+struct'
        entities['structs'] = re.findall(struct_pattern, code)

        # Extract interfaces
        interface_pattern = r'type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+interface'
        entities['interfaces'] = re.findall(interface_pattern, code)

        # Extract functions
        function_pattern = r'func\s+(?:\([^)]*\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        entities['functions'] = re.findall(function_pattern, code)

        # Extract variables
        variable_pattern = r'(?:var|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        entities['variables'] = re.findall(variable_pattern, code)

        # Extract short variable declarations
        short_var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:='
        entities['variables'].extend(re.findall(short_var_pattern, code))

        return entities

    def extract_called_entities(self, code: str) -> List[str]:
        # Function calls
        call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        calls = re.findall(call_pattern, code)

        # Method calls
        method_pattern = r'\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        method_calls = re.findall(method_pattern, code)

        # Filter out function declarations
        function_pattern = r'func\s+(?:\([^)]*\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        declarations = set(re.findall(function_pattern, code))

        # Filter keywords
        keywords = {'if', 'for', 'switch', 'select', 'defer', 'go', 'func'}
        filtered_calls = [call for call in calls if call not in declarations and call not in keywords]

        # Combine calls
        all_calls = filtered_calls + method_calls

        return list(set(all_calls))



def get_language_extractor(language: str) -> Optional[EntityExtractor]:
    """
    Get the appropriate entity extractor for the given language

    Args:
        language: Programming language string (case-insensitive)

    Returns:
        EntityExtractor instance or None if language not supported
    """
    language = language.lower()
    extractors = {
        'python': PythonExtractor(),
        'javascript': JavaScriptExtractor(),
        'js': JavaScriptExtractor(),
        'java': JavaExtractor(),
        'c': CExtractor(),
        'c++': CPlusPlusExtractor(),
        'cpp': CPlusPlusExtractor(),
        'go': GoExtractor(),
    }

    return extractors.get(language)

def get_language_from_filename(file_name:str) -> str:
    file_extension = file_name.split('.')[-1]
    extension_mapping = {
        'c': 'c',
        'h': 'c',
        'cpp': 'c++',
        'cc': 'c++',
        'cxx': 'c++',
        'hpp': 'c++',
        'hh': 'c++',
        'hxx': 'c++',
        'go': 'go',
        'java': 'java',
        'py': 'python',
        'pyc': 'python',
        'pyw':'python',
        'js': 'javascript',
        'mjs': 'javascript',
        'cjs': 'javascript',
    }
    # Throws error if language not defined
    return extension_mapping[file_extension]




def extract_entities(code: str, file_name:str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Extract declared and called entities from code in the specified language

    Args:
        code: The source code as a string
        language: Programming language of the code

    Returns:
        Tuple of (declared_entities, called_entities)
        declared_entities is a Dict with entity types as keys and lists of entity names as values
        called_entities is a List of called entity names
    """
    try:
        language = get_language_from_filename(file_name)
        extractor = get_language_extractor(language)

        if not extractor:
            supported_languages = ["python", "javascript", "java", "c", "c++", "go", "html"]
            raise ValueError(f"Unsupported language: {language}. Supported languages: {', '.join(supported_languages)}")

        declared_entities = extractor.extract_declared_entities(code)
        called_entities = extractor.extract_called_entities(code)

        return declared_entities, called_entities
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        logger.error("Returning empty extracted entities")
        return {}, []

