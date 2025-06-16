from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict

@dataclass
class Node:
    id: str = ''
    name: str = ''
    node_type: str = ''
    description: Optional[str] = None
    defined_entities: Dict = field(default_factory=dict)  # Classes, functions, variables
    called_entities: List[str] = field(  default_factory=list)  # Classes, functions, variables, but also external libraries

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

@dataclass
class DirectoryNode(Node):
    path: str = ''
    node_type: str = 'directory'


@dataclass
class FileNode(Node):
    path: str = ''
    content: str = ''
    node_type: str = 'file'
    language : str = ''


@dataclass
class ChunkNode(FileNode):
    node_type: str = 'chunk'
    order_in_file: int = field(default_factory=int)
    embedding : list = None

    def get_field_to_embed(self) -> Optional[str]:
        return self.content
