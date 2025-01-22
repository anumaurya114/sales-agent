from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel

class NodeType(Enum):
    WELCOME = "welcome"
    DECISION = "decision"
    RAG = "rag"
    QUESTION = "question"
    RECOMMENDATION = "recommendation"
    END = "end"
    FAREWELL = "farewell"

class Node(BaseModel):
    id: str
    type: NodeType
    content: str
    next_nodes: List[str] = []
    metadata: Dict = {}

class Edge(BaseModel):
    from_node: str
    to_node: str
    condition: Optional[str] = None

class ConversationGraph(BaseModel):
    nodes: Dict[str, Node]
    edges: List[Edge]
    current_node: str

    def next_node(self, condition: Optional[str] = None) -> Optional[Node]:
        current = self.nodes[self.current_node]
        for edge in self.edges:
            if edge.from_node == current.id and (not edge.condition or edge.condition == condition):
                self.current_node = edge.to_node
                return self.nodes[edge.to_node]
        return None