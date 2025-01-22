from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel

from app.services.product_database import Product


class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = datetime.now()
    metadata: Dict = {}


class Conversation(BaseModel):
    id: str
    messages: List[Message] = []
    context: Dict = {}
    customer_preferences: Dict = {}
    last_products: List[Product] = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.messages.append(Message(role=role, content=content, metadata=metadata or {}))
