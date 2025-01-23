from typing import Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from app.models.conversation import Conversation
from app.services.agent import SalesAgent
from app.services.init_product_db import init_database
from app.services.product_database import Product

# Load .env file
load_dotenv()
app = FastAPI()
agent = SalesAgent()

# In-memory storage (replace with database in production)
conversations: Dict[str, Conversation] = {}


@app.on_event("startup")
async def startup_event():
    # Initialize the product database
    await init_database()
    print("Product database initialized successfully!")


class ProductReference(BaseModel):
    product_id: str
    aspect: Optional[str] = None  # e.g., "price", "features", "specs", "comparison"


class MessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    selected_product: Optional[ProductReference] = None
    action: Optional[str] = None
    context: Optional[Dict] = {}


class ConversationResponse(BaseModel):
    conversation_id: str
    response: str
    products: List[Product] = []
    suggested_questions: List[str] = []
    available_actions: List[str] = []


@app.post("/chat")
async def chat(request: MessageRequest) -> ConversationResponse:
    # Get or create conversation
    conversation_id = request.conversation_id or str(uuid4())
    conversation = conversations.get(conversation_id)
    if not conversation:
        conversation = Conversation(id=conversation_id)
        conversations[conversation_id] = conversation

    # Update conversation context with any new information
    if request.context:
        conversation.context.update(request.context)
    if request.action:
        conversation.context["action"] = request.action
    if request.selected_product:
        conversation.context["selected_product"] = request.selected_product.dict()

    # Process message
    response, products = await agent.process_message(conversation, request.message)
    
    # Important: Update conversation's last_products if new products are returned
    if products:
        conversation.last_products = products
    
    # Store updated conversation
    conversations[conversation_id] = conversation

    # Generate suggested questions based on the products
    suggested_questions = []
    if conversation.last_products:  # Use conversation's last_products instead
        product = conversation.last_products[0]
        suggested_questions = [
            f"Can you tell me more about the {product.name}?",
            f"What are the key features of the {product.name}?",
            f"How does the {product.name} compare to other models?",
        ]

    return ConversationResponse(
        conversation_id=conversation_id,
        response=response,
        products=conversation.last_products,  # Use conversation's last_products
        suggested_questions=suggested_questions,
        available_actions=["details", "compare", "features", "price"],
    )