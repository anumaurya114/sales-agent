# AI Sales Assistant

An intelligent sales assistant powered by OpenAI's GPT-4 that helps customers find and purchase laptops based on their needs and preferences.

## Features

### Interactive Conversations

Natural language interaction with customers to understand their needs

### Smart Product Search

Intelligent product matching based on customer requirements

### Dynamic Recommendations

Personalized product recommendations considering budget, features, and use cases

### Product Database

Vector-based search using ChromaDB for efficient and relevant product matching

### Contextual Memory

Maintains conversation history to provide consistent and relevant responses

### Flexible API

RESTful API endpoints for easy integration with frontend applications

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- ChromaDB
- FastAPI

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-sales-assistant
```
2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Set up environment variables**
Create a `env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
### Running the Application

1. **Start the FastAPI server**
```bash
uvicorn app.main:app --reload
```
2. **Access the API**
- API will be available at `http://localhost:8000`
- Swagger documentation at `http://localhost:8000/docs`

## API Usage

### Chat Endpoint

`POST /chat`

Request body:
```json
{
"message": "I'm looking for a gaming laptop under $2000",
"conversation_id": "optional-conversation-id",
"selected_product": {
"product_id": "optional-product-id",
"aspect": "optional-aspect"
}
}
```
Response:
```json
{
"conversation_id": "unique-conversation-id",
"response": "AI assistant's response",
"products": ["list of matching products"],
"suggested_questions": ["list of follow-up questions"],
"available_actions": ["list of available actions"]
}
```

## Project Structure
pp/
├── main.py # FastAPI application entry point
├── config.py # Configuration settings
├── models/
│ └── conversation.py # Conversation model
├── services/
│ ├── agent.py # AI Sales Agent implementation
│ ├── product_database.py # Product database interface
│ └── init_product_db.py # Database initialization

## Features in Detail

### Conversation Flow

1. **Welcome**: Initial greeting and understanding customer intent
2. **Understanding Needs**: Gathering basic requirements
3. **Gathering Requirements**: Collecting specific details
4. **Product Search**: Finding matching products
5. **Recommendations**: Suggesting best-fit products
6. **Handling Questions**: Answering product-specific queries
7. **Selection**: Helping with final product selection

### Product Search Capabilities

- Price range filtering
- Brand preferences
- Feature requirements
- Use case matching
- Category filtering

## Configuration

The application can be configured through environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- Additional configuration can be added in `app/config.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 API
- ChromaDB for vector database
- FastAPI framework

## Important Notes

- This is a demonstration project and may need additional security measures for production use
- Rate limiting and error handling should be implemented for production deployment
- The product database is initialized with sample data and should be replaced with real product data in production