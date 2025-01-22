from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load env file
env_path = os.path.join(os.path.dirname(__file__), 'env')
load_dotenv(env_path)


class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/sales_agent_db")
    
    class Config:
        env_file = "env"
        env_file_encoding = "utf-8"

# Create a global settings instance
settings = Settings()