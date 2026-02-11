import json
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "physical_ai_book"

    # Neon Postgres
    database_url: str

    # Auth
    jwt_secret: str = "change-this-in-production-to-a-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # App
    cors_origins: str = '["http://localhost:3000","https://mromankhan.github.io"]'
    chunk_size: int = 512
    chunk_overlap: int = 50

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.cors_origins)

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
