import os
import base64
from dotenv import load_dotenv
from huggingface_hub import login

class Config:
    """Configuration management for the test suite generator."""
    
    def __init__(self):
        load_dotenv()
        self._setup_huggingface()
        self._setup_langfuse()
    
    def _setup_huggingface(self):
        """Setup Hugging Face authentication."""
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)
    
    def _setup_langfuse(self):
        """Setup Langfuse configuration for OpenTelemetry."""
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        
        if self.langfuse_public_key and self.langfuse_secret_key:
            langfuse_auth = base64.b64encode(
                f"{self.langfuse_public_key}:{self.langfuse_secret_key}".encode()
            ).decode()
            
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
    
    @property
    def model_config(self):
        """Model configuration for the AI agent."""
        return {
            "model_id": "local-model",
            "api_base": "http://vllm:8000/v1",
            "api_key": "not-needed",
        }

# Global config instance
config = Config()