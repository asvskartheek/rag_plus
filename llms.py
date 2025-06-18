from dspy import LM
from dotenv import load_dotenv
import os

# Gemma3 - 4b
gemma_4b = LM(
    "lm_studio/google/gemma-3-4b",
    api_base="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

# Azure OpenAI
load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_gpt4o = LM(
    model=f"azure/{deployment_name}",
    api_key=api_key,
    api_base=azure_endpoint,
    api_version=api_version,
    model_type="chat"
)