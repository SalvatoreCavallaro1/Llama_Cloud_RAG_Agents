from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from dotenv import load_dotenv
import os
import base64

load_dotenv()
os.environ['OPENAI_API_KEY'] = base64.urlsafe_b64decode(os.getenv("OPENAI_API_KEY")).decode('utf-8')
os.environ['LLAMA_CLOUD_API_KEY'] = base64.urlsafe_b64decode(os.getenv("LLAMA_CLOUD_API_KEY")).decode('utf-8')

##### Setup tracing #####

tracer_provider = register(
  project_name="vector_agent",
  endpoint="https://app.phoenix.arize.com/v1/traces"
)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

persist_dir = "storage_chroma"
llm = OpenAI(model="gpt-4o")
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.embed_model = embed_model
Settings.llm = llm