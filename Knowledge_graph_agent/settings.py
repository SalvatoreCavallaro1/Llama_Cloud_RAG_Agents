from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jPGStore
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
import base64
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = base64.urlsafe_b64decode(os.getenv("OPENAI_API_KEY")).decode('utf-8')
os.environ['LLAMA_CLOUD_API_KEY'] = base64.urlsafe_b64decode(os.getenv("LLAMA_CLOUD_API_KEY")).decode('utf-8')

#### Setup tracing #####

tracer_provider = register(
  project_name="kg_agent",
  endpoint="https://app.phoenix.arize.com/v1/traces"
)


LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

#################### Setup Model ######################

llm = OpenAI(model="gpt-4o",temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.llm = llm
Settings.embed_model = embed_model

####################### Setup Kg Store ######################

graph_store = Neo4jPGStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)