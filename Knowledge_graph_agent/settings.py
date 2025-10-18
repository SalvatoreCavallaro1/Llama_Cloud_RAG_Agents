from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from dotenv import load_dotenv
import os


load_dotenv()
# os.environ['OPENAI_API_KEY'] = base64.urlsafe_b64decode(os.getenv("OPENAI_API_KEY")).decode('utf-8')
# os.environ['LLAMA_CLOUD_API_KEY'] = base64.urlsafe_b64decode(os.getenv("LLAMA_CLOUD_API_KEY")).decode('utf-8')
os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv("LLAMA_CLOUD_API_KEY")
API_BASE = os.getenv("API_BASE", "http://localhost:11434/v1")
API_KEY = os.getenv("API_KEY")
LLM_MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
EMB_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "bge-m3")

#### Setup tracing #####
#
# tracer_provider = register(
#   project_name="kg_agent",
#   endpoint="https://app.phoenix.arize.com/v1/traces"
# )
#
#
# LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

#################### Setup Model ######################

# llm = OpenAI(model="gpt-4o",temperature=0)
llm = Ollama(
    model=LLM_MODEL_NAME,
    base_url=API_BASE,
    request_timeout=360,
    additional_kwargs={
        "num_ctx": 4096,   # prova 2048 o 4096
        "num_batch": 1     # meno memoria durante la generazione
    },
)
# embed_model = OpenAIEmbedding(model="text-embedding-3-large")
embed_model = OllamaEmbedding(
    model_name=EMB_MODEL_NAME,
    base_url=API_BASE
)

Settings.llm = llm
Settings.embed_model = embed_model

####################### Setup Kg Store ######################

graph_store = Neo4jPGStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)