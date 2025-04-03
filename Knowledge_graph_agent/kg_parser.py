from llama_cloud_services.parse import ResultType
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, PropertyGraphIndex, StorageContext
from llama_cloud_services import LlamaParse
from llama_index.core.schema import TextNode, Document
from llama_index.core import VectorStoreIndex
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
import nest_asyncio
nest_asyncio.apply()
import os
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv
import base64

load_dotenv()
os.environ['OPENAI_API_KEY'] = base64.urlsafe_b64decode(os.getenv("OPENAI_API_KEY")).decode('utf-8')
os.environ['LLAMA_CLOUD_API_KEY'] = base64.urlsafe_b64decode(os.getenv("LLAMA_CLOUD_API_KEY")).decode('utf-8')

#################### Setup Model ######################

llm = OpenAI(model="gpt-4o")
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.llm = llm
Settings.embed_model = embed_model

###################### Parse Data ######################

BASE_DIR = Path(__file__).resolve().parent

DOCS = BASE_DIR / "data"

file="NewIDP2.docx"


docs = LlamaParse(result_type=ResultType.TXT).load_data(os.path.join(DOCS, file))


####################### Setup Kg Store ######################

graph_store = Neo4jPGStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)
vec_store = None

################# Split docs into pages ################

def get_sub_docs(docs):
    """Split docs into pages, by separator."""
    sub_docs = []
    for doc in docs:
        doc_chunks = doc.text.split("\n---\n")
        for doc_chunk in doc_chunks:
            sub_doc = Document(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            sub_docs.append(sub_doc)

    return sub_docs


# this will split into pages
sub_docs = get_sub_docs(docs)

####################Build Baseline Vector Index #####################
persist_dir = "./storage"
os.makedirs(persist_dir, exist_ok=True)

storage_context = StorageContext.from_defaults()
base_index = VectorStoreIndex.from_documents(sub_docs, embed_model=embed_model, storage_context=storage_context)

storage_context._persist_dir = persist_dir

base_index.storage_context.persist()


################## Construct Knowledge Graph ####################

index = PropertyGraphIndex.from_documents(
    sub_docs,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    kg_extractors=[
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3),
            num_workers=4,
            max_paths_per_chunk=10,
        ),
    ],
    property_graph_store=graph_store,
    show_progress=True,
)






