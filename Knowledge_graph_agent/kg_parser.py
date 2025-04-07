from llama_cloud_services.parse import ResultType
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PropertyGraphIndex, StorageContext
from llama_cloud_services import LlamaParse
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
import nest_asyncio
from copy import deepcopy
from pathlib import Path
from settings import *

nest_asyncio.apply()

###################### Parse Data ######################

BASE_DIR = Path(__file__).resolve().parent

DOCS = BASE_DIR / "data"

file="Guida_utente_1.pdf"


docs = LlamaParse(result_type=ResultType.TXT).load_data(os.path.join(DOCS, file))


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






