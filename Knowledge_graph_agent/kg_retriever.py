from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core import PropertyGraphIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from typing import List
from settings import *


index = PropertyGraphIndex.from_existing(
    graph_store,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    kg_extractors=[
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3),
            num_workers=4,
            max_paths_per_chunk=10,
        ),
    ],
    show_progress=True,
)

################## Define Vector Retriever ####################

kg_retriever = VectorContextRetriever(
    index.property_graph_store,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    similarity_top_k=2,
    path_depth=1,
    # include_text=False,
    include_text=True,
)


################## Load Baseline Vector Index ####################
persist_dir = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
base_index = load_index_from_storage(storage_context)
base_retriever = base_index.as_retriever(similarity_top_k=2)
base_query_engine = RetrieverQueryEngine(base_retriever)



################### Build Custom Retriever ######################
#Build joint retriever that combines vector and KG search.

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both KG vector search and direct vector search."""

    def __init__(self, kg_retriever, vector_retriever):
        self._kg_retriever = kg_retriever
        self._vector_retriever = vector_retriever

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        kg_nodes = self._kg_retriever.retrieve(query_bundle)
        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        unique_nodes = {n.node_id: n for n in kg_nodes}
        unique_nodes.update({n.node_id: n for n in vector_nodes})
        return list(unique_nodes.values())


custom_retriever = CustomRetriever(kg_retriever, base_retriever)
