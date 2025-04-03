from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores.types import (
    VectorStoreInfo,
    VectorStoreQuerySpec,
    MetadataInfo,
    MetadataFilters,
    FilterCondition, VectorStoreQuery,
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize, BaseSynthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker
from typing import List
from dotenv import load_dotenv
import os
import base64
#################################################RETIEVER ######################################################

load_dotenv()
os.environ['OPENAI_API_KEY'] = base64.urlsafe_b64decode(os.getenv("OPENAI_API_KEY")).decode('utf-8')
os.environ['LLAMA_CLOUD_API_KEY'] = base64.urlsafe_b64decode(os.getenv("LLAMA_CLOUD_API_KEY")).decode('utf-8')

persist_dir = "storage_chroma"
llm = OpenAI(model="gpt-4o")
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

vector_store = ChromaVectorStore.from_params(
    collection_name="text_nodes", persist_dir=persist_dir
)
index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)


Settings.embed_model = embed_model
Settings.llm = llm

chunk_retriever = index.as_retriever(similarity_top_k=3)

def section_retrieve(query: str, verbose: bool = False) -> List[NodeWithScore]:
    """Retrieve sections."""
    if verbose:
        print(f">> Identifying the right sections to retrieve")
    chunk_nodes = chunk_retriever.retrieve(query)

    all_section_nodes = {}
    for node in chunk_nodes:
        section_id = node.node.metadata["section_id"]
        if verbose:
            print(f">> Retrieving section: {section_id}")
        filters = MetadataFilters.from_dicts(
            [
                {"key": "section_id", "value": section_id, "operator": "=="},
                {
                    "key": "paper_path",
                    "value": node.node.metadata["paper_path"],
                    "operator": "==",
                },
            ],
            condition=FilterCondition.AND,
        )


        query_obj = VectorStoreQuery(filters=filters)
        result = index.vector_store.query(query_obj)
        section_nodes_raw = result.nodes
        section_nodes = [NodeWithScore(node=n) for n in section_nodes_raw]
        # order and consolidate nodes
        section_nodes_sorted = sorted(
            section_nodes, key=lambda x: x.metadata["page_num"]
        )

        all_section_nodes.update({n.id_: n for n in section_nodes_sorted})
    return all_section_nodes.values()


########### Try out Section-Level Retrieval as a Full RAG Pipeline ##########################
class SectionRetrieverRAGEngine(CustomQueryEngine):
    """RAG Query Engine."""

    synthesizer: BaseSynthesizer
    verbose: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(synthesizer=TreeSummarize(llm=llm))

    def custom_query(self, query_str: str):
        nodes = section_retrieve(query_str, verbose=self.verbose)
        response_obj = self.synthesizer.synthesize(query_str, nodes)
        return response_obj

query_engine = SectionRetrieverRAGEngine()


# response = query_engine.query(
#     "Come faccio a chiudere un ticket?"
# )
# print(str(response))


################### Build Agent ######################


kg_query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="query_tool",
        description="Provides information about the New IDP",
    ),
)

agent_worker = FunctionCallingAgentWorker.from_tools(
    [kg_query_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = agent_worker.as_agent()

##### Agent test #####
# response = agent.chat("Come posso risolvere un ticket dove il problema è un utente che non riceve OTP per IDP?")
response = agent.chat("Come faccio a chiudere un ticket?")






