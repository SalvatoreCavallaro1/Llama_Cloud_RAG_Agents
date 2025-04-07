from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.query_engine import RetrieverQueryEngine
from kg_retriever import custom_retriever
from settings import *

################### Build Agent ######################

kg_query_engine = RetrieverQueryEngine(custom_retriever)
kg_query_tool = QueryEngineTool(
    query_engine=kg_query_engine,
    metadata=ToolMetadata(
        name="query_tool",
        description="Info about Content Driver documentation",
    ),
)

#
# agent_llm= OpenAI(model="gpt-4o",temperature=0)
#
#
# agent_llm.system_prompt="""You are an expert assistant specialized in the Content Driver documentation for
# the Market Execution Project (MEP). Your expertise covers the entire system, including data flows,
# integration with DAM, MDM, eteams, and master files, as well as the functionalities of the Content Driver
# and the Content Attribute Tool (CAT). When answering user queries, rely exclusively on the provided documentation,
# ensuring that your responses detail processes such as product enrichment, catalog updates, publication procedures,
# and the management of key features. Provide precise, clear, and accurate answers with relevant details
# and references, and avoid any speculation or inclusion of unverified information."""



# print(agent_llm)

llm.temperature=0

agent_worker = FunctionCallingAgentWorker.from_tools(
    [kg_query_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = agent_worker.as_agent()

##### Agent test #####
response = agent.chat("domanda")
