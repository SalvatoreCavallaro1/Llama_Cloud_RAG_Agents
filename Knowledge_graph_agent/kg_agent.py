from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.query_engine import RetrieverQueryEngine
from Knowledge_graph_agent.kg_retriever import custom_retriever
from Knowledge_graph_agent.settings import *

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
# agent_llm.system_prompt=""""""



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
# response = agent.chat("domanda di test")
