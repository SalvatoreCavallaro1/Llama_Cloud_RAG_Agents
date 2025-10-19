from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.agent import AgentWorkflow
from Knowledge_graph_agent.kg_retriever import custom_retriever
from Knowledge_graph_agent.settings import *
from llama_index.llms.ollama import Ollama
import asyncio

llm2 = Ollama(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://localhost:11434",
    is_function_calling_model=True,
    json_mode=True,
    request_timeout=300,
    additional_kwargs={
                "num_ctx": 4096,
                "num_batch": 1
            })
llm2.temperature = 0

def create_agent_instance():
    kg_query_engine = RetrieverQueryEngine(custom_retriever)
    kg_query_tool = QueryEngineTool(
        query_engine=kg_query_engine,
        metadata=ToolMetadata(name="query_tool", description="Info about documentation"),
    )
    agent2 = AgentWorkflow.from_tools_or_functions([kg_query_tool], llm=llm2, system_prompt="")
    return agent2

async def main():
    agent = create_agent_instance()
    resp = await agent.run(user_msg="domanda di test, cosa sai dell' intro del corso?")
    print(str(resp))

if __name__ == "__main__":
    asyncio.run(main())
