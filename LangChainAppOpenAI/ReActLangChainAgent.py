import os

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import render_text_description

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ["SERPAPI_API_KEY"] = '03bb1e48b4d8ce43a99fddc0ec3ee121a93a6d17a80be873d6684af8ce4d564f'


def react_agent_llm():
    # The code to create the ChatModel and give it tools is really simple
    llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
    chat_model = ChatHuggingFace(llm=llm)
    # You can make the chat_model into an agent by giving it a ReAct style prompt and tools:
    serpapi = SerpAPIWrapper()
    # setup tools
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # setup ReAct style prompt
    prompt = hub.pull("hwchase17/react-json")
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    # define the agent
    chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            }
            | prompt
            | chat_model_with_stop
            | ReActJsonSingleInputOutputParser()
    )
    # instantiate AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    agent_executor.invoke(
        {
            "input": """What are the paid LLM models getting used with Lang chain and which is most popular in 2024 
            with Langchain and what is the trend in last 2 years"""
        }
    )


react_agent_llm()
