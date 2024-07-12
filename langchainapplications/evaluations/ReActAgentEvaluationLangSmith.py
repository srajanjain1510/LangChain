import functools
import os

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.evaluation import EvaluatorType
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import render_text_description
from langsmith import EvaluationResult, Client
from langsmith.evaluation import run_evaluator
from langsmith.schemas import Run, Example

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ["SERPAPI_API_KEY"] = '03bb1e48b4d8ce43a99fddc0ec3ee121a93a6d17a80be873d6684af8ce4d564f'

client = Client()
# llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
llm_model = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.8})
chat_model = ChatHuggingFace(llm=llm_model)
# You can make the chat_model into an agent by giving it a ReAct style prompt and tools:
serpapi = SerpAPIWrapper()
# setup tools
tools = load_tools(["serpapi", "llm-math"], llm=llm_model)
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])


def react_agent_llm(prompt_template, llm_with_tools):
    # define the agent

    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            }
            | prompt_template
            | llm_with_tools
            | ReActJsonSingleInputOutputParser()
    )
    # instantiate AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor


@run_evaluator
def check_not_idk(run: Run, example: Example):
    """Illustration of a custom evaluator."""
    agent_response = run.outputs["output"]
    if "don't know" in agent_response or "not sure" in agent_response:
        score = 0
    else:
        score = 1
    return EvaluationResult(
        key="not_uncertain",
        score=score,
    )


evaluation_config = RunEvalConfig(
    evaluators=[
        EvaluatorType.QA,
        EvaluatorType.COT_QA,
        RunEvalConfig.LabeledCriteria("helpfulness", llm=llm_model),
        # The LabeledScoreString evaluator outputs a score on a scale from 1-10.
        # You can use default criteria or write our own rubric
        RunEvalConfig.LabeledScoreString(
            {
                "accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor errors or omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference."""
            },
            normalize_by=10,
        )
    ],
    custom_evaluators=[check_not_idk],
    eval_llm=llm_model
)

dataset_name = "ReAct_Prompt_Evaluation_Dataset"
chain_results = run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=functools.partial(
        react_agent_llm, prompt_template=prompt, llm_with_tools=chat_model_with_stop
    ),
    evaluation=evaluation_config,
    verbose=True,

)
