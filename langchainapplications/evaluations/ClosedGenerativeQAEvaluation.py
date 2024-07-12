import os

from langchain.smith import run_on_dataset, RunEvalConfig
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langsmith import Client

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

client = Client()

llm_model = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.4})

evaluation_config = RunEvalConfig(
    evaluators=[RunEvalConfig.Criteria(
        criteria={"usefulness": "The prediction is useful if..."}, llm=llm_model),
        RunEvalConfig.Criteria(
            criteria={"Coherence": "The prediction is Coherence if..."}, llm=llm_model),
        RunEvalConfig.Criteria(
            criteria={"Harmfulness": "The prediction is Harmful if..."}, llm=llm_model),
        RunEvalConfig.Criteria(
            criteria={"Maliciousness": "The prediction is Malicious if..."}, llm=llm_model),
        RunEvalConfig.Criteria(
            criteria={"relevance": "The prediction is relevant if..."},
            llm=llm_model,
        )]
)

results = run_on_dataset(
    client=client,
    dataset_name="ds-artistic-attitude-34",
    llm_or_chain_factory=llm_model,
    evaluation=evaluation_config
)
