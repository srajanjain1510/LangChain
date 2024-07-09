import os

from langchain.chains.llm import LLMChain
from langchain.smith import run_on_dataset, RunEvalConfig
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
from langsmith import Client, traceable

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

client = Client()

datasets = client.list_datasets()

for dataset in datasets:
    print(dataset)

@traceable
def language_translator_latest(sentence_to_translate: str, language: str):
    # Create the  prompt template
    sys_prompt: PromptTemplate = PromptTemplate(
        input_variables=["sentence_to_translate", "language"],
        template="""You are a language translater, an English speaker wants to translate/
        {sentence_to_translate} to {language}. Tell him the correct answer."""
    )
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)
    human_prompt: PromptTemplate = PromptTemplate(
        input_variables=["sentence_to_translate", "language"],
        template="Translate {sentence_to_translate} to {language}"
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    # chat prompt
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # Create the LLM chain
    llm_chain: LLMChain = LLMChain(prompt=chat_prompt,
                                   llm=HuggingFaceHub(repo_id="google/flan-t5-large",
                                                      model_kwargs={"temperature": 0.7, "max_length": 100}))
    # make a call to the models
    prediction_msg: dict = llm_chain.run(
        sentence_to_translate=sentence_to_translate, language=language)
    return prediction_msg


llm_model = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.8})

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
