import os

from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.llms.huggingface_hub import HuggingFaceHub

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


def language_translator():
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
        sentence_to_translate="What is your name?", language="Spanish")
    print(prediction_msg)


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


language_translator()

language_translator_latest("what is your name", "french")
