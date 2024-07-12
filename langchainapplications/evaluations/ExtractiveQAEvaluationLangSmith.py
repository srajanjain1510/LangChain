import os
from urllib.request import urlretrieve

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.evaluation import EvaluatorType
from langchain.indexes import VectorstoreIndexCreator
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langsmith import Client
from langsmith.utils import LangSmithError

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

os.makedirs("../architectural_view_archive", exist_ok=True)
files = [
    "https://www.cs.ubc.ca/~gregor/teaching/papers/4+1view-architecture.pdf",
    "https://www.cse.iitk.ac.in/users/cs455/slides/2.pdf",
    "https://ics.uci.edu/~andre/ics223w2006/kruchten3.pdf",
    "https://research.cs.queensu.ca/home/ahmed/home/teaching/CISC322/F08/slides/CISC322_06_4and1Views.pdf"
]
for url in files:
    file_path = os.path.join("../architectural_view_archive", url.rpartition("/")[2])
    urlretrieve(url, file_path)

# Load pdf files in the local directory
loader = PyPDFDirectoryLoader("../architectural_view_archive/")

index = VectorstoreIndexCreator(embedding=huggingface_embeddings).from_loaders([loader])

llm_model = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.6})

# -- Evaluation --#
# ---RAG-----------#
eval_questions = [
    "Who designed the 4+1 view model?",
    "What is Development View?",
    "What is difference between physical view and logical view?"
]

eval_answers = [
    "Philippe Kruchten",  # correct Answer
    "New Delhi",  # Incorrect Answer
    "Physical View deals with the system's physical deployment and hardware considerations, "
    "the Logical View focuses on the system's functional requirements and abstractions.",

]

examples = zip(eval_questions, eval_answers)

client = Client()
dataset_name = "rag_eval_dataset"

try:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print("using existing dataset: ", dataset.name)
except LangSmithError:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="sample evaluation dataset",
    )
    for question, answer in examples:
        client.create_example(
            inputs={"input": question},
            outputs={"answer": answer},
            dataset_id=dataset.id,
        )

    print("Created a new dataset: ", dataset.name)


def create_qa_chain(return_context=True):
    qa_chain = RetrievalQA.from_chain_type(
        llm_model,
        retriever=index.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=return_context,
    )
    return qa_chain


evaluation_config = RunEvalConfig(
    evaluators=[
        EvaluatorType.QA,
        EvaluatorType.COT_QA,
        EvaluatorType.CONTEXT_QA,
    ],
    prediction_key="result",
    eval_llm=llm_model
)

client = Client()
run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=create_qa_chain,
    client=client,
    evaluation=evaluation_config,
    verbose=True,
)
