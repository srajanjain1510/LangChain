import os
from urllib.request import urlretrieve

import numpy as np
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from langsmith.utils import LangSmithError

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

os.makedirs("../architectural_view_archive", exist_ok=True)
files = [
    "https://www.cs.ubc.ca/~gregor/teaching/papers/4+1view-architecture.pdf",
    "https://www.cse.iitk.ac.in/users/cs455/slides/2.pdf",
    "https://ics.uci.edu/~andre/ics223w2006/kruchten3.pdf"
]
for url in files:
    file_path = os.path.join("../architectural_view_archive", url.rpartition("/")[2])
    urlretrieve(url, file_path)

# Load pdf files in the local directory
loader = PyPDFDirectoryLoader("../architectural_view_archive/")

# Split documents to smaller chunks
docs_before_split = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
docs_after_split = text_splitter.split_documents(docs_before_split)

avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs]) // len(docs)
avg_char_before_split = avg_doc_length(docs_before_split)
avg_char_after_split = avg_doc_length(docs_after_split)

# Text Embeddings with Hugging Face models
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# Now we can see how a sample embedding would look like for one of those chunks.
sample_embedding = np.array(huggingface_embeddings.
                            embed_query(docs_after_split[0].page_content))

# Retrieval System for vector embeddings
vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)

query = """Explain the process decomposition model for resiliency in microservices for login application"""
# Sample question, change to other questions you are interested in.
relevant_documents = vectorstore.similarity_search(query)
print(
    f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. '
    f'Display the first one:\n')
for doc in relevant_documents:
    print(doc.page_content)

# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

retriever_invoke_result = retriever.invoke(query)

print(retriever_invoke_result)

# -- Evaluation --#
# ---RAG-----------#
eval_questions = [
    "Who designed the 4+1 view model?",
    "What is Development View?",
    "What is difference between physical view and logical view?"
]

eval_answers = [
    "Philippe Kruchten",  # correct Answer
    "The major themes are happiness and trustworthiness.",  # Incorrect Answer
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


def create_qa_chain(llm, vstore, return_context=True):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vstore.as_retriever(),
        return_source_documents=return_context,
    )
    return qa_chain
