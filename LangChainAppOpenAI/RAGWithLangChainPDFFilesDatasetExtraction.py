import os
from urllib.request import urlretrieve

import numpy as np
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

os.makedirs("architectural_view_archive", exist_ok=True)
files = [
    "https://www.cs.ubc.ca/~gregor/teaching/papers/4+1view-architecture.pdf",
    "https://www.cse.iitk.ac.in/users/cs455/slides/2.pdf"
]
for url in files:
    file_path = os.path.join("architectural_view_archive", url.rpartition("/")[2])
    urlretrieve(url, file_path)

# Load pdf files in the local directory
loader = PyPDFDirectoryLoader("./architectural_view_archive/")

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

print(
    f'Before split, there were {len(docs_before_split)} documents loaded, with average characters equal to {avg_char_before_split}.')
print(
    f'After split, there were {len(docs_after_split)} documents (chunks), with average characters equal to {avg_char_after_split} (average chunk length)')

# Text Embeddings with Hugging Face models
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# Now we can see how a sample embedding would look like for one of those chunks.
sample_embedding = np.array(huggingface_embeddings.
                            embed_query(docs_after_split[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)

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

# Now we have our vector store and retrieval system ready.
# We then need a large language model (LLM) to process information and answer the question.
# Open-source LLMs from Hugging Face
hf = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.8, "max_length": 10000})
query = """Explain the process decomposition model for resiliency in microservices for login application"""
result = hf.invoke(query)
print(result.format())
