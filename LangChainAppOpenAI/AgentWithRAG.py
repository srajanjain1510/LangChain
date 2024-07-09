import datasets
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# We first load a knowledge base on which we want to perform RAG:
# this dataset is a compilation of the documentation pages
# for many huggingface packages, stored as markdown.
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

source_docs = [
    Document(
        page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}
    ) for doc in knowledge_base
]

docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(source_docs)[:1000]

embedding_model = HuggingFaceEmbeddings("thenlper/gte-small")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model
)
