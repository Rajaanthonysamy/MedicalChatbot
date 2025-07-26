from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdf_documents(directory):
    """
    Load PDF documents from a specified directory.
    """
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_doc(docs : List[Document]) -> List[Document]:
    """
    Filter documents to keep only the minimal content.
    """
    filtered_docs : List[Document] = []
    for doc in docs:
        src= doc.metadata.get("source")
        filtered_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": src
                }
            )
        )
    return filtered_docs


def text_splitter(minimal_docs):
    """
    Split the text of documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    split_chunks = text_splitter.split_documents(minimal_docs)
    return split_chunks


def download_embeddings():
    """
    Download and return HuggingFace embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embeddings