from src.helper import load_pdf_documents , filter_to_minimal_doc, text_splitter, download_embeddings
import os 

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


extract_data=load_pdf_documents("data/")

extracted_sources=filter_to_minimal_doc(extract_data)

text_chnk=text_splitter(extracted_sources)

embedding=download_embeddings()




pinecone_db=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name="medical-chatbot"

if not pinecone_db.has_index(index_name):
    pinecone_db.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embeddings
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
        )
    )


index = pinecone_db.Index(index_name)

retriever=doc_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3})

doc_search = PineconeVectorStore.from_documents(
    documents=text_chnk,
    embedding=embedding,
    index_name=index_name
)