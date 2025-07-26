from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import system_prompt
from src.helper import download_embeddings
import os 


app= Flask(__name__)

load_dotenv()

index_name="medical-chatbot"
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

ebbedding= embedding=download_embeddings()

doc_search= PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

prompt= ChatPromptTemplate.from_messages(
    [
    ('system', system_prompt),
    ('human' , "{input}")]
)

retriever=doc_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3})

model= ChatOpenAI(
    model_name="gpt-4o")

qus_answer_chain = create_stuff_documents_chain(model, prompt=prompt)
ret_chain=create_retrieval_chain(retriever,qus_answer_chain)


@app.route('/')
def  home():
    return render_template('chat.html')


@app.route('/get', methods=['POST',"GET"])
def ask():
   question=request.form.get('msg')
   response=ret_chain.invoke({"input": question})
   return str(response['answer'])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)