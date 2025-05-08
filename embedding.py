from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
#Define sample documents as Document objects
documents = [
    Document(page_content="LangChain is a framework for building context-aware reasoning applications."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="The quick brown fox jumps over the lazy dog.")
]
#Index documents in Chroma vector store
db = Chroma.from_documents(documents, embedding=embeddings)
print("Documents indexed in Chroma successfully.")
#Define a retriever to fetch relevant documents
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":6})
#Define a prompt template for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are nuero, an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
    ("human", "Question: {question}\nContext: {context}\nAnswer:")
])
#Initialize the chat model
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)
#Set up the RAG chain pipeline
rag_chain = (
    {"context":retriever, "question":RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
#Use the RAG pipeline to answer a user question
question = "What did the fox do?"
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)