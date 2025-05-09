import os 
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tiktoken
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#Load the document
file_path = os.path.join(os.getcwd(), "2025-Economic-Survey.pdf")
raw_documents = PyPDFLoader(file_path=file_path).load()
#Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
#Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
#Index the document chunks in chroma vector store
db = Chroma.from_documents(documents=documents, embedding=embeddings)
print("Document indexed in chroma successfully")
#Define a retriever for similarity search
retriever =db.as_retriever(search_type="similarity", search_kwargs={"k":5})
#Define the prompt template for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are nuero, an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
    ("human", "Question: {question}\nContext: {context}\nAnswer:")
])
#Initialize the model
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
#Set up the RAG chain
rag_chain = (
    {"context":retriever, "question":RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
while True:
    question = input("Ask a question (type 'quit' or 'exit' to stop): ").strip()
    if question.lower() in ['quit', 'exit']:
        print("Exiting the program.")
        break

    # Generate response using rag_chain
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()  # Add newline after the full response
