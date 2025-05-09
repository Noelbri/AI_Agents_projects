import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tiktoken
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
#Load and process Local Document
file_path = os.path.join(os.getcwd(), "Building LLM Powered Applic_ (Z-Library) (1).pdf")
pdf_loader = PyPDFLoader(file_path)
pdf_documents = pdf_loader.load()
#Split PDF into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_pdf_documents = text_splitter.split_documents(pdf_documents)
#Load and process Web Content
urls = [
    "https://github.com/facebookresearch/faiss",
    "https://github.com/facebookresearch/faiss/wiki"
]
#Load web content
web_loader = WebBaseLoader(web_paths=urls)
web_documents = web_loader.load()

#Split web documents into chunks
split_web_documents = text_splitter.split_documents(web_documents)
#Combine Local and Web Documents
all_documents = split_pdf_documents + split_web_documents
#Index all documents in Chroma
db = Chroma.from_documents(documents=all_documents, embedding=embeddings)
print("All documents indexed in Chroma successfully.")

#Define a retriever to fetch relevant documents from the combined sources
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})
#Define the prompt template for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are nuero, an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
    ("human", "Question: {question}\nContext: {context}\nAnswer:")
])
#Intialize the model
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
#Set up the RAG
rag_chain = (
    {"context":retriever, "question":RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
while True:
    question = input("You:").strip()
    if question.lower() in ['quit', 'exit']:
        print("Exiting the program.")
        break
    #Generate response using rag_chain
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()
    