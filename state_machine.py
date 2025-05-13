import asyncio
from typing import List, TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#Index 3 websites by adding them to the vector DB
urls = [
    "https://github.com/facebookresearch/faiss",
    "https://github.com/facebookresearch/faiss/wiki",
    "https://github.com/facebookresearch/faiss/wiki/Faiss-indexes"
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250,
    chunk_overlap = 0
)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=HuggingFaceEmbeddings(),
)
retriever = vectorstore.as_retriever()

#Prepare the RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are nuero, an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
    ("human", "Question: {question}\nContext: {context}\nAnswer:")
])

model = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
rag_chain = (
    prompt | model | StrOutputParser()
)

#define the graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
       question: question
       generation: LLM generation
       documents: list of documents
    """
    question:str
    generation:str
    web_search:str
    documents:List[str]

#Retrieve node 
def retrieve(state):
    """
    Retrieve documents
    Args:
      state(dict): The current graph state
    Returns@
      state(dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    #Retrieval
    documents = retriever.invoke(question)
    return {"documents":documents, "question":question}

#Generate node
def generate(state):
    """
    Generate answer
    Args:
      state(dict): The current graph state
    Returns:
      state(dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    #RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

#Define the workflow
def create_workflow():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile(checkpointer=MemorySaver())

# Run the workflow
async def run_workflow():
    app = create_workflow()
    config = {
        "configurable": {"thread_id": "1"},
        "recursion_limit": 50
    }

    while True:
        user_input = input("type 'exit' to quit: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        inputs = {"question": user_input}

        try:
            async for event in app.astream(inputs, config=config, stream_mode="values"):
                if "error" in event:
                    print(f"Error: {event['error']}")
                    break
                print(event)
        except Exception as e:
            print(f"Workflow execution failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_workflow())
