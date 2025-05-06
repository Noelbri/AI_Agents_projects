import operator
import os
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()
api_key =os.getenv("GROQ_API_KEY")
#Define diagnostic and action tools
@tool
def check_cpu_usage():
    """Simulates checking the CPU usage."""
    return "CPU Usage is 85%."
@tool
def check_disk_space():
    """Simulates checking the disk space."""
    return "Disk space is 10% free."
@tool
def check_network():
    """Simulates checking the network connectivity."""
    return "Network connectivity is stable."
@tool
def restart_server():
    """Simulates restarting the server."""
    return "Server restarted successfully."
#Setup Tools
tools = [check_cpu_usage, check_disk_space, check_network, restart_server]
#Set up the model and agent executor
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an IT diagnostics agent."),
        ("placeholder", "{messages}")
    ]
)
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)
#Define the plan and Execution structure
class PlanExecute(TypedDict):
    input:str
    plan:List[str]
    past_steps:Annotated[List[Tuple], operator.add]
    response:str
class Plan(BaseModel):
    steps:List[str]= Field(description="Tasks to check and resolve server issues")
class Response(BaseModel):
    response:str
class Act(BaseModel):
    action:Union[Response, Plan] = Field(description="Action to perform. If you want to respond to user, use Response. If you need to further use tools to get the answer, use Plan.")
#Planning steps
planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """For the given server issue, create a step-by-step diagnostic plan including CPU, disk, and network checks, followed by a server restart if necessary."""),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt|ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key).with_structured_output(Plan)
#Replanning step
replanner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a replanning assistant. For the given task and completed steps, return the next action as either a Plan (with steps) or a Response (with a message to the user). 

Respond ONLY in JSON with this structure:

{{  
  "action": {{
    "__type__": "Response", 
    "message": "..." 
  }}
}}

OR

{{  
  "action": {{
    "__type__": "Plan", 
    "steps": ["step1", "step2"]
  }}
}}"""),
    ("human", "Original task: {input}\nCompleted steps: {past_steps}")
])


replanner = replanner_prompt|ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key).with_structured_output(Act)
#Execution step function
async def execute_step(state:PlanExecute):
    plan = state["plan"]
    task = plan[0]
    task_formatted = f"Executing step: {task}."
    agent_response = await agent_executor.ainvoke({"messages":[("user", task_formatted)]})
    return {
        "past_steps":[(task, agent_response["messages"][-1].content)],
    }
#Planning step function
async def plan_step(state:PlanExecute):
    plan = await planner.ainvoke({"messages":[("user",state["input"])]})
    return {"plan":plan.steps}
# re-planning step function (in case execution needs adjustment)
async def replan_step(state:PlanExecute):
    output = await replanner.ainvoke(state)
    #If the replanner decides to return a response, we use it as the final answer 
    if isinstance(output.action, Response): #Final response provided
        return {"response":output.action.response}  #Return the response to the user
    else:
        # Otherwise, we continue with the new plan (if replanning suggests more steps)
        return {"plan":output.action.steps}
#Conditional check for ending
def should_end(state:PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"
#Build the workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
# Add edges to transition between nodes 
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])
#Compile the workflow into an executable application
app = workflow.compile()
#Example of running the agent
config = {"recursion_limit":50}
import asyncio
# Function to run the Plan-and-Execute agent
async def run_plan_and_execute():
    # Input from user
    inputs = {"input":"Diagnose the server issue and restart if necessary"}
    # Run the Plan-and-Execute agent asynchronously
    async for event in app.astream(inputs, config=config):
        print(event)
#Run the async function
if __name__=="__main__":
    asyncio.run(run_plan_and_execute())
