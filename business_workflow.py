import asyncio
import operator
import os
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union, Optional
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#Enhanced tools with error handling and input validation
@tool
def validate_expense_report(report_id:str)->str:
    """Validate an empoloyee's expense report."""
    if not report_id or not isinstance(report_id, str):
        return "Error: Invalid report ID provided"
    try:
        return f"Expense report{report_id} is valid"
    except Exception as e:
        return f"Error validating expense report:{str(e)}"
@tool
def check_policy_compliance(report_id:str)->str:
    """Checks whether the report complies with company policy."""
    if not report_id or not isinstance(report_id, str):
        return "Error:Invalid report ID provided"
    try:
        return f"Report {report_id} complies with company policy."
    except Exception as e:
        return f"Error checking policy complience:{str(e)}"
@tool
def route_to_manager(report_id:str)->str:
    """Routes the report to the manager for approval"""
    if not report_id or not isinstance(report_id, str):
        return "Error: Invalid report ID provided"
    try:
        return f"Report {report_id} has been routed to the manager."
    except Exception as e:
        return f"Error routing to manager:{str(e)}"
@tool
def notify_employee(report_id:str,status:str)->str:
    """Notifies the employee of the report's status"""
    if not report_id or not status:
        return "Error:Invalid report ID provided"
    try:
        return f"Employee notified that report {report_id} is {status}."
    except Exception as e:
        return f"Error notifying employee: {str(e)}"
tools = [validate_expense_report, check_policy_compliance, route_to_manager, notify_employee]
#Enhanced state management
class PlanExecute(TypedDict):
    input:str 
    plan:List[str]
    past_steps:Annotated[List[Tuple], operator.add]
    response:Optional[str]
    error:Optional[str]
class Plan(BaseModel):
    steps:List[str] = Field(description="Numbered unique steps to follow, in order")
class Response(BaseModel):
    response:str = Field(description="Response to user")
class Act(BaseModel):
    action:Union[Response, Plan] = Field(description="Action to perform")
#Improved system prompts 
SYSTEM_PROMPT = """You are an expense report processing assistant. Your task is to validate and process expense reports
using the available tools. Always extract the report ID from the input and use it consistently across all steps.
Available tools:
1. validate_expense_report-Validates the report
2. check_policy_compliance-Checks policy compliance
3. route_to_manager-Routes to manager
4. notify_employee-Notifies the employee
Ensure each step is completed before moving to the next one."""
#Enhanced agent setup
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)
#Improved planning step
async def plan_steps(state:PlanExecute)->dict:
    try:
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{messages}")
        ])
        planner = planner_prompt|llm.with_structured_output(Plan)
        plan = await planner.ainvoke({"messages":[("user", state["input"])]})
        return {"plan": plan.steps}
    except Exception as e:
        return {"error": f"Planning error:{str(e)}"}
#Improved execution step with error handling
async def execute_step(state:PlanExecute)->dict:
    try:
        if "error" in state:
            return {"response": f"Workflow failed:{state['error']}"}
        plan = state["plan"]
        if not plan:
            return {"response":"No plan steps available to execute"}
        task = plan[0]
        agent_response = await agent_executor.ainvoke({"mesages":[("user", task)]})
        return {"past_steps":[(task, agent_response["messages"][-1].content)]}
    except Exception as e:
        return {"error": f"Execution error: {str(e)}"}
#Enhanced replanning with better error handling
async def replan_step(state:PlanExecute)->dict:
    try:
        if "error" in state:
            return {"response": f"Workflow failed: {state['error']}"}
        replanner_prompt = ChatPromptTemplate.from_template("""
                                                            Given the objective: {input}
                                                            Original plan: {plan}
                                                            Completed steps: {past_steps}
                                                            Please either:
                                                            1. Provide next steps if more work is needed
                                                            2. Provide a final response if the workflow is complete
                                                            Only include steps that still need to be done.
                                                            """)
        replanner = replanner_prompt|llm.with_structured_output(Act)
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response":output.action.response}
        return {"plan":output.action.steps}
    except Exception as e:
        return {"error":f"Replanning error: {str(e)}"}
# Setup workflow
def create_workflow():
    workflow = StateGraph(PlanExecute)
    #Add nodes
    workflow.add_node("planner", plan_steps)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    #Add edges
    workflow.add_edge(START,"planner")
    workflow.add_edge("planner","agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges(
        "replan",
        lambda s: END if ("response" in s or "error" in s) else "agent",
        [END]
    )
    return workflow.compile(checkpointer=MemorySaver())
async def run_workflow(report_id:str):
    app = create_workflow()
    config = {
        "configurable": {"thread_id":"1"},
        "recursion_limit": 50
    }
    inputs = {"input": f"Validate and process the expense report with report ID {report_id}"}
    try:
        async for event in app.astream(inputs, config=config, stream_mode="values"):
            if "error" in event:
                print(f"Error: {event['error']}")
                break
            print(event)
    except Exception as e:
        print(f"Workflow execution failed:{str(e)}")
if __name__=="__main__":
    asyncio.run(run_workflow("12345"))