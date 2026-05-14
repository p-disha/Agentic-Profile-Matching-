import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tools

# Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    job_requirements: Optional[Dict[str, Any]]
    candidate_shortlist: List[Dict[str, Any]]
    reasoning_logs: List[str]
    current_stage: str # "init", "parsed", "searched", "ranked", "reported", "feedback"
    jd_text: Optional[str]

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Nodes
def parse_jd(state: AgentState):
    """Identify if the user provided a JD or query."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    if os.environ.get("MOCK_MODE") == "1":
        intent = "JD"
        if "compare" in last_message.lower():
            intent = "COMPARE"
        elif "why" in last_message.lower():
            intent = "EXPLAIN"
        elif "actually" in last_message.lower() or "also" in last_message.lower():
            intent = "FEEDBACK"
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Determine if the following user input is a Job Description, a search query for candidates, a request to compare candidates, a question about rankings, or general feedback. Respond with ONLY one of: 'JD', 'QUERY', 'COMPARE', 'EXPLAIN', 'FEEDBACK'"),
            ("user", "{input}")
        ])
        chain = prompt | llm
        intent = chain.invoke({"input": last_message}).content.strip()
    
    logs = state.get("reasoning_logs", [])
    logs.append(f"Intent detected: {intent}")
    
    return {"current_stage": intent, "reasoning_logs": logs}

def extract_requirements(state: AgentState):
    """Extract requirements using the tool."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    reqs = tools.extract_requirements(last_message)
    
    logs = state.get("reasoning_logs", [])
    logs.append(f"Extracted requirements: {reqs}")
    
    return {
        "job_requirements": reqs, 
        "jd_text": last_message,
        "current_stage": "parsed",
        "reasoning_logs": logs
    }

def search_resumes(state: AgentState):
    """Search resumes based on requirements."""
    reqs = state.get("job_requirements", {})
    if not reqs:
        # Fallback if no explicit JD was parsed
        reqs = tools.extract_requirements(state["messages"][-1].content)
        
    candidates = tools.search_resumes(reqs)
    
    logs = state.get("reasoning_logs", [])
    logs.append(f"Found {len(candidates)} candidates.")
    
    return {
        "candidate_shortlist": candidates,
        "current_stage": "searched",
        "reasoning_logs": logs
    }

def rank_candidates(state: AgentState):
    """Rank candidates (Multi-round screening logic)."""
    candidates = state.get("candidate_shortlist", [])
    # Second round: Deep analysis mock
    # Just selecting top 3 for this mock
    top_candidates = sorted(candidates, key=lambda x: x.get("match_score", 0), reverse=True)[:3]
    
    logs = state.get("reasoning_logs", [])
    logs.append(f"Ranked candidates, selected top {len(top_candidates)}.")
    
    return {
        "candidate_shortlist": top_candidates,
        "current_stage": "ranked",
        "reasoning_logs": logs
    }

def generate_report(state: AgentState):
    """Generate final report for user."""
    candidates = state.get("candidate_shortlist", [])
    if not candidates:
        msg = "No candidates found matching the criteria."
    else:
        msg = "Here are the top candidates matching your criteria:\n\n"
        for i, c in enumerate(candidates):
            msg += f"{i+1}. **{c['name']}** (Score: {c.get('match_score', 0)})\n"
            msg += f"   - Experience: {c['experience']} years\n"
            msg += f"   - Skills: {', '.join(c['skills'])}\n"
            msg += f"   - Summary: {c['summary']}\n\n"
            
    return {"messages": [AIMessage(content=msg)], "current_stage": "reported"}

def handle_feedback(state: AgentState):
    """Handle conversational refinement or questions."""
    messages = state["messages"]
    intent = state.get("current_stage")
    last_user_msg = messages[-1].content
    
    if intent == "COMPARE":
        # Mock logic to extract IDs from message or just compare top 2
        candidates = state.get("candidate_shortlist", [])
        if len(candidates) >= 2:
            ids = [candidates[0]["id"], candidates[1]["id"]]
            report = tools.compare_candidates(ids)
            return {"messages": [AIMessage(content=report)]}
        else:
            return {"messages": [AIMessage(content="Not enough candidates in shortlist to compare.")]}
            
    elif intent == "EXPLAIN":
        # Explainability
        if os.environ.get("MOCK_MODE") == "1":
            response_content = "Jane ranked higher because her React experience and TypeScript skills match your criteria better."
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an HR assistant. The user is asking to explain the candidate rankings. The current shortlist is: {shortlist}. Answer the user's question concisely."),
                ("user", "{input}")
            ])
            chain = prompt | llm
            response_content = chain.invoke({"input": last_user_msg, "shortlist": state.get("candidate_shortlist", [])}).content
        return {"messages": [AIMessage(content=response_content)]}
        
    elif intent == "FEEDBACK":
        # Allow refinement
        if os.environ.get("MOCK_MODE") == "1":
            response_content = "I've noted the new requirements. Searching again..."
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an HR assistant. The user is providing feedback or new requirements: {input}. Acknowledge the feedback and state that the search will be updated."),
                ("user", "{input}")
            ])
            chain = prompt | llm
            response_content = chain.invoke({"input": last_user_msg}).content
        return {"messages": [AIMessage(content=response_content)], "current_stage": "feedback_received"}

    return {"messages": [AIMessage(content="I didn't quite understand that. Could you provide a JD or a query?")]}

# Edge logic
def route_after_parse(state: AgentState):
    intent = state["current_stage"]
    if intent == "JD" or intent == "QUERY":
        return "extract_requirements"
    elif intent in ["COMPARE", "EXPLAIN", "FEEDBACK"]:
        return "human_feedback_loop"
    return "human_feedback_loop"

def route_after_feedback(state: AgentState):
    if state.get("current_stage") == "feedback_received":
        return "extract_requirements" # Loop back
    return END

# Build Graph
builder = StateGraph(AgentState)

builder.add_node("parse_jd", parse_jd)
builder.add_node("extract_requirements", extract_requirements)
builder.add_node("search_resumes", search_resumes)
builder.add_node("rank_candidates", rank_candidates)
builder.add_node("generate_report", generate_report)
builder.add_node("human_feedback_loop", handle_feedback)

builder.add_edge(START, "parse_jd")
builder.add_conditional_edges("parse_jd", route_after_parse)
builder.add_edge("extract_requirements", "search_resumes")
builder.add_edge("search_resumes", "rank_candidates")
builder.add_edge("rank_candidates", "generate_report")
builder.add_edge("generate_report", END)
builder.add_conditional_edges("human_feedback_loop", route_after_feedback)

graph = builder.compile()

def run_agent(user_input: str, thread_id: str = "default"):
    # Wrapper for Streamlit
    config = {"configurable": {"thread_id": thread_id}}
    # Note: State persistence is needed for full LangGraph history.
    # For this assignment, we'll manage state externally in Streamlit 
    # or pass it all into `invoke` if using in-memory Checkpointer.
    # We will just invoke the graph.
    pass
