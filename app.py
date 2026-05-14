import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="AI Recruiter Matcher", layout="wide")

st.title("🤝 AI Recruiter Candidate Matcher")
st.markdown("Enter a job description, search query, or ask for candidate comparisons.")

# API Key input
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to continue. (Use 'MOCK_KEY_FOR_VIDEO' to run in mock mode for demo purposes).")
    st.stop()
elif api_key in ["MOCK_KEY_FOR_VIDEO", "sk-mock-key-for-video", "mock"]:
    os.environ["MOCK_MODE"] = "1"
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-video"
    from matching_agent import graph
else:
    os.environ["MOCK_MODE"] = "0"
    os.environ["OPENAI_API_KEY"] = api_key
    # Only import graph after API key is set so LangChain components initialize correctly
    from matching_agent import graph

# Initialize session state for LangGraph state
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "job_requirements": None,
        "candidate_shortlist": [],
        "reasoning_logs": [],
        "current_stage": "init",
        "jd_text": None
    }

# Sidebar for debug/explainability
with st.sidebar:
    st.header("Agent Brain 🧠")
    if st.session_state.agent_state.get("job_requirements"):
        st.subheader("Extracted Requirements")
        st.json(st.session_state.agent_state["job_requirements"])
    
    if st.session_state.agent_state.get("reasoning_logs"):
        st.subheader("Reasoning Logs")
        for log in st.session_state.agent_state["reasoning_logs"]:
            st.text(f"> {log}")

# Display chat history
for msg in st.session_state.agent_state["messages"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input
if prompt := st.chat_input("Type your JD or question here..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Append to state
    st.session_state.agent_state["messages"].append(HumanMessage(content=prompt))
    
    # Run graph
    with st.spinner("Agent is thinking..."):
        # We invoke the graph with the current state
        result_state = graph.invoke(st.session_state.agent_state)
        
        # Update session state with the result
        st.session_state.agent_state = result_state
        
        # Display latest AI message
        if st.session_state.agent_state["messages"] and isinstance(st.session_state.agent_state["messages"][-1], AIMessage):
            with st.chat_message("assistant"):
                st.write(st.session_state.agent_state["messages"][-1].content)
        st.rerun()
