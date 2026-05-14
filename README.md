# LangGraph Candidate Matching Agent Walkthrough

I have successfully completed the implementation of the AI candidate matching agent according to the assignment requirements.

## What Was Accomplished

1. **Agent Architecture (LangGraph)**:
   - Created `matching_agent.py` using `langgraph` to manage the state machine and conversation history.
   - Designed an `AgentState` that tracks messages, extracted job requirements, shortlists, reasoning logs, and the current screening stage.
   - Implemented nodes for parsing intent, extracting requirements, searching resumes, ranking candidates, generating reports, and handling human feedback loops.

2. **RAG Tooling Integration**:
   - Created `tools.py` with full Retrieval-Augmented Generation (RAG) capabilities.
   - The tool dynamically loads resumes (PDF, DOCX, TXT) from `C:\Users\disha\Downloads`.
   - Utilizes `FAISS` and OpenAI embeddings to index the resumes.
   - Uses OpenAI (`gpt-3.5-turbo`) with structured outputs to extract requirements and perform head-to-head comparisons.

3. **Interactive UI**:
   - Developed `app.py` using Streamlit.
   - Integrated a secure API Key prompt in the sidebar. The user must provide their OpenAI API key to initialize the application.
   - Implemented a conversational chat interface with an "Agent Brain" sidebar that displays internal reasoning and extracted requirements in real-time.

4. **Documentation**:
   - Provided `state_machine.mermaid` detailing the exact workflow visually.
   - Provided `test_scenarios.md` containing 5 comprehensive chat flows (standard JD, refinement, comparison, explainability, interview questions).

## Video Demonstration

Below is the automated recording showing the agent performing requirement extraction and candidate comparisons in real-time.

![Agent Demo Recording](agent_demo.webp)

## How to Run the App Yourself

1. Ensure your active workspace is `C:\Users\disha\.gemini\antigravity\scratch\langgraph_agent`.
2. Ensure you have activated your environment or have the packages installed. I've already run `pip install -r requirements.txt`.
3. In your terminal, run:
```bash
streamlit run app.py
```
4. Enter your OpenAI API key in the sidebar, and you can chat with the agent based on the resumes in your Downloads folder!

