import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Global state for vectorstore
vectorstore = None

class Requirements(BaseModel):
    must_have: list[str] = Field(description="Must-have requirements")
    nice_to_have: list[str] = Field(description="Nice-to-have requirements")
    summary: str = Field(description="Brief summary of the role")

def init_rag():
    """Initialize the RAG vector store from resumes in the downloads folder."""
    global vectorstore
    if vectorstore is not None:
        return
        
    resume_dir = r"C:\Users\disha\Downloads"
    resume_files = [
        "resume_alex_davis.docx",
        "resume_david_wilson.pdf",
        "resume_emma_brown.txt",
        "resume_jane_smith.pdf",
        "resume_john_doe.pdf",
        "resume_maria_garcia.txt"
    ]
    
    documents = []
    for filename in resume_files:
        filepath = os.path.join(resume_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding='utf-8')
            else:
                continue
                
            docs = loader.load()
            # Add metadata for candidate identification
            candidate_id = filename.split('.')[0]
            for d in docs:
                d.metadata["candidate_id"] = candidate_id
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not documents:
        # Fallback if files aren't found for some reason
        return
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

def extract_requirements(jd: str) -> dict:
    """Extract structured requirements from a JD using LLM."""
    if os.environ.get("MOCK_MODE") == "1":
        return {
            "must_have": ["React", "3+ years experience"],
            "nice_to_have": ["TypeScript", "AWS"],
            "summary": "Looking for an experienced frontend developer."
        }
        
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm = llm.with_structured_output(Requirements)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the job requirements from the following description."),
        ("user", "{input}")
    ])
    chain = prompt | structured_llm
    
    try:
        reqs = chain.invoke({"input": jd})
        return {
            "must_have": reqs.must_have,
            "nice_to_have": reqs.nice_to_have,
            "summary": reqs.summary
        }
    except Exception as e:
        return {
            "must_have": ["Requirements extraction failed"],
            "nice_to_have": [],
            "summary": str(e)
        }

def search_resumes(requirements: dict) -> list:
    """Use RAG to search resumes based on requirements."""
    if os.environ.get("MOCK_MODE") == "1":
        return [
            {
                "id": "Jane Smith",
                "name": "Jane Smith",
                "skills": ["React", "JavaScript", "TypeScript", "Node.js"],
                "experience": 4,
                "summary": "Frontend developer with strong React experience.",
                "match_score": 90
            },
            {
                "id": "John Doe",
                "name": "John Doe",
                "skills": ["React", "Python", "AWS", "SQL"],
                "experience": 3,
                "summary": "Full stack engineer transitioning to frontend heavy roles.",
                "match_score": 80
            }
        ]
        
    init_rag()
    if vectorstore is None:
        return [{"id": "error", "name": "Error loading resumes", "skills": [], "experience": 0, "summary": "Could not initialize RAG."}]
        
    query = " ".join(requirements.get("must_have", [])) + " " + " ".join(requirements.get("nice_to_have", []))
    if not query.strip():
        query = requirements.get("summary", "experienced candidate")
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    
    # Aggregate chunks by candidate
    candidate_map = {}
    for doc in docs:
        c_id = doc.metadata.get("candidate_id", "unknown")
        if c_id not in candidate_map:
            candidate_map[c_id] = {
                "id": c_id,
                "name": c_id.replace("resume_", "").replace("_", " ").title(),
                "summary": "",
                "skills": ["Extracted from resume"],
                "experience": "N/A"
            }
        candidate_map[c_id]["summary"] += doc.page_content + " "
        
    # Summarize with LLM to keep context small
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    results = []
    
    for c_id, c_data in candidate_map.items():
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this candidate's resume content highlighting their experience and skills relevant to: {query}. Keep it to 2-3 sentences."),
            ("user", "{content}")
        ])
        chain = summary_prompt | llm
        try:
            summary = chain.invoke({"query": query, "content": c_data["summary"]}).content
            c_data["summary"] = summary
            results.append(c_data)
        except:
            results.append(c_data)
            
    # Mock scores for ranking based on position in retrieval
    for i, c in enumerate(results):
        c["match_score"] = 100 - (i * 10)
        
    return sorted(results, key=lambda x: x["match_score"], reverse=True)

def compare_candidates(candidate_ids: list) -> str:
    """Compare specific candidates head-to-head using LLM."""
    if not candidate_ids:
        return "No candidates provided to compare."
        
    if os.environ.get("MOCK_MODE") == "1":
        return "Head-to-Head Comparison:\n- **Jane Smith**: 4 years exp, strong with TypeScript.\n- **John Doe**: 3 years exp, has AWS knowledge but less frontend focus."
        
    # We would retrieve their full text again for a deep comparison
    init_rag()
    if vectorstore is None:
        return "Vectorstore not initialized."
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    content = ""
    for cid in candidate_ids:
        docs = retriever.invoke(cid)
        text = "\n".join([d.page_content for d in docs if d.metadata.get("candidate_id") == cid])
        content += f"\n--- Candidate {cid} ---\n{text}\n"
        
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Provide a detailed head-to-head comparison of these candidates based on their resumes. Highlight strengths and weaknesses of each."),
        ("user", "{content}")
    ])
    
    try:
        response = (prompt | llm).invoke({"content": content})
        return response.content
    except Exception as e:
        return str(e)

def generate_interview_questions(candidate_id: str) -> str:
    """Generate screening questions for a candidate using LLM."""
    if os.environ.get("MOCK_MODE") == "1":
        return f"Interview Questions for {candidate_id}:\n1. Can you describe a complex problem you solved using React?\n2. What architectural patterns do you favor?"
        
    init_rag()
    if vectorstore is None:
        return "Vectorstore not initialized."
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(candidate_id)
    content = "\n".join([d.page_content for d in docs if d.metadata.get("candidate_id") == candidate_id])
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 3 specific, technical screening questions for this candidate based on their resume to verify their claimed skills."),
        ("user", "{content}")
    ])
    
    try:
        response = (prompt | llm).invoke({"content": content})
        return response.content
    except Exception as e:
        return str(e)
