from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Internal Imports
import session_manager
from rag_engine import RAGEngine

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

WORKDIR = os.path.dirname(__file__)
RAG_DIR = os.path.join(WORKDIR, "knowledge_base")
TEMP_UPLOAD_DIR = os.path.join(WORKDIR, "temp_uploads")

# --- Initialize Systems ---
if not os.path.exists(RAG_DIR):
    os.makedirs(RAG_DIR)
if not os.path.exists(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR)

print(f"Initializing RAG System at {RAG_DIR}...")
rag_system = RAGEngine(use_openai_embeddings=False, storage_dir=RAG_DIR)

app = FastAPI(title="UTL Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI Client ---
def get_client() -> OpenAI:
    """Returns configured OpenAI client."""
    base_url = None
    if API_KEY and API_KEY.startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"
    return OpenAI(api_key=API_KEY, base_url=base_url)

client = get_client()

# --- Models ---
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[ChatMessage]
    stream: bool = False
    
    # Custom fields for session management
    user: Optional[str] = None # Can be used for session_id if strictly following OpenAI format is hard

class SessionCreate(BaseModel):
    participants: List[str]

# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "running", "service": "UTL Backend"}

@app.get("/sessions")
def list_sessions():
    return session_manager.list_sessions()

@app.post("/sessions")
def create_session(data: SessionCreate):
    session_id = session_manager.create_session(data.participants)
    return {"session_id": session_id}

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    data = session_manager.get_session_data(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data

@app.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    """Uploads and ingests files into the Knowledge Base."""
    temp_files = []
    try:
        for file in files:
            file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            temp_files.append(file_path)
        
        # Ingest
        rag_system.load_documents(temp_files)
        return {"status": "success", "message": f"Ingested {len(temp_files)} files."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass

@app.post("/chat/completions")
def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat endpoint with RAG and History integration.
    Expects 'model' field to potentially hold the 'session_id' for simplicity, 
    or we can use a custom header/field. 
    
    Strategy:
    1. Extract last user message.
    2. Retrieve relevant context (RAG).
    3. Load session history (from DB).
    4. Construct full prompt.
    5. Call OpenAI.
    6. Save response.
    """
    try:
        # 1. Parse Input
        if not request.messages:
             raise HTTPException(status_code=400, detail="No messages provided")
             
        latest_msg = request.messages[-1]
        if latest_msg.role != "user":
             # If last message isn't user, just pass through (unlikely usage)
             pass
             
        user_query = latest_msg.content
        session_id = request.model if len(request.model) > 20 else None # Basic heuristic for UUID in model field
        
        # 2. RAG Retrieval
        context_str = ""
        relevant_chunks = rag_system.retrieve_relevant_chunks(user_query, k=2)
        if relevant_chunks:
            context_str = rag_system.generate_context_string(relevant_chunks)
            print(f"RAG: Found {len(relevant_chunks)} chunks for '{user_query}'")
            
        # 3. Build Contextual System Prompt
        base_system_prompt = "You are a helpful AI assistant."
        rag_instruction = ""
        if context_str:
            rag_instruction = f"\n\nRELAVENT INFORMATION FROM UPLOADED DOCUMENTS:\n{context_str}\n\nINSTRUCTION: Answer the user's question using the information above. If the answer is not in the context, use your general knowledge but mention you couldn't find it in the documents."
        
        # 4. Construct API Messages
        # We need to rebuild the message list. 
        # OpenAI needs: [System, History..., User]
        
        api_messages = []
        
        # Check if system prompt exists in request
        has_system = any(m.role == "system" for m in request.messages)
        if not has_system:
             api_messages.append({"role": "system", "content": base_system_prompt + rag_instruction})
        else:
            # Inject RAG into existing system prompt
            for m in request.messages:
                if m.role == "system":
                    api_messages.append({"role": "system", "content": m.content + rag_instruction})
                else:
                    api_messages.append({"role": m.role, "content": m.content})
                    
        # If no system prompt was in messages, we need to add the rest
        if not has_system:
            for m in request.messages:
                api_messages.append({"role": m.role, "content": m.content})

        # 5. Call OpenAI
        print("Sending request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=api_messages,
            stream=False # Simplifying to non-stream for stability, can enable stream if needed
        )
        
        ai_content = response.choices[0].message.content
        
        # 6. Save to History (if session_id is valid)
        if session_id:
            session_manager.add_message(session_id, "assistant", ai_content, "AI")

        # Return serializable dict
        return json.loads(response.model_dump_json())

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
