import streamlit as st
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
import pathlib

# --- Configuration ---
# Point to the local Backend API
API_BASE_URL = "http://localhost:8000" 
PAGE_TITLE = "UTL AI Assistant"
PAGE_ICON = "🧠"

def get_api_client():
    """Initializes OpenAI client pointing to local backend."""
    return OpenAI(
        base_url=API_BASE_URL,
        api_key="dummy-key-for-local-proxy" 
    )

# --- UI Components ---
def setup_page():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
            html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
            .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #ffffff; }
            [data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-right: 1px solid rgba(255, 255, 255, 0.1); }
            .main-header { text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #ff00cc, #333399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; font-size: 3rem; margin-bottom: 2rem; }
            .stChatMessage { background-color: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 10px; margin-bottom: 5px; border: 1px solid rgba(255, 255, 255, 0.08); }
        </style>
    """, unsafe_allow_html=True)

def sidebar_logic():
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # 1. Session Management
        st.subheader("Sessions")
        try:
            sessions_resp = requests.get(f"{API_BASE_URL}/sessions")
            if sessions_resp.status_code == 200:
                sessions = sessions_resp.json()
                if sessions:
                    options = {s["session_id"]: f"{s['title'][:20]}... ({s['created_at'][:10]})" for s in sessions}
                    selected_id = st.selectbox("Load Session", options=list(options.keys()), format_func=lambda x: options[x])
                    
                    if st.button("Load Chat"):
                        st.session_state.session_id = selected_id
                        st.rerun()
                else:
                    st.caption("No history found.")
            else:
                st.error("Backend offline?")
        except Exception:
            st.error("Cannot connect to Backend.")

        if st.button("➕ New Session", type="primary"):
            try:
                # Create standard session
                resp = requests.post(f"{API_BASE_URL}/sessions", json={"participants": ["User", "AI"]})
                if resp.status_code == 200:
                    sid = resp.json()["session_id"]
                    st.session_state.session_id = sid
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to create: {e}")

        st.markdown("---")
        
        # 2. File Upload (RAG)
        st.subheader("📄 Knowledge Base")
        uploaded_files = st.file_uploader("Upload Docs (PDF/TXT)", accept_multiple_files=True)
        if uploaded_files and st.button("Ingest Files"):
            files_payload = [('files', (f.name, f.getvalue(), f.type)) for f in uploaded_files]
            try:
                with st.spinner("Ingesting..."):
                    resp = requests.post(f"{API_BASE_URL}/ingest", files=files_payload)
                    if resp.status_code == 200:
                        st.success(resp.json()["message"])
                    else:
                        st.error("Ingestion failed.")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Main App ---
def main():
    setup_page()
    st.markdown('<div class="main-header">UTL AI Assistant</div>', unsafe_allow_html=True)

    # Initialize State
    if "session_id" not in st.session_state:
        # Try to create a default one on startup
        try:
            resp = requests.post(f"{API_BASE_URL}/sessions", json={"participants": ["User", "AI"]})
            if resp.status_code == 200:
                st.session_state.session_id = resp.json()["session_id"]
        except:
             st.warning("Could not auto-create session. Is Backend running?")

    # Fetch and Display History
    if "session_id" in st.session_state:
        try:
            resp = requests.get(f"{API_BASE_URL}/sessions/{st.session_state.session_id}")
            if resp.status_code == 200:
                session_data = resp.json()
                messages = session_data.get("messages", [])
                
                for msg in messages:
                    role = msg["role"]
                    if role == "system": continue
                    
                    with st.chat_message(role):
                        st.markdown(msg["content"])
            else:
                st.error("Session fetch error.")
                manual_history = []
        except:
            pass

    # Chat Input
    if prompt := st.chat_input("Ask something..."):
        if "session_id" not in st.session_state:
            st.error("No active session.")
            return

        # Display User Message Immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send to API
        client = get_api_client()
        
        # We need to construct the messages for the API call.
        # But wait, our API endpoint expects a message list.
        # It handles retrieval.
        # We should send the history + new message?
        # Actually, the API refactor I did has a "stateless" looking input (messages list).
        # BUT it also takes a `model` param for session_id.
        # If I send JUST the last message with session_id in `model`, the backend will:
        # 1. Fetch history from DB.
        # 2. Add last msg.
        # 3. Do RAG.
        # 4. Save response.
        #
        # BUT, the backend `chat_completions` logic tries to use `request.messages`.
        # Simplest path: Send [User Message] + session_id in 'model'.
        # Backend will handle context.
        
        # HOWEVER, standard OpenAI clients send the whole history usually.
        # My backend `chat_completions` logic:
        # "1. Extract last user message... 3. Load session history... 4. Construct full prompt"
        # So I can just send the NEW message.
        
        try:
            with st.chat_message("assistant"):
                # We use stream=False for now as simplified in backend
                with st.spinner("Thinking..."):
                    completion = client.chat.completions.create(
                        model=st.session_state.session_id, # Passing Session ID here!
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    response_text = completion.choices[0].message.content
                    st.markdown(response_text)
                    
        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    sidebar_logic()
    main()
