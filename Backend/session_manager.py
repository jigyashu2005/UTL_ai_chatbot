import sqlite3
import uuid
import json
from datetime import datetime
import os
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

DB_FILE = os.path.join(os.path.dirname(__file__), "chat_history.db")

@contextmanager
def get_db_connection():
    """Context manager for SQLite database connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row # Enable accessing columns by name
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the SQLite database with Sessions and Messages tables."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("PRAGMA foreign_keys = ON;")
        
        # Create Sessions Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS Sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                participants TEXT
            )
        ''')
        
        # Create Messages Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS Messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                name TEXT,
                image_path TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES Sessions(session_id) ON DELETE CASCADE
            )
        ''')
        conn.commit()

def generate_uuid() -> str:
    """Generates a standard UUID v4 string."""
    return str(uuid.uuid4())

def create_session(participants: Optional[List[str]] = None) -> str:
    """Creates a new session and returns its ID."""
    session_id = generate_uuid()
    participants_json = json.dumps(participants or [])
    created_at = datetime.now().isoformat()
    
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO Sessions (session_id, created_at, participants) VALUES (?, ?, ?)",
            (session_id, created_at, participants_json)
        )
        conn.commit()
    
    return session_id

def update_participants(session_id: str, participants: List[str]):
    """Updates the participants list for an existing session."""
    participants_json = json.dumps(participants or [])
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE Sessions SET participants = ? WHERE session_id = ?",
            (participants_json, session_id)
        )
        conn.commit()

def add_message(session_id: str, role: str, content: str, name: Optional[str] = None) -> str:
    """Adds a new message to the database."""
    message_id = generate_uuid()
    created_at = datetime.now().isoformat()
    
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO Messages (message_id, session_id, role, content, name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, session_id, role, content, name, created_at)
        )
        conn.commit()
    return message_id

def get_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full session data including participants and messages."""
    with get_db_connection() as conn:
        # Get Session Info
        cursor = conn.execute("SELECT * FROM Sessions WHERE session_id = ?", (session_id,))
        session_row = cursor.fetchone()
        
        if not session_row:
            return None
            
        participants = json.loads(session_row["participants"])
        
        # Get Messages
        cursor = conn.execute("SELECT * FROM Messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
        message_rows = cursor.fetchall()
        
        messages = []
        for row in message_rows:
            msg = {
                "role": row["role"],
                "content": row["content"]
            }
            if row["name"]:
                msg["name"] = row["name"]
            messages.append(msg)
            
        return {
            "session_id": session_id,
            "participants": participants,
            "messages": messages
        }

def list_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    """Lists recent session summaries."""
    with get_db_connection() as conn:
        # Get session_id, created_at, and the first user message content as title
        cursor = conn.execute(f'''
            SELECT s.session_id, s.created_at, 
                   COALESCE((SELECT content FROM Messages m WHERE m.session_id = s.session_id AND m.role = 'user' ORDER BY m.created_at ASC LIMIT 1), 'New Chat') as title
            FROM Sessions s 
            ORDER BY s.created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        
        return [{"session_id": row["session_id"], "created_at": row["created_at"], "title": row["title"]} for row in rows]

# Auto-initialize DB on import
init_db()
