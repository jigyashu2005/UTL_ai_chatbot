import sqlite3
import os

DB_FILE = "chat_history.db"

def verify_db():
    if not os.path.exists(DB_FILE):
        print(f"❌ Database file '{DB_FILE}' not found.")
        return

    print(f"✅ Database found: {DB_FILE}\n")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row

    # 1. Show Schema
    print("--- 1. Database Schema ---")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            # Format: (cid, name, type, notnull, dflt_value, pk)
            for col in columns:
                print(f"  - {col[1]} ({col[2]}) {'PRIMARY KEY' if col[5] else ''}")
    except Exception as e:
        print(f"Error reading schema: {e}")

    # 2. Show Data
    print("\n--- 2. Current Data ---")
    try:
        # Sessions
        print("\n[Sessions Table]")
        cursor.execute("SELECT * FROM Sessions")
        sessions = cursor.fetchall()
        if sessions:
            print(f"{'session_id':<38} | {'created_at':<20} | {'participants'}")
            print("-" * 80)
            for s in sessions:
                print(f"{s['session_id']:<38} | {s['created_at']:<20} | {s['participants']}")
        else:
            print("  (Empty)")

        # Messages
        print("\n[Messages Table]")
        cursor.execute("SELECT * FROM Messages")
        messages = cursor.fetchall()
        if messages:
            print(f"{'role':<10} | {'name':<15} | {'content'}")
            print("-" * 80)
            for m in messages:
                content = m['content'][:50] + "..." if len(m['content']) > 50 else m['content']
                print(f"{m['role']:<10} | {m['name'] or 'None':<15} | {content}")
        else:
            print("  (Empty)")
            
    except Exception as e:
        print(f"Error reading data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    verify_db()
