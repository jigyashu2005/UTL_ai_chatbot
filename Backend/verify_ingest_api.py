import requests
import os

files = [
    ('files', ('test_doc.pdf', open('test_doc.pdf', 'rb'), 'application/pdf')),
    ('files', ('test_doc.docx', open('test_doc.docx', 'rb'), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'))
]

print("Sending ingestion request...")
response = requests.post("http://localhost:8000/ingest", files=files)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
