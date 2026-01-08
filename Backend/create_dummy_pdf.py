import fitz

doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "UTL Solar is a leading company in renewable energy.\nThis is a test document for RAG system.\nPage 1 content.")
page = doc.new_page()
page.insert_text((50, 50), "Solar inverters are crucial components.\nThey convert DC to AC.\nPage 2 content details about technology.")
doc.save("test_rag_doc.pdf")
print("PDF created: test_rag_doc.pdf")
