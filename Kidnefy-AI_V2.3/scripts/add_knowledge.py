import os
from pathlib import Path
from src.rag.gemini_rag import KidneyKnowledgeBase

def add_new_medical_documents(folder_path="medical_documents"):
    """Reads all .txt files from a folder and adds them to the Knowledge Base."""
    folder = Path(folder_path)
    
    # Create folder if it doesn't exist
    if not folder.exists():
        print(f"Creating folder: {folder_path} ...")
        folder.mkdir(parents=True, exist_ok=True)
        print("Folder created! Please put your .txt medical files inside it and run this script again.")
        return

    # Initialize Knowledge Base
    print("Loading Knowledge Base Database...")
    kb = KidneyKnowledgeBase()
    
    txt_files = list(folder.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in '{folder_path}' folder. Please add some!")
        return
        
    print(f"Found {len(txt_files)} medical documents. Adding to AI Brain...")
    
    added_count = 0
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            doc_id = file_path.stem.replace(" ", "_").lower()
            source_name = file_path.name
            
            # Add to ChromaDB
            kb.add_document(doc_id=doc_id, content=content, source=source_name)
            print(f" ✅ Successfully learned from: {source_name}")
            added_count += 1
            
        except Exception as e:
            print(f" ❌ Failed to read {file_path.name}: {e}")
            
    print(f"\nAwesome! The AI Chatbot is now smarter. {added_count} new documents added to its memory.")

if __name__ == "__main__":
    add_new_medical_documents()
