import sys
from pathlib import Path

# Add src and config to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import os

# Fix: manually load .env 
env_path = Path(".env")
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from src.rag.gemini_rag import GeminiRAG

st.set_page_config(page_title="AI Medical Assistant", page_icon="", layout="centered")

st.title(" Kidney AI Medical Assistant")
st.caption("Ask questions about kidney disease, your lab results, or medical terms.")

# Initialize RAG in session state
if "rag" not in st.session_state:
    with st.spinner("Initializing AI Model..."):
        try:
            st.session_state.rag = GeminiRAG()
            st.session_state.ready = True
        except Exception as e:
            st.error(f"Failed to load Gemini API: {e}")
            st.session_state.ready = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if st.session_state.get("ready"):
    # Output language selection
    language = st.radio("Select Response Language / اختر لغة الرد:", ["English", "العربية (Arabic)"], horizontal=True)
    
    if prompt := st.chat_input("What is CKD Stage 3? / ما هي المرحلة الثالثة لمرض الكلى؟"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing medical context..."):
                try:
                    # Append language instruction to the prompt contextually (hidden from chat UI)
                    lang_instruction = " (Please ensure your response is entirely in English.)" if language == "English" else " (أرجو أن تكون إجابتك بالكامل باللغة العربية الطبية السليمة.)"
                    augmented_prompt = prompt + lang_instruction
                    
                    result = st.session_state.rag.ask(augmented_prompt)
                    answer = result.get('answer', 'Sorry, I could not generate an answer.')
                    sources = result.get('sources', [])
                    
                    response_md = answer
                    if sources:
                        source_names = [s.get('source', str(s)) if isinstance(s, dict) else str(s) for s in sources]
                        response_md += f"\n\n**Sources:** {', '.join(source_names)}"
                    
                    message_placeholder.markdown(response_md)
                    st.session_state.messages.append({"role": "assistant", "content": response_md})
                
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    if not st.runtime.exists():
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])

