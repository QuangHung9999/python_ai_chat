# The Notgpt

Streamlit app to interact with OpenAI GPT-4o, with text, images and audio (using Whisper and TTS).

---

To run the app locally:

git clone it

`pip install -r requirements.txt`

`streamlit run app.py`

```
project_root/  # This is "python_ai_chat" directory
├── app/  
│   ├── __init__.py
│   ├── main_app.py      # main Streamlit UI (refactor your current main file here)
│   ├── auth.py          # SQLite auth functions
│   ├── rag_components.py # For LangChain RAG logic
│   └── utils.py         # Utilities
├── documents/           # Store PDFs here
│   └── source_document.pdf
├── vector_store_data/   #For FAISS index etc.
├── Dockerfile        
├── docker-compose.yml 
├── requirements.txt
└── .env
```
