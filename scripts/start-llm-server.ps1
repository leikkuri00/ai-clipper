# Start the FastAPI LLM adapter
.\.venv\Scripts\Activate.ps1
py -m uvicorn src.llm_server:app --host 127.0.0.1 --port 5100 --log-level info
