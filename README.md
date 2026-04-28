* Agentic AI Service
- FastAPI
- Pydantic
- LangGraph
- LLM
- Qwen2.5
- Docker

* TODO:
- Asyncio
- Makefile
- Linting
- DVC
- Evaluation


<!-- Build Docker

docker build -t fastapi-app .  
-->

<!-- Run Docker

docker run -p 8000:8000 fastapi-app 
-->

<!-- Run Docker so that HuggingFace with reuse downloaded weights and won't have to download every time

docker run -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  fastapi-app
-->


<!-- Run Agent Locally (Outside of Docker)

uvicorn main:api --reload
-->


<!-- Run with volume:

docker run -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  fastapi-app 
-->

<!-- Call Docker after running it with curl command 

curl -X POST "http://127.0.0.1:8000/ask_question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who won the 2025 world snooker championship?",
    "filename": ""
  }' 

curl "http://127.0.0.1:8000/queries"

curl "http://127.0.0.1:8000/queries?first_n=5"

with debugging:
curl -v -X POST "http://127.0.0.1:8000/ask_question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who won the 2025 world snooker championship?",
    "filename": ""
  }'
-->
