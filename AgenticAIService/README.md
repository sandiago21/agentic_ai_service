* Agentic AI Service
- FastAPI
- Pydantic
- LangGraph
- LLM
- Qwen2.5

* TODO:
- GitHub Actions (CI/CD)
- Docker


<!-- uvicorn main:api --reload     - Run Locally (Outside of Docker) -->


<!-- Right now Docker may be re-downloading and duplicating memory usage.

Run with volume:

docker run -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  fastapi-app -->