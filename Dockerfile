FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV DEBUG=False
ENV PORT=5000

# CMD ["python3", "main.py"]
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
