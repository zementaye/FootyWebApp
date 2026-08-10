FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Single worker: the model is trained once in-process on startup and held in memory.
# Multiple workers would each train their own copy (wasteful but not incorrect) —
# fine to increase if your host gives you enough RAM/CPU.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
