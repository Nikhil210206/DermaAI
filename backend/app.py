# Main server file
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "DermaAI"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
