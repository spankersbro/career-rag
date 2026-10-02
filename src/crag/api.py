from fastapi import FastAPI

app = FastAPI(title="Career RAG")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
