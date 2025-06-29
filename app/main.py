from fastapi import FastAPI, Request, File, UploadFile
from app.api.routes import router as api_router
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from langserve import add_routes
from app.rag_system import ingest

from typing import AsyncIterator
import os
import json

app = FastAPI(title="Love Paper API")

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}  # PDF, DOCX
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

os.makedirs(UPLOAD_DIR, exist_ok=True)

def cleanup_file(file_path: str):
    os.remove(file_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI workspace!"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")

    # Create a unique filename
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid4().hex}{ext}"
    file_location = os.path.join(UPLOAD_DIR, unique_filename)

    # Save file
    with open(file_location, "wb") as f:
        f.write(contents)

    vectorized_data = ingest.do_ingest(unique_filename)
    
    return JSONResponse(content={
        "original_filename": file.filename,
        "saved_as": unique_filename,
        "path": file_location,
        "content_type": file.content_type,
        "size_bytes": len(contents),
        "vectorized_data": vectorized_data
    })

# from app.rag_system.rag_chain import rag_chain as paper_rag_chain
# add_routes(app, paper_rag_chain, path="/paper-rag")

from app.rag_system import dynamic_rag_chain

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question")
    index_name = body.get("index_name", "default-index")

    if not question or not index_name:
        return {"error": "Missing 'question' or 'index_name'"}

    chain = dynamic_rag_chain.build_rag_chain(index_name)

    # async def sse_stream():
    #     async for chunk in chain.astream(question):
    #         # SSE format: 'data: <your-data>\n\n'
    #         yield f"data: {chunk}\n\n"

    # return StreamingResponse(sse_stream(), media_type="text/event-stream")

    
    async def sse_json_stream():
        async for chunk in chain.astream(question):
            data = {"chunk": chunk}
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(sse_json_stream(), media_type="text/event-stream")

# Optional: add interactive playground at `/langserve`
add_routes(app, dynamic_rag_chain.build_rag_chain("pertanian-384"), path="/langserve")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
