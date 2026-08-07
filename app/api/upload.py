from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from app.services.document_service import DocumentProcessingService

router = APIRouter()

processor = DocumentProcessingService()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    Path("documents").mkdir(exist_ok=True)

    file_path = f"documents/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return processor.process_document(file_path)