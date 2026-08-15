from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from app.services.document_service import DocumentProcessingService

router = APIRouter()

processor = DocumentProcessingService()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    from fastapi import HTTPException
    import traceback

    try:
        Path("documents").mkdir(exist_ok=True)
        file_path = f"documents/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return processor.process_document(file_path)
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)