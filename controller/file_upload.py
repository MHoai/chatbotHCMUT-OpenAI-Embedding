from typing import Optional, List

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from haystack import Pipeline
from haystack.nodes import BaseConverter, PreProcessor

from utils import get_app, get_pipelines
from rest_api.config import FILE_UPLOAD_PATH
from rest_api.controller.utils import as_form


router = APIRouter()
app: FastAPI = get_app()
indexing_pipeline: Pipeline = get_pipelines().get("indexing_pipeline", None)


@as_form
class FileConverterParams(BaseModel):
    remove_numeric_tables: Optional[bool] = None
    valid_languages: Optional[List[str]] = None


@as_form
class PreprocessorParams(BaseModel):
    clean_whitespace: Optional[bool] = None
    clean_empty_lines: Optional[bool] = None
    clean_header_footer: Optional[bool] = None
    split_by: Optional[str] = None
    split_length: Optional[int] = None
    split_overlap: Optional[int] = None
    split_respect_sentence_boundary: Optional[bool] = None


class Response(BaseModel):
    file_id: str


@router.post("/file-upload")
def upload_file(
    files: List[UploadFile] = File(...),
    # JSON serialized string
    meta: Optional[str] = Form("null"),  # type: ignore
    additional_params: Optional[str] = Form("null"),  # type: ignore
    fileconverter_params: FileConverterParams = Depends(FileConverterParams.as_form),  # type: ignore
    preprocessor_params: PreprocessorParams = Depends(PreprocessorParams.as_form),  # type: ignore
    keep_files: Optional[bool] = False,
):
    """
    You can use this endpoint to upload a file for indexing
    (see https://docs.haystack.deepset.ai/docs/rest_api#indexing-documents-in-the-haystack-rest-api-documentstore).

    Note: files are removed immediately after being indexed. If you want to keep them, pass the
    `keep_files=true` parameter in the request payload.
    """
    if not indexing_pipeline:
        raise HTTPException(
            status_code=501, detail="Indexing Pipeline is not configured."
        )

    file_paths: list = []
    file_metas: list = []

    meta_form = json.loads(meta) or {}  # type: ignore
    if not isinstance(meta_form, dict):
        raise HTTPException(
            status_code=500,
            detail=f"The meta field must be a dict or None, not {type(meta_form)}",
        )

    for file in files:
        try:
            #file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
            file_path = Path(FILE_UPLOAD_PATH) / f"{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_paths.append(file_path)
            meta_form["name"] = file.filename
            file_metas.append(meta_form)
        finally:
            file.file.close()

    params = json.loads(additional_params) or {}  # type: ignore

    # Find nodes names
    converters = indexing_pipeline.get_nodes_by_class(BaseConverter)
    preprocessors = indexing_pipeline.get_nodes_by_class(PreProcessor)

    for converter in converters:
        params[converter.name] = fileconverter_params.dict()
    for preprocessor in preprocessors:
        params[preprocessor.name] = preprocessor_params.dict()

    indexing_pipeline.run(file_paths=file_paths, meta=file_metas, params=params)

    # Clean up indexed files
    if not keep_files:
        for p in file_paths:
            p.unlink()
