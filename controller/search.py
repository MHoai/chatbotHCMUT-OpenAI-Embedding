from typing import Dict, Any

import logging
import time
import json
import random

from pydantic import BaseConfig
from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse
import haystack
from haystack import Pipeline

from utils import get_app, get_pipelines
from rest_api.config import LOG_LEVEL
from schema import QueryRequest, QueryResponse, ChatUIQueryRequest
from envs import DEFAULT_ANSWERS

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


BaseConfig.arbitrary_types_allowed = True


router = APIRouter()
app: FastAPI = get_app()
query_pipeline: Pipeline = get_pipelines().get("query_pipeline", None)
concurrency_limiter = get_pipelines().get("concurrency_limiter", None)


@router.get("/initialized")
def check_status():
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@router.get("/hs_version")
def haystack_version():
    """
    Get the running Haystack version.
    """
    return {"hs_version": haystack.__version__}


async def async_query(request: ChatUIQueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    with concurrency_limiter.run():
        result = await _process_request(query_pipeline, request)
        
    yield "data:" + json.dumps(
        {
            "index": 1,
            "token": {"id": 1, "text": result["generated_text"], "logprob": 0, "special": False},
            "generated_text": None,
            "details": None
        }, ensure_ascii=False
    ) + "\n\n"
            
    yield "data:" + json.dumps(
        {
            "index": 2,
            "token": {"id": 1, "text": "<eos>", "logprob": 0, "special": True},
            "generated_text": result["generated_text"],
            "details": None
        }, ensure_ascii=False, separators=(',', ':')
    ) + "\n\n"


@router.post(
    "/generate_stream"
)
async def stream_query(request: ChatUIQueryRequest):
    headers = {"X-Accel-Buffering": "no"}
    return StreamingResponse(
        async_query(request), headers=headers, media_type="text/event-stream"
    )


@router.post(
    "/generate", response_model=QueryResponse, response_model_exclude_none=True
)
@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    with concurrency_limiter.run():
        result = await _process_request(query_pipeline, request)
        return result


async def _process_request(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()

    params = request.parameters or {}
    result = pipeline.run(query=request.inputs, params=params, debug=request.debug)

    # Ensure answers and documents exist, even if they're empty lists
    if not "documents" in result:
        result["documents"] = []
    if not "answers" in result:
        result["answers"] = []
        chosen_ans = random.choice(DEFAULT_ANSWERS)
        result["generated_text"] = chosen_ans
    else:
        result["generated_text"] = result["answers"][0].answer.strip()

    logger.info(
        json.dumps(
            {
                "request": request,
                "response": result,
                "time": f"{(time.time() - start_time):.2f}",
            },
            default=str,
        )
    )
    return {"generated_text": result["generated_text"]}
