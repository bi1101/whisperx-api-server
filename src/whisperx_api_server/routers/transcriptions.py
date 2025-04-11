import logging
import uuid
from .models import handle_default_openai_model
from fastapi import (
    APIRouter,
    UploadFile,
    Form,
    HTTPException,
    Request,
    status
)
from fastapi.responses import Response, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Literal, Annotated, Optional
from pydantic import AfterValidator
import time
import whisperx_api_server.transcriber as transcriber
from whisperx_api_server.dependencies import ConfigDependency
from whisperx_api_server.formatters import format_transcription
from whisperx_api_server.config import (
    Language,
    ResponseFormat,
    MediaType,
)
from whisperx_api_server.models import (
    load_model_instance,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Annotated ModelName for validation and defaults
ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

async def get_timestamp_granularities(request: Request) -> list[Literal["segment", "word"]]:
    TIMESTAMP_GRANULARITIES_COMBINATIONS = [
        [],
        ["segment"],
        ["word"],
        ["word", "segment"],
        ["segment", "word"],
    ]
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return ["segment"]
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    )
    return timestamp_granularities

def apply_defaults(config, model, language=None, response_format=None):
    if model is None:
        model = config.whisper.model
    if language is None:
        language = config.default_language
    if response_format is None:
        response_format = config.default_response_format
    return model, language, response_format

"""
OpenAI-like endpoint to transcribe audio files using the Whisper ASR model.

Args:
    request (Request): The HTTP request object.
    file (UploadFile): The audio file to transcribe.
    model (ModelName): The model to use for the transcription.
    language (Language): The language to use for the transcription. Defaults to "en".
    prompt (str): The prompt to use for the transcription.
    response_format (ResponseFormat): The response format to use for the transcription. Defaults to "json".
    temperature (float): The temperature to use for the transcription. Defaults to 0.0.
    timestamp_granularities (list[Literal["segment", "word"]]): The timestamp granularities to use for the transcription. Defaults to ["segment"].
    stream (bool): Whether to enable streaming mode. Defaults to False.
    hotwords (str): The hotwords to use for the transcription.
    suppress_numerals (bool): Whether to suppress numerals in the transcription. Defaults to True.
    highlight_words (bool): Whether to highlight words in the transcription (Applies only to VTT and SRT). Defaults to False.
    align (bool): Whether to do transcription timings alignment. Defaults to True.
    diarize (bool): Whether to diarize the transcription. Defaults to False.

Returns:
    Transcription: The transcription of the audio file.
"""
@router.post(
    "/v1/audio/transcriptions",
    description="Transcribe audio files using the Whisper ASR model.",
    tags=["Transcription"],
)
async def transcribe_audio(
    config: ConfigDependency,
    request: Request,
    file: UploadFile,
    model: Annotated[ModelName, Form()] = None,
    language: Annotated[Language, Form()] = None,
    prompt: Annotated[str, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        list[Literal["segment", "word"]],
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str, Form()] = None,
    suppress_numerals: Annotated[bool, Form()] = True,
    highlight_words: Annotated[bool, Form()] = False,
    align: Annotated[bool, Form()] = True,
    diarize: Annotated[bool, Form()] = False,
) -> Response:
    model, language, response_format = apply_defaults(config, model, language, response_format)
    timestamp_granularities = await get_timestamp_granularities(request)
    request_id = request.state.request_id
    logger.debug(f"Request ID: {request_id} - Received transcription request")
    start_time = time.time()  # Start the timer
    logger.debug(f"Request ID: {request_id} - Received request to transcribe {file.filename} with parameters: \
        model: {model}, \
        language: {language}, \
        prompt: {prompt}, \
        response_format: {response_format}, \
        temperature: {temperature}, \
        timestamp_granularities: {timestamp_granularities}, \
        stream: {stream}, \
        hotwords: {hotwords}, \
        suppress_numerals: {suppress_numerals} \
        highlight_words: {highlight_words} \
        align: {align}, \
        diarize: {diarize}")
    
    if not align:
        if response_format in ('vtt', 'srt', 'aud', 'vtt_json'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Subtitles format ('vtt', 'srt', 'aud', 'vtt_json') requires alignment to be enabled."
            )
        
        if diarize:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Diarization requires alignment to be enabled."
            )

    # Determine if word timestamps are required
    word_timestamps = "word" in timestamp_granularities

    # Build ASR options
    asr_options = {
        "suppress_numerals": suppress_numerals,
        "temperatures": temperature,
        "word_timestamps": word_timestamps,
        "initial_prompt": prompt,
        "hotwords": hotwords,
    }

    model_load_time = time.time()
    # Get model instance (reuse if cached)
    model_instance = await load_model_instance(model)

    logger.info(f"Loaded model {model} in {time.time() - model_load_time:.2f} seconds")

    try:
        transcription = await transcriber.transcribe(
            audio_file=file,
            asr_options=asr_options,
            language=language,
            whispermodel=model_instance,
            align=align,
            diarize=diarize,
            request_id=request_id
        )
    except Exception as e:
        logger.exception(f"Request ID: {request_id} - Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An unexpected error occurred while processing the transcription request."
        ) from e

    total_time = time.time() - start_time
    logger.info(f"Request ID: {request_id} - Transcription process took {total_time:.2f} seconds")

    return format_transcription(transcription, response_format)

"""
OpenAI-like endpoint to translate audio files using the Whisper ASR model.

Args:
    request (Request): The HTTP request object.
    file (UploadFile): The audio file to translate.
    model (ModelName): The model to use for the translation.
    prompt (str): The prompt to use for the translation.
    response_format (ResponseFormat): The response format to use for the translation. Defaults to "json".
    temperature (float): The temperature to use for the translation. Defaults to 0.0.

Returns:
    Translation: The translation of the audio file.
"""
@router.post(
    "/v1/audio/translations",
    description="Translate audio files using the Whisper ASR model",
    tags=["Translation"],
)
async def translate_audio(
    config: ConfigDependency,
    request: Request,
    file: UploadFile,
    model: Annotated[ModelName, Form()] = None,
    prompt: Annotated[str, Form()] = "",
    response_format: Annotated[ResponseFormat, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
) -> Response:
    model, _, response_format = apply_defaults(config, model, language=None, response_format=response_format)
    request_id = request.state.request_id
    logger.debug(f"Request ID: {request_id} - Received translation request")
    start_time = time.time()  # Start the timer
    logger.debug(f"Request ID: {request_id} - Received request to translate {file.filename} with parameters: \
        model: {model}, \
        prompt: {prompt}, \
        response_format: {response_format}, \
        temperature: {temperature}")
    
    # Build ASR options
    asr_options = {
        "initial_prompt": prompt,
        "temperatures": temperature,
    }

    model_load_time = time.time()
    # Get model instance (reuse if cached)
    model_instance = await load_model_instance(model)

    logger.info(f"Loaded model {model} in {time.time() - model_load_time:.2f} seconds")

    try:
        translation = await transcriber.transcribe(
            audio_file=file,
            asr_options=asr_options,
            whispermodel=model_instance,
            request_id=request_id,
            task="translate"
        )
    except Exception as e:
        logger.exception(f"Request ID: {request_id} - Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An unexpected error occurred while processing the translation request."
        ) from e

    total_time = time.time() - start_time
    logger.info(f"Request ID: {request_id} - Translation process took {total_time:.2f} seconds")

    return format_transcription(translation, response_format)

"""
Endpoint to align transcript with audio

Args:
    request (Request): The HTTP request object.
    file (UploadFile): The audio file to align with the transcript.
    transcript (str): The text transcript to align with the audio.
    language_code (Language): The language of the transcript. Defaults to "en".
    align_model (str): Name of phoneme-level ASR model to do alignment. Default is None.
    interpolate_method (str): Method to assign timestamps to non-aligned words. Defaults to "nearest".
    return_char_alignments (bool): Whether to return character-level alignments in the output. Defaults to False.
    response_format (ResponseFormat): The response format to use. Defaults to "json".

Returns:
    Response: The aligned transcript with timing information.
"""
@router.post(
    "/v1/audio/align",
    description="Align transcript with audio using WhisperX alignment models",
    tags=["Alignment"],
)
async def align_audio(
    config: ConfigDependency,
    request: Request,
    file: UploadFile,
    transcript: Annotated[str, Form()],
    language_code: Annotated[Language, Form()] = "en",
    align_model: Annotated[str, Form()] = None,
    interpolate_method: Annotated[str, Form()] = "nearest",
    return_char_alignments: Annotated[bool, Form()] = False,
    response_format: Annotated[ResponseFormat, Form()] = None,
) -> Response:
    _, _, response_format = apply_defaults(config, None, None, response_format)
    request_id = request.state.request_id
    logger.debug(f"Request ID: {request_id} - Received alignment request")
    
    start_time = time.time()
    logger.debug(f"Request ID: {request_id} - Aligning {file.filename} with parameters: \
        language_code: {language_code}, \
        align_model: {align_model}, \
        interpolate_method: {interpolate_method}, \
        return_char_alignments: {return_char_alignments}")
    
    try:
        device = config.whisper.inference_device.value
        result = await transcriber.align_whisper_output(
            transcript=transcript,
            audio_file=file,
            language_code=language_code,
            request_id=request_id,
            device=device,
            align_model=align_model,
            interpolate_method=interpolate_method,
            return_char_alignments=return_char_alignments
        )
    except Exception as e:
        logger.exception(f"Request ID: {request_id} - Alignment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during alignment: {str(e)}"
        ) from e
    
    total_time = time.time() - start_time
    logger.info(f"Request ID: {request_id} - Alignment process took {total_time:.2f} seconds")
    
    return format_transcription(result, response_format)

"""
Endpoint for speaker diarization of audio

Args:
    request (Request): The HTTP request object.
    file (UploadFile): The audio file to perform speaker diarization on.
    min_speakers (int): Minimum number of speakers to detect. Default is None.
    max_speakers (int): Maximum number of speakers to detect. Default is None.
    response_format (ResponseFormat): The response format to use. Defaults to "json".

Returns:
    Response: The diarization result with speaker segments.
"""
@router.post(
    "/v1/audio/diarize",
    description="Perform speaker diarization on audio using WhisperX diarization models",
    tags=["Diarization"],
)
async def diarize_audio(
    config: ConfigDependency,
    request: Request,
    file: UploadFile,
    min_speakers: Annotated[Optional[int], Form()] = None,
    max_speakers: Annotated[Optional[int], Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = None,
) -> Response:
    _, _, response_format = apply_defaults(config, None, None, response_format)
    request_id = request.state.request_id
    logger.debug(f"Request ID: {request_id} - Received diarization request")
    
    start_time = time.time()
    logger.debug(f"Request ID: {request_id} - Diarizing {file.filename} with parameters: \
        min_speakers: {min_speakers}, \
        max_speakers: {max_speakers}")
    
    try:
        device = config.whisper.inference_device.value
        result = await transcriber.diarize(
            audio_file=file,
            request_id=request_id,
            device=device,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    except Exception as e:
        logger.exception(f"Request ID: {request_id} - Diarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during diarization: {str(e)}"
        ) from e
    
    total_time = time.time() - start_time
    logger.info(f"Request ID: {request_id} - Diarization process took {total_time:.2f} seconds")
    
    # For most formats, we fall back to JSON since other formats like SRT don't make sense for diarization only
    return JSONResponse(
        content=result,
        media_type=MediaType.APPLICATION_JSON
    )