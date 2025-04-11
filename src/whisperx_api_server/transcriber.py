import os
from whisperx import transcribe as whisperx_transcribe
from whisperx import audio as whisperx_audio
from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize
from whisperx import types as whisperx_types
from whisperx import load_audio, align
from fastapi import UploadFile
import logging
import time
import tempfile

from whisperx_api_server.config import (
    Language,
)
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.models import (
    CustomWhisperModel,
    load_align_model_cached,
    load_diarize_model_cached,
)

logger = logging.getLogger(__name__)

config = get_config()

async def transcribe(
    audio_file: UploadFile,
    batch_size: int = config.batch_size,
    asr_options: dict = {},
    language: Language = config.default_language,
    whispermodel: CustomWhisperModel = config.whisper.model,
    align: bool = False,
    diarize: bool = False,
    request_id: str = "",
    task: str = "transcribe",
) -> whisperx_types.TranscriptionResult:
    start_time = time.time()  # Start timing
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as temp_file:
        temp_file.write(audio_file.file.read())
        file_path = temp_file.name

    logger.info(f"Request ID: {request_id} - Saving uploaded file took {time.time() - start_time:.2f} seconds")

    try:
        logger.info(f"Request ID: {request_id} - Transcribing {audio_file.filename} with model: {whispermodel.model_size_or_path} and options: {asr_options}")
        model_loading_start = time.time()
        model = whisperx_transcribe.load_model(
            whisper_arch=whispermodel.model_size_or_path,
            device=whispermodel.device,
            compute_type=whispermodel.compute_type,
            language=language,
            asr_options=asr_options,
            vad_model=config.whisper.vad_model,
            vad_method=config.whisper.vad_method,
            vad_options=config.whisper.vad_options,
            model=whispermodel,
            task=task,
        )
        logger.info(f"Request ID: {request_id} - Loading model took {time.time() - model_loading_start:.2f} seconds")

        audio_loading_start = time.time()
        audio = whisperx_audio.load_audio(file_path)
        logger.info(f"Request ID: {request_id} - Loading audio took {time.time() - audio_loading_start:.2f} seconds")

        transcription_start = time.time()
        result = model.transcribe(
            audio=audio,
            batch_size=batch_size,
            num_workers=config.whisper.num_workers,
            language=language,
            task=task,
        )
        logger.info(f"Request ID: {request_id} - Transcription took {time.time() - transcription_start:.2f} seconds")

        if align or diarize:
            alignment_model_start = time.time()
            logger.info(f"Request ID: {request_id} - Loading alignment model")
            model_a, metadata = await load_align_model_cached(
                language_code=result["language"],
            )
            logger.info(f"Request ID: {request_id} - Alignment model loaded")
            logger.info(f"Request ID: {request_id} - Loading alignment model took {time.time() - alignment_model_start:.2f} seconds")

            alignment_start = time.time()
            result["segments"] = whisperx_alignment.align(
                transcript=result["segments"],
                model=model_a,
                align_model_metadata=metadata,
                audio=audio,
                device=whispermodel.device,
                return_char_alignments=False
            )
            logger.info(f"Request ID: {request_id} - Alignment took {time.time() - alignment_start:.2f} seconds")

        if diarize:
            diarization_model_start = time.time()

            logger.info(f"Request ID: {request_id} - Loading diarization model")

            diarize_model = await load_diarize_model_cached(model_name="tensorlake/speaker-diarization-3.1")

            logger.info(f"Request ID: {request_id} - Diarization model loaded. Loading took {time.time() - diarization_model_start:.2f} seconds. Starting diarization")

            diarize_start = time.time()

            diarize_segments = diarize_model(audio)

            result["segments"] = whisperx_diarize.assign_word_speakers(diarize_segments, result["segments"])

            logger.info(f"Request ID: {request_id} - Diarization took {time.time() - diarize_start:.2f} seconds")

        if align or diarize:
            result["text"] = '\n'.join([segment["text"].strip() for segment in result["segments"]["segments"] if segment["text"].strip()])
        else:
            result["text"] = '\n'.join([segment["text"].strip() for segment in result["segments"] if segment["text"].strip()])

        logger.info(f"Request ID: {request_id} - Transcription completed for {audio_file.filename}")
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Transcription failed for {audio_file.filename} with error: {e}")
        raise
    finally:
        try:
            os.remove(file_path)
        except Exception:
            logger.error(f"Request ID: {request_id} - Could not remove temporary file: {file_path}")

    return result

async def align_whisper_output(
    transcript: str,
    audio_file: UploadFile,
    language_code: str,
    request_id: str = "",
    device: str = config.whisper.inference_device.value,
    align_model: str = None,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
) -> dict:
    """
    Align the transcript to the original audio.

    Args:
       transcript: The text transcript.
       audio_file: The original audio file.
       language_code: The language code.
       request_id: Request identifier for logging.
       device: Device to use for processing.
       align_model: Name of phoneme-level ASR model to do alignment.
       interpolate_method: For word .srt, method to assign timestamps to non-aligned words.
       return_char_alignments: Whether to return character-level alignments in the output json file.

    Returns:
       The aligned transcript result.
    """
    from whisperx_api_server.models import _determine_inference_device
    
    start_time = time.time()
    logger.debug(f"Request ID: {request_id} - Starting alignment process")

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as temp_file:
        temp_file.write(audio_file.file.read())
        file_path = temp_file.name

    try:
        # Load audio
        audio_loading_start = time.time()
        audio = load_audio(file_path)
        logger.info(f"Request ID: {request_id} - Loading audio took {time.time() - audio_loading_start:.2f} seconds")
        
        # Create segments with required fields - the align function expects segments with start/end times
        # We'll use placeholder values that span the entire audio duration
        audio_duration = audio.shape[0] / 16000  # Assuming 16kHz audio
        segments = [{"text": transcript, "start": 0.0, "end": audio_duration}]
        
        # Use the utility function to handle device configuration
        inference_device = _determine_inference_device()
        
        # Get alignment model
        alignment_model_start = time.time()
        model_a, metadata = await load_align_model_cached(
            language_code=language_code,
            model_name=align_model
        )
        logger.info(f"Request ID: {request_id} - Loading alignment model took {time.time() - alignment_model_start:.2f} seconds")
        
        # Perform alignment
        alignment_start = time.time()
        aligned_segments = align(
            segments,
            model_a,
            metadata,
            audio,
            inference_device,
            interpolate_method=interpolate_method,
            return_char_alignments=return_char_alignments
        )
        logger.info(f"Request ID: {request_id} - Alignment took {time.time() - alignment_start:.2f} seconds")
        
        # Prepare result
        result = {
            "task": "align",
            "language": language_code,
            "duration": audio_duration,
            "text": transcript,
            "segments": aligned_segments
        }
        
        logger.info(f"Request ID: {request_id} - Alignment completed for {audio_file.filename}")
        return result
    
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Alignment failed for {audio_file.filename} with error: {e}")
        raise
    finally:
        # Clean up temporary file
        try:
            os.remove(file_path)
        except Exception:
            logger.error(f"Request ID: {request_id} - Could not remove temporary file: {file_path}")

async def diarize(
    audio_file: UploadFile, 
    request_id: str = "",
    device: str = config.whisper.inference_device.value,
    min_speakers: int = None, 
    max_speakers: int = None
) -> dict:
    """
    Diarize an audio file using the PyAnnotate model.

    Args:
       audio_file: The audio to diarize.
       request_id: Request identifier for logging.
       device: Device to use for processing.
       min_speakers: Minimum number of speakers to detect.
       max_speakers: Maximum number of speakers to detect.

    Returns:
       Diarizartion: The diarization result.
    """
    from whisperx_api_server.models import _determine_inference_device
    import json
    
    start_time = time.time()
    logger.debug(f"Request ID: {request_id} - Starting diarization process")

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as temp_file:
        temp_file.write(audio_file.file.read())
        file_path = temp_file.name

    try:
        # Load audio
        audio_loading_start = time.time()
        audio = load_audio(file_path)
        logger.info(f"Request ID: {request_id} - Loading audio took {time.time() - audio_loading_start:.2f} seconds")
        
        # Use the utility function to handle device configuration
        inference_device = _determine_inference_device()
        
        # Get diarization model
        diarization_model_start = time.time()
        diarize_model = await load_diarize_model_cached(
            model_name="tensorlake/speaker-diarization-3.1"
        )
        logger.info(f"Request ID: {request_id} - Loading diarization model took {time.time() - diarization_model_start:.2f} seconds")
        
        # Perform diarization
        diarize_start = time.time()
        diarize_segments = diarize_model(
            audio=audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        logger.info(f"Request ID: {request_id} - Diarization took {time.time() - diarize_start:.2f} seconds")
        
        # Log the segments for debugging
        logger.info(f"Request ID: {request_id} - Diarization segments: {diarize_segments}")
        
        # Create a clean JSON-serializable structure
        serializable_segments = []
        
        # Based on the logs, diarize_segments is a pandas DataFrame with specific columns
        if hasattr(diarize_segments, 'to_dict'):
            # Convert DataFrame to records format
            records = diarize_segments.to_dict(orient="records")
            for record in records:
                # Create a clean dictionary with only what we need
                segment_dict = {
                    "start": float(record.get("start", 0)),
                    "end": float(record.get("end", 0)),
                    "speaker": str(record.get("speaker", "")),
                }
                
                # Add label if available
                if "label" in record:
                    segment_dict["label"] = str(record["label"])
                    
                # Add segment representation if available (like "[ 00:00:01.448 -->  00:00:04.249]")
                if "segment" in record or 0 in record:  # Index 0 might contain the segment representation
                    segment_repr = record.get("segment", record.get(0, ""))
                    if segment_repr:
                        segment_dict["segment_text"] = str(segment_repr)
                
                serializable_segments.append(segment_dict)
        else:
            # Fallback to our previous handling methods
            logger.warning(f"Request ID: {request_id} - Unexpected diarization format: {type(diarize_segments)}")
            
            # Try to handle it as an iterable of segments
            try:
                if hasattr(diarize_segments, "__iter__"):
                    for i, segment in enumerate(diarize_segments):
                        if hasattr(segment, "start") and hasattr(segment, "end"):
                            segment_dict = {
                                "start": float(segment.start),
                                "end": float(segment.end),
                                "speaker": getattr(segment, "speaker", f"SPEAKER_{i}")
                            }
                            
                            # Add label if available
                            if hasattr(segment, "label"):
                                segment_dict["label"] = str(segment.label)
                                
                            serializable_segments.append(segment_dict)
            except Exception as e:
                logger.error(f"Request ID: {request_id} - Failed to process diarization segments: {e}")
                # Provide a fallback with minimal information
                serializable_segments = [{"warning": "Could not process diarization data"}]
        
        # Test json serialization
        try:
            json.dumps(serializable_segments)
        except TypeError as e:
            logger.error(f"Request ID: {request_id} - Serialization test failed: {e}")
            # If serialization fails, provide a simple format that will definitely work
            serializable_segments = [{"warning": "Diarization data simplified due to serialization issues"}]
        
        # Prepare result
        result = {
            "task": "diarize",
            "duration": audio.shape[0] / 16000,  # Assuming 16kHz audio
            "diarization": serializable_segments
        }
        
        logger.info(f"Request ID: {request_id} - Diarization completed for {audio_file.filename}")
        return result
    
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Diarization failed for {audio_file.filename} with error: {e}")
        raise
    finally:
        # Clean up temporary file
        try:
            os.remove(file_path)
        except Exception:
            logger.error(f"Request ID: {request_id} - Could not remove temporary file: {file_path}")