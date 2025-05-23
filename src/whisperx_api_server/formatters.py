from whisperx.utils import WriteSRT, WriteVTT, WriteAudacity
from fastapi.responses import JSONResponse, Response
from whisperx_api_server.config import MediaType

class ListWriter:
    """Helper class to store written lines in memory."""
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)

    def get_output(self):
        return ''.join(self.lines)

    def flush(self):
        pass

def update_options(kwargs, defaults):
    """
    Helper function to update default options with values from kwargs.
    
    :param kwargs: Keyword arguments from the function call.
    :param defaults: Dictionary of default values.
    :return: Updated options dictionary.
    """
    options = defaults.copy()
    options.update({key: kwargs.get(key, value) for key, value in defaults.items()})
    return options

def handle_whisperx_format(transcript, writer_class, options):
    """
    Helper function to handle "srt", "vtt" and "aud" formats using whisperx writers.
    
    :param transcript: The transcript dictionary.
    :param writer_class: The writer class (WriteSRT, WriteVTT or WriteAudacity).
    :param options: Options for the writer.
    :return: Formatted output as a string.
    """
    writer = writer_class(output_dir=None)
    output = ListWriter()

    transcript["segments"]["language"] = transcript["language"]
    
    writer.write_result(transcript["segments"], output, options)

    return output.get_output()

def format_verbose_json(transcript):

    segments = transcript["segments"]["segments"]
    word_segments = transcript["segments"]["word_segments"]
    
    transformed_segments = []
    for idx, segment in enumerate(segments, start=1):
        new_segment = {"id": idx}
        new_segment.update(segment)
        transformed_segments.append(new_segment)
    
    new_transcript = {
        "task": transcript["task"],
        "language": transcript["language"],
        "duration": transcript["duration"],
        "text": transcript["text"],
        "words": word_segments,
        "segments": transformed_segments
    }
    
    return new_transcript

def format_transcription(transcript, format, **kwargs) -> Response:
    """
    Format a transcript into a given format and return a FastAPI Response object.
    
    :param transcript: The transcript to format, a dictionary with a "segments" key that contains a list of segments with start and end times and text.
    :param format: The format to generate the transcript in. Supported formats are "json", "text", "srt", "vtt" and "aud".
    :param kwargs: Additional keyword arguments to pass to the formatter.
    :return: A FastAPI Response or JSONResponse object with the formatted transcript and appropriate media type.
    """
    # Default options, used for formats imported from whisperx.utils
    defaults = {
        "max_line_width": 1000,
        "max_line_count": None,
        "highlight_words": kwargs.get("highlight_words", False),
    }
    options = update_options(kwargs, defaults)

    if format == "json":
        response_data = {"text": transcript.get("text", "")}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    elif format == "verbose_json":
        return format_verbose_json(transcript)
    elif format == "vtt_json":
        transcript["vtt_text"] = handle_whisperx_format(transcript, WriteVTT, options)
        return JSONResponse(content=transcript, media_type=MediaType.APPLICATION_JSON)
    elif format == "text":
        return Response(content=transcript.get("text", ""), media_type=MediaType.TEXT_PLAIN)
    elif format == "srt":
        content = handle_whisperx_format(transcript, WriteSRT, options)
        return Response(content=content, media_type=MediaType.TEXT_PLAIN)
    elif format == "vtt":
        content = handle_whisperx_format(transcript, WriteVTT, options)
        return Response(content=content, media_type=MediaType.TEXT_VTT)
    elif format == "aud":
        content = handle_whisperx_format(transcript, WriteAudacity, options)
        return Response(content=content, media_type=MediaType.TEXT_PLAIN)
    else:
        raise ValueError(f"Unsupported format: {format}")
