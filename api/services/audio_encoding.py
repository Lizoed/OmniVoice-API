import io
import soundfile as sf
import numpy as np

DEFAULT_SAMPLE_RATE = 24000

def get_content_type(response_format: str) -> str:
    """Map requested audio format to HTTP content type."""
    format_mapping = {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    return format_mapping.get(response_format.lower(), "audio/wav")

def encode_audio(audio_data: np.ndarray, sample_rate: int, format: str) -> bytes:
    """
    Encode numpy audio array into the requested format bytes.
    Currently supports wav, pcm. Other formats would require pydub or similar.
    """
    if format.lower() == "pcm":
        # Ensure it's 16-bit PCM
        if audio_data.dtype != np.int16:
            # Scale float to int16
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        return audio_data.tobytes()

    # For other formats (wav, flac, etc.) use soundfile
    out_io = io.BytesIO()
    sf.write(out_io, audio_data, sample_rate, format=format.upper() if format.lower() != 'mp3' else 'WAV')
    return out_io.getvalue()
