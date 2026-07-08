from typing import Literal, Optional
from pydantic import BaseModel, Field

class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    object: str = "model"
    created: int
    owned_by: str

class VoiceInfo(BaseModel):
    """Information about an available voice."""
    voice_id: str
    name: str
    language: str
    description: Optional[str] = None

class NormalizationOptions(BaseModel):
    """Options for the text normalization system."""
    normalize: bool = Field(default=True, description="Normalizes input text")
    unit_normalization: bool = Field(default=True, description="Transforms units")
    url_normalization: bool = Field(default=True, description="Changes URLs")
    email_normalization: bool = Field(default=True, description="Changes emails")
    optional_pluralization_normalization: bool = Field(default=True)
    phone_normalization: bool = Field(default=True)
    replace_remaining_symbols: bool = Field(default=True)

class OpenAISpeechRequest(BaseModel):
    """Request schema for OpenAI-compatible speech endpoint."""

    model: str = Field(
        default="omnivoice",
        description="The model to use for generation. Supported: omnivoice",
    )
    input: str = Field(
        ...,
        description="The text to generate audio for. Maximum length is 4096 characters.",
        max_length=4096,
    )
    voice_id: str = Field(
        default="default",
        description="The voice to use for generation.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, aac, flac, wav, pcm.",
    )
    speed: float = Field(
        default=1.0,
        description="The speed of the generated audio.",
        ge=0.25,
        le=4.0,
    )
    stream: bool = Field(
        default=False,
        description="If True, return audio as a chunked stream. Only supports pcm and wav formats.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language of the text.",
    )
    instruct: Optional[str] = Field(
        default=None,
        description="Optional voice/style instructions.",
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=None,
        description="Optional text normalization options.",
    )



class CreateVoiceRequest(BaseModel):
    """Request schema for saving a new voice to DB."""
    
    name: str = Field(
        ...,
        description="Human-readable name of the voice.",
    )
    language: str = Field(
        default="Auto",
        description="Primary language of the voice.",
    )
    ref_audio: str = Field(
        ...,
        description="Base64-encoded reference audio file (WAV, MP3, etc.).",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript of the reference audio. Better for quality if provided.",
        max_length=4096,
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the voice characteristics.",
    )
