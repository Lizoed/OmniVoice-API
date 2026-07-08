import aiosqlite
import io
import torch
from typing import Optional, List, Dict, Any
from fastapi import HTTPException
import logging
from ..config import settings

logger = logging.getLogger(__name__)

# In-memory cache for fast synthesis
_voice_cache: Dict[str, Any] = {}

async def init_db():
    """Initializes the SQLite database and table for voices asynchronously."""
    try:
        async with aiosqlite.connect(settings.DB_PATH) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS voices (
                    voice_id TEXT PRIMARY KEY,
                    name TEXT,
                    language TEXT,
                    description TEXT,
                    prompt_blob BLOB
                )
            """)
            await conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize voice DB: {e}")

def _serialize_prompt(prompt: Any) -> bytes:
    """Serializes the VoiceClonePrompt object to bytes using PyTorch."""
    buffer = io.BytesIO()
    torch.save(prompt, buffer)
    return buffer.getvalue()

def _deserialize_prompt(blob: bytes) -> Any:
    """Deserializes bytes back into a VoiceClonePrompt object."""
    buffer = io.BytesIO(blob)
    return torch.load(buffer, weights_only=False)

async def save_voice(
    voice_id: str, 
    name: str, 
    language: str, 
    prompt_obj: Any, 
    description: Optional[str] = None
):
    """
    Serializes the VoiceClonePrompt and saves it to SQLite database alongside metadata.
    Also updates the in-memory cache.
    """
    try:
        blob = _serialize_prompt(prompt_obj)
        
        async with aiosqlite.connect(settings.DB_PATH) as conn:
            await conn.execute("""
                INSERT INTO voices (voice_id, name, language, description, prompt_blob)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(voice_id) DO UPDATE SET
                    name=excluded.name,
                    language=excluded.language,
                    description=excluded.description,
                    prompt_blob=excluded.prompt_blob
            """, (voice_id, name, language, description, blob))
            await conn.commit()
            
        # Update cache
        _voice_cache[voice_id] = prompt_obj
        logger.info(f"Successfully saved voice '{voice_id}' to DB and cache.")
    except Exception as e:
        logger.error(f"Error saving voice {voice_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save voice: {e}")

async def get_voice(voice_id: str) -> Optional[Any]:
    """
    Retrieves the VoiceClonePrompt. Checks the in-memory cache first,
    otherwise loads and deserializes from the SQLite DB.
    """
    if voice_id in _voice_cache:
        return _voice_cache[voice_id]
        
    try:
        async with aiosqlite.connect(settings.DB_PATH) as conn:
            async with conn.execute("SELECT prompt_blob FROM voices WHERE voice_id = ?", (voice_id,)) as cursor:
                row = await cursor.fetchone()
                
                if row and row[0]:
                    blob = row[0]
                    prompt_obj = _deserialize_prompt(blob)
                    _voice_cache[voice_id] = prompt_obj
                    logger.info(f"Loaded voice '{voice_id}' from DB into cache.")
                    return prompt_obj
                    
        return None
    except Exception as e:
        logger.error(f"Error loading voice {voice_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load voice: {e}")

async def get_all_voices_metadata() -> List[Dict[str, Any]]:
    """
    Returns the list of voices (metadata only) without loading the heavy BLOBs.
    Matches the VoiceInfo structure.
    """
    voices = []
    try:
        async with aiosqlite.connect(settings.DB_PATH) as conn:
            async with conn.execute("SELECT voice_id, name, language, description FROM voices") as cursor:
                async for row in cursor:
                    voices.append({
                        "voice_id": row[0],
                        "name": row[1],
                        "language": row[2],
                        "description": row[3]
                    })
    except Exception as e:
        logger.error(f"Error fetching voice list: {e}")
        
    return voices

async def delete_voice(voice_id: str) -> bool:
    """
    Deletes the voice from the SQLite DB and in-memory cache.
    Returns True if a voice was deleted, False otherwise.
    """
    deleted = False
    try:
        async with aiosqlite.connect(settings.DB_PATH) as conn:
            async with conn.execute("DELETE FROM voices WHERE voice_id = ?", (voice_id,)) as cursor:
                if cursor.rowcount > 0:
                    deleted = True
            await conn.commit()
            
        if voice_id in _voice_cache:
            del _voice_cache[voice_id]
            deleted = True
            
        if deleted:
            logger.info(f"Successfully deleted voice '{voice_id}'.")
        else:
            logger.warning(f"Voice '{voice_id}' not found for deletion.")
            
        return deleted
    except Exception as e:
        logger.error(f"Error deleting voice {voice_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {e}")
