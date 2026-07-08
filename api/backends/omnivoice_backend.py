import torch
import numpy as np
from typing import List, Union, Optional, Tuple, Generator, Any, Dict
import logging

from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt, OmniVoiceGenerationConfig
from omnivoice.utils.audio import cross_fade_chunks, fade_and_pad_audio, remove_silence

logger = logging.getLogger(__name__)

class OmniVoiceStreamModel:
    """
    A HuggingFace-style wrapper for OmniVoice models that provides:
      - from_pretrained() initialization
      - generation APIs mimicking Qwen3TTSModel:
          * generate_voice_clone() for batch generation
          * stream_generate_voice_clone() for chunk-by-chunk streaming generation
    """

    def __init__(self, model: OmniVoice):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "OmniVoiceStreamModel":
        model = OmniVoice.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model=model)

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, Tuple[torch.Tensor, int]],
        ref_text: Optional[str] = None,
        preprocess_prompt: bool = True,
    ) -> VoiceClonePrompt:
        """
        Create a reusable voice clone prompt from reference audio.
        """
        return self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt
        )

    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str], None] = None,
        ref_audio: Optional[Union[str, List[str], Tuple[torch.Tensor, int], List[Tuple[torch.Tensor, int]]]] = None,
        ref_text: Optional[Union[str, List[str]]] = None,
        voice_clone_prompt: Optional[Union[VoiceClonePrompt, List[VoiceClonePrompt]]] = None,
        instruct: Union[str, List[str], None] = None,
        **kwargs
    ) -> Tuple[List[np.ndarray], int]:
        """
        Voice clone speech using OmniVoice in batch mode.
        Mimics Qwen3TTSModel.generate_voice_clone.
        """
        if isinstance(text, str):
            text = [text]

        audios = self.model.generate(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_audio=ref_audio,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
            **kwargs
        )
        return audios, self.model.sampling_rate

    @torch.no_grad()
    def stream_generate_voice_clone(
        self,
        text: str,
        language: Optional[str] = None,
        ref_audio: Optional[Union[str, Tuple[torch.Tensor, int]]] = None,
        ref_text: Optional[str] = None,
        voice_clone_prompt: Optional[VoiceClonePrompt] = None,
        instruct: Optional[str] = None,
        audio_chunk_duration: float = 5.0, # smaller duration for better streaming latency
        **kwargs
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Stream voice clone speech generation chunk-by-chunk.
        Since OmniVoice generates iteratively using masked language modeling,
        true frame-by-frame streaming is not natively supported.
        Instead, this method uses chunked generation and yields audio 
        chunk-by-chunk as soon as each chunk finishes iterative decoding.
        
        Args:
            text: Text to synthesize (single string only).
            language: Language for synthesis.
            ref_audio: Reference audio for prompt building (required if voice_clone_prompt not provided).
            ref_text: Reference text.
            voice_clone_prompt: Pre-built VoiceClonePrompt.
            instruct: Style instruction for voice design mode.
            audio_chunk_duration: Controls the streaming chunk size.
            **kwargs: Generation parameters for OmniVoiceGenerationConfig.
            
        Yields:
            Tuple[np.ndarray, int]: (pcm_chunk as float32 array, sample_rate)
        """
        if isinstance(text, list):
            raise ValueError("stream_generate_voice_clone only supports single text, not batch")

        kwargs["audio_chunk_duration"] = audio_chunk_duration
        kwargs["audio_chunk_threshold"] = audio_chunk_duration
        
        gen_config = OmniVoiceGenerationConfig.from_dict(kwargs)
        
        task = self.model._preprocess_all(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_audio=ref_audio,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
            preprocess_prompt=gen_config.preprocess_prompt,
        )
        
        avg_tokens_per_char = task.target_lens[0] / len(task.texts[0])
        text_chunk_len = int(
            gen_config.audio_chunk_duration
            * self.model.audio_tokenizer.config.frame_rate
            / avg_tokens_per_char
        )
        from omnivoice.utils.text import chunk_text_punctuation
        chunks = chunk_text_punctuation(
            text=task.texts[0],
            chunk_len=text_chunk_len,
            min_chunk_len=3,
        )
        logger.debug(f"Streaming: item chunked into {len(chunks)} pieces")
        
        if len(chunks) == 0:
            return
            
        has_ref = task.ref_audio_tokens[0] is not None
        ref_audio_tokens = task.ref_audio_tokens[0]
        ref_text_item = task.ref_texts[0]
        ref_rms = task.ref_rms[0]
        
        def _run_single_chunk(c_text, c_ref_audio, c_ref_text):
            speed = task.speed[0] if task.speed else 1.0
            target_len = self.model._estimate_target_tokens(
                c_text,
                c_ref_text,
                c_ref_audio.size(-1) if c_ref_audio is not None else None,
                speed=speed,
            )
            sub_task = type(task)(
                batch_size=1,
                texts=[c_text],
                target_lens=[target_len],
                langs=[task.langs[0]],
                instructs=[task.instructs[0]],
                ref_texts=[c_ref_text],
                ref_audio_tokens=[c_ref_audio],
                ref_rms=[task.ref_rms[0]],
                speed=[speed],
            )
            return self.model._generate_iterative(sub_task, gen_config)[0]

        sr = self.model.sampling_rate
        
        def _post_process_chunk(chunk_audio, r_rms):
            # Similar to model._post_process_audio but applied per chunk
            if gen_config.postprocess_output:
                chunk_audio = remove_silence(
                    chunk_audio, sr, mid_sil=500, lead_sil=100, trail_sil=100
                )
            if r_rms is not None and r_rms < 0.1:
                chunk_audio = chunk_audio * r_rms / 0.1
            elif r_rms is None:
                peak = np.abs(chunk_audio).max()
                if peak > 1e-6:
                    chunk_audio = chunk_audio / peak * 0.5
                    
            chunk_audio = fade_and_pad_audio(
                chunk_audio,
                pad_duration=gen_config.pad_duration,
                fade_duration=gen_config.fade_duration,
                sample_rate=sr,
            )
            return chunk_audio.squeeze(0)

        # Generating and yielding chunks sequentially
        if has_ref:
            for ci, c_text in enumerate(chunks):
                gen_tokens = _run_single_chunk(c_text, ref_audio_tokens, ref_text_item)
                audio_waveform = self.model.audio_tokenizer.decode(
                    gen_tokens.to(self.model.audio_tokenizer.device).unsqueeze(0)
                ).audio_values[0].cpu().numpy()
                
                audio_waveform = _post_process_chunk(audio_waveform, ref_rms)
                yield audio_waveform, sr
                
        else:
            gen_tokens_0 = _run_single_chunk(chunks[0], None, None)
            audio_waveform = self.model.audio_tokenizer.decode(
                gen_tokens_0.to(self.model.audio_tokenizer.device).unsqueeze(0)
            ).audio_values[0].cpu().numpy()
            
            audio_waveform = _post_process_chunk(audio_waveform, ref_rms)
            yield audio_waveform, sr
            
            for ci in range(1, len(chunks)):
                gen_tokens = _run_single_chunk(chunks[ci], gen_tokens_0, chunks[0])
                audio_waveform = self.model.audio_tokenizer.decode(
                    gen_tokens.to(self.model.audio_tokenizer.device).unsqueeze(0)
                ).audio_values[0].cpu().numpy()
                
                audio_waveform = _post_process_chunk(audio_waveform, ref_rms)
                yield audio_waveform, sr
