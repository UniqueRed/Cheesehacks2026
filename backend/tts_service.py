"""
Gemini Text-to-Speech Service with Advanced Emotional Control

This service uses Google Cloud Text-to-Speech API with Gemini-TTS models
for emotion-based text-to-speech with natural language style prompts.
Uses Vertex AI service account credentials.

Usage:
    from tts_service import TTSService
    
    tts = TTSService()
    audio_data = await tts.synthesize_speech(
        text="Hello, how are you?",
        emotion="happiness"
    )
"""

import os
from typing import Optional, Dict
import logging
import asyncio
from google.cloud import texttospeech

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSService:
    """
    Service for emotion-based text-to-speech using Cloud TTS API with Gemini-TTS models.
    Supports natural language style prompts for advanced emotional control.
    Uses Vertex AI service account credentials.
    """
    
    # Emotion to tone prompt mapping
    EMOTION_STYLE_MAP: Dict[str, str] = {
        "happiness": "Say the following in a cheerful, upbeat, and energetic tone",
        "sadness": "Say the following in a soft, gentle, and melancholic tone",
        "anger": "Say the following in an intense, firm, and sharp tone",
        "fear": "Say the following in a cautious, hesitant, and softer voice",
        "surprise": "Say the following with elevated pitch and energetic excitement",
        "neutral": "Say the following in a natural, conversational tone"
    }
    
    # Gemini-TTS model options
    DEFAULT_MODEL = "gemini-2.5-flash-tts"
    
    # Available Gemini-TTS voices
    DEFAULT_VOICE = "Orus"
    AVAILABLE_VOICES = [
        "Kore", "Aoede", "Callirrhoe", "Puck", "Charon", "Fenrir", "Leda", "Orus", "Zephyr"
    ]
    
    def __init__(self, language_code: str = "en-US", model: str = None, voice: str = None):
        """
        Initialize the TTS service using Cloud TTS API with Gemini-TTS models.
        
        Args:
            language_code: Language code for TTS (default: "en-US")
            model: Gemini-TTS model name (default: gemini-2.5-flash-tts)
            voice: Voice name (default: Orus)
        
        Raises:
            ValueError: If GOOGLE_APPLICATION_CREDENTIALS is not set
        """
        # Check for credentials
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set. "
                "Please set it to the path of your service account JSON file."
            )
        
        # Initialize the TTS client (uses Vertex AI credentials)
        try:
            self.client = texttospeech.TextToSpeechClient()
            self.language_code = language_code
            self.model = model or self.DEFAULT_MODEL
            self.voice = voice or self.DEFAULT_VOICE
            logger.info(f"Gemini TTS Service initialized with model: {self.model}, voice: {self.voice}, language: {language_code}")
        except Exception as e:
            logger.error(f"Failed to initialize TTS client: {e}")
            raise
    
    def _get_emotion_prompt(self, emotion: str) -> str:
        """
        Get the tone prompt for a given emotion.
        
        Args:
            emotion: Emotion name (e.g., "happiness", "sadness")
        
        Returns:
            Tone prompt string for the emotion, or neutral if emotion not found
        """
        emotion_lower = emotion.lower().strip()
        return self.EMOTION_STYLE_MAP.get(emotion_lower, self.EMOTION_STYLE_MAP["neutral"])
    
    async def synthesize_speech(
        self,
        text: str,
        emotion: str = "neutral",
        custom_prompt: Optional[str] = None,
        voice: Optional[str] = None,
        audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.MP3
    ) -> bytes:
        """
        Synthesize speech from text with emotion-based styling using Gemini-TTS.
        
        Uses the proper Gemini-TTS API format with separate prompt and text fields
        for natural language style control.
        
        Args:
            text: The text to convert to speech
            emotion: Emotion to apply (happiness, sadness, anger, fear, surprise, neutral)
            custom_prompt: Optional custom tone prompt (overrides emotion mapping)
            voice: Optional voice name (overrides default)
            audio_encoding: Audio encoding format (default: LINEAR16)
        
        Returns:
            Audio data as bytes
        
        Raises:
            Exception: If synthesis fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Get tone prompt (use custom if provided, otherwise use emotion mapping)
        style_prompt = custom_prompt or self._get_emotion_prompt(emotion)
        
        # Use specified voice or default
        voice_name = voice or self.voice
        
        try:
            # Combine prompt and text into a single text field
            # Format: "{prompt}: {text}"
            combined_text = f"{style_prompt}: {text.strip()}"
            logger.info(f"[TTS] Combined text for synthesis: '{combined_text[:100]}...' (length: {len(combined_text)})")
            synthesis_input = texttospeech.SynthesisInput(text=combined_text)
            
            # Configure voice with Gemini-TTS model name
            logger.info(f"[TTS] Configuring voice: name={voice_name}, model={self.model}, language={self.language_code}")
            voice_config = texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=voice_name,  # Gemini-TTS voice name (e.g., "Kore", "Aoede")
                model_name=self.model  # Gemini-TTS model (e.g., "gemini-2.5-flash-tts")
            )
            
            # Configure audio settings
            logger.info(f"[TTS] Configuring audio: encoding={audio_encoding}")
            audio_config = texttospeech.AudioConfig(
                audio_encoding=audio_encoding,
            )
            
            # Perform the synthesis using Gemini-TTS
            # Run in executor since synthesize_speech is a blocking synchronous call
            logger.info(f"[TTS] Calling synthesize_speech API (running in executor)...")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice_config,
                    audio_config=audio_config
                )
            )
            
            audio_content = response.audio_content
            logger.info(f"[TTS] API call successful! Received audio content length: {len(audio_content)} bytes")
            logger.info(f"[TTS] Successfully synthesized speech using Gemini-TTS (model: {self.model}, voice: {voice_name}) for emotion: {emotion}, text length: {len(text)}")
            return audio_content
            
        except Exception as e:
            logger.error(f"[TTS] Failed to synthesize speech: {e}")
            import traceback
            logger.error(f"[TTS] Traceback: {traceback.format_exc()}")
            raise Exception(f"TTS synthesis failed: {str(e)}")
    
    def get_available_emotions(self) -> list:
        """
        Get list of available emotions.
        
        Returns:
            List of emotion names
        """
        return list(self.EMOTION_STYLE_MAP.keys())
    
    def get_available_voices(self) -> list:
        """
        Get list of available Gemini-TTS voices.
        
        Returns:
            List of voice names
        """
        return self.AVAILABLE_VOICES.copy()
    
    def is_emotion_valid(self, emotion: str) -> bool:
        """
        Check if an emotion is valid.
        
        Args:
            emotion: Emotion name to check
        
        Returns:
            True if emotion is valid, False otherwise
        """
        return emotion.lower().strip() in self.EMOTION_STYLE_MAP
