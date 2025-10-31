import asyncio
import boto3
import json
import logging
import numpy as np
import time
import uuid
import base64
import io
from typing import Optional, Callable, Dict, Any
from botocore.exceptions import ClientError, BotoCoreError
import wave

logger = logging.getLogger(__name__)


class AWSSageMakerWhisperClient:
    """
    AWS SageMaker client for Whisper speech-to-text inference.
    
    Handles real-time audio processing by sending audio chunks to a SageMaker endpoint
    and managing the response callbacks in a way that's compatible with the existing
    RealtimeSTT interface.
    """
    
    def __init__(
        self,
        endpoint_name: str,
        profile_name: str = None,
        region_name: str = 'us-east-1',
        language: str = 'en',
        sample_rate: int = 16000,
        chunk_duration_ms: int = 1000,
        min_silence_duration: float = 0.5,
        max_silence_duration: float = 2.0,
    ):
        """
        Initialize AWS SageMaker Whisper client.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            profile_name: AWS profile name to use
            region_name: AWS region for the endpoint
            language: Language code for transcription
            sample_rate: Audio sample rate (16000 for Whisper)
            chunk_duration_ms: Duration of audio chunks to accumulate before sending
            min_silence_duration: Minimum silence before considering speech ended
            max_silence_duration: Maximum silence before forcing transcription
        """
        self.endpoint_name = endpoint_name
        self.profile_name = profile_name
        self.region_name = region_name
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        
        # Audio accumulation
        self.audio_buffer = []
        self.buffer_start_time = None
        self.last_audio_time = None
        self.silence_start_time = None
        self.is_recording = False
        
        # Callbacks (matching RealtimeSTT interface)
        self.realtime_transcription_callback: Optional[Callable[[str], None]] = None
        self.full_transcription_callback: Optional[Callable[[str], None]] = None
        self.potential_full_transcription_callback: Optional[Callable[[str], None]] = None
        self.potential_full_transcription_abort_callback: Optional[Callable[[], None]] = None
        self.potential_sentence_end: Optional[Callable[[str], None]] = None
        self.before_final_sentence: Optional[Callable[[Optional[np.ndarray], Optional[str]], bool]] = None
        self.on_recording_start: Optional[Callable[[], None]] = None
        self.on_turn_detection_start: Optional[Callable[[], None]] = None
        self.on_turn_detection_stop: Optional[Callable[[], None]] = None
        
        # Current transcription state
        self.current_partial_text = ""
        self.is_processing = False
        self.shutdown_requested = False
        
        # Initialize AWS client
        self._init_aws_client()
        
        # Start background processing
        self._processing_task = None
        
    def _init_aws_client(self):
        """Initialize the AWS SageMaker Runtime client."""
        try:
            if self.profile_name:
                session = boto3.Session(profile_name=self.profile_name)
                self.sagemaker_runtime = session.client(
                    'runtime.sagemaker',
                    region_name=self.region_name
                )
            else:
                self.sagemaker_runtime = boto3.client(
                    'runtime.sagemaker',
                    region_name=self.region_name
                )
            
            logger.info(f"🔊 AWS SageMaker client initialized with endpoint: {self.endpoint_name}")
            
        except Exception as e:
            logger.error(f"🔊💥 Failed to initialize AWS SageMaker client: {e}")
            raise
    
    def _encode_audio_for_whisper(self, audio_data: np.ndarray) -> str:
        """
        Encode audio data as base64 WAV for Whisper endpoint.
        
        Args:
            audio_data: Audio samples as int16 numpy array
            
        Returns:
            Base64 encoded WAV file content
        """
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_data = wav_buffer.getvalue()
        return base64.b64encode(wav_data).decode('utf-8')
    
    async def _invoke_whisper_endpoint(self, audio_base64: str) -> Optional[str]:
        """
        Invoke the SageMaker Whisper endpoint asynchronously.
        
        Args:
            audio_base64: Base64 encoded audio data
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            payload = {
                "audio": audio_base64,
                "language": self.language,
                "task": "transcribe"
            }
            
            # Use asyncio.to_thread for async AWS call
            response = await asyncio.to_thread(
                self.sagemaker_runtime.invoke_endpoint,
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            
            if 'text' in result:
                return result['text'].strip()
            else:
                logger.warning(f"🔊⚠️ Unexpected response format: {result}")
                return None
                
        except ClientError as e:
            logger.error(f"🔊💥 AWS SageMaker client error: {e}")
            return None
        except Exception as e:
            logger.error(f"🔊💥 Error invoking Whisper endpoint: {e}")
            return None
    
    def _detect_silence(self, audio_chunk: np.ndarray, threshold: int = 500) -> bool:
        """
        Simple silence detection based on RMS amplitude.
        
        Args:
            audio_chunk: Audio samples
            threshold: RMS threshold below which audio is considered silence
            
        Returns:
            True if silence detected
        """
        if len(audio_chunk) == 0:
            return True
        
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
        return rms < threshold
    
    def feed_audio(self, audio_bytes: bytes, metadata: Optional[Dict[str, Any]] = None):
        """
        Feed audio data to the processor (compatible with RealtimeSTT interface).
        
        Args:
            audio_bytes: Raw audio bytes (int16)
            metadata: Optional metadata about the audio
        """
        if self.shutdown_requested:
            return
            
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        if len(audio_data) == 0:
            return
        
        current_time = time.time()
        
        # Detect if this chunk contains speech
        is_silence = self._detect_silence(audio_data)
        
        if not is_silence:
            # Speech detected
            if not self.is_recording:
                self.is_recording = True
                self.buffer_start_time = current_time
                logger.info("🔊🎙️ Speech detected, starting recording")
                
                if self.on_recording_start:
                    self.on_recording_start()
                if self.on_turn_detection_stop:
                    self.on_turn_detection_stop()
            
            self.last_audio_time = current_time
            self.silence_start_time = None
            
        else:
            # Silence detected
            if self.is_recording and self.silence_start_time is None:
                self.silence_start_time = current_time
                if self.on_turn_detection_start:
                    self.on_turn_detection_start()
        
        # Add to buffer if recording
        if self.is_recording:
            self.audio_buffer.extend(audio_data)
            
            # Check if we should process the buffer
            should_process = False
            
            if self.silence_start_time:
                silence_duration = current_time - self.silence_start_time
                if silence_duration >= self.min_silence_duration:
                    should_process = True
            
            # Also process if buffer gets too long
            buffer_duration = len(self.audio_buffer) / self.sample_rate
            if buffer_duration >= self.max_silence_duration:
                should_process = True
            
            if should_process and not self.is_processing:
                # Process accumulated audio
                self._schedule_processing()
    
    def _schedule_processing(self):
        """Schedule processing of accumulated audio buffer."""
        if not self._processing_task or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_audio_buffer())
    
    async def _process_audio_buffer(self):
        """Process the accumulated audio buffer."""
        if self.is_processing or len(self.audio_buffer) == 0:
            return
        
        self.is_processing = True
        
        try:
            # Copy and clear buffer
            audio_to_process = np.array(self.audio_buffer, dtype=np.int16)
            self.audio_buffer = []
            
            logger.info(f"🔊⚙️ Processing {len(audio_to_process) / self.sample_rate:.2f}s of audio")
            
            # Call before_final_sentence callback if available
            if self.before_final_sentence:
                try:
                    self.before_final_sentence(audio_to_process, self.current_partial_text)
                except Exception as e:
                    logger.error(f"🔊💥 Error in before_final_sentence callback: {e}")
            
            # Encode audio for Whisper
            audio_base64 = self._encode_audio_for_whisper(audio_to_process)
            
            # Send to AWS endpoint
            transcription = await self._invoke_whisper_endpoint(audio_base64)
            
            if transcription:
                logger.info(f"🔊✅ Transcription: {transcription}")
                
                # Update current partial text
                self.current_partial_text = transcription
                
                # Call realtime callback for partial updates
                if self.realtime_transcription_callback:
                    self.realtime_transcription_callback(transcription)
                
                # Detect potential sentence endings
                if transcription.rstrip().endswith(('.', '!', '?')):
                    if self.potential_sentence_end:
                        self.potential_sentence_end(transcription)
                
                # Call final transcription callback
                if self.full_transcription_callback:
                    self.full_transcription_callback(transcription)
                
            else:
                logger.warning("🔊⚠️ No transcription received from AWS endpoint")
            
            # Reset recording state
            self.is_recording = False
            self.silence_start_time = None
            
        except Exception as e:
            logger.error(f"🔊💥 Error processing audio buffer: {e}")
        finally:
            self.is_processing = False
    
    def text(self, callback: Callable[[str], None]):
        """
        Set final transcription callback (compatible with RealtimeSTT interface).
        
        Args:
            callback: Function to call with final transcription text
        """
        self.full_transcription_callback = callback
    
    def shutdown(self):
        """Shutdown the AWS Whisper client."""
        logger.info("🔊🛑 Shutting down AWS Whisper client")
        self.shutdown_requested = True
        
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        
        # Clear buffers
        self.audio_buffer = []
        self.current_partial_text = ""
        self.is_recording = False
        self.is_processing = False


class AWSWhisperTranscriptionProcessor:
    """
    Adapter class that provides the same interface as TranscriptionProcessor
    but uses AWS SageMaker Whisper endpoint instead of local models.
    """
    
    def __init__(
        self,
        source_language: str = "en",
        realtime_transcription_callback: Optional[Callable[[str], None]] = None,
        full_transcription_callback: Optional[Callable[[str], None]] = None,
        potential_full_transcription_callback: Optional[Callable[[str], None]] = None,
        potential_full_transcription_abort_callback: Optional[Callable[[], None]] = None,
        potential_sentence_end: Optional[Callable[[str], None]] = None,
        before_final_sentence: Optional[Callable[[Optional[np.ndarray], Optional[str]], bool]] = None,
        silence_active_callback: Optional[Callable[[bool], None]] = None,
        on_recording_start_callback: Optional[Callable[[], None]] = None,
        is_orpheus: bool = False,
        local: bool = True,
        tts_allowed_event: Optional[asyncio.Event] = None,
        pipeline_latency: float = 0.5,
        endpoint_name: str = "jumpstart-dft-hf-asr-whisper-small-20251024-185556",
        profile_name: str = "tszkukle_bindle",
    ):
        """
        Initialize AWS Whisper transcription processor.
        
        Args:
            source_language: Language code for transcription
            realtime_transcription_callback: Callback for partial transcriptions
            full_transcription_callback: Callback for final transcriptions
            potential_full_transcription_callback: Callback for potential full transcriptions
            potential_full_transcription_abort_callback: Callback for aborted transcriptions
            potential_sentence_end: Callback for potential sentence endings
            before_final_sentence: Callback before final sentence processing
            silence_active_callback: Callback for silence state changes
            on_recording_start_callback: Callback when recording starts
            is_orpheus: Compatibility flag (unused for AWS)
            local: Compatibility flag (unused for AWS)
            tts_allowed_event: TTS synchronization event (unused for AWS)
            pipeline_latency: Pipeline latency estimate
            endpoint_name: SageMaker endpoint name
            profile_name: AWS profile name
        """
        self.source_language = source_language
        self.realtime_transcription_callback = realtime_transcription_callback
        self.full_transcription_callback = full_transcription_callback
        self.potential_full_transcription_callback = potential_full_transcription_callback
        self.potential_full_transcription_abort_callback = potential_full_transcription_abort_callback
        self.potential_sentence_end = potential_sentence_end
        self.before_final_sentence = before_final_sentence
        self.silence_active_callback = silence_active_callback
        self.on_recording_start_callback = on_recording_start_callback
        self.pipeline_latency = pipeline_latency
        
        # Compatibility attributes
        self.on_tts_allowed_to_synthesize: Optional[Callable] = None
        self.realtime_text: Optional[str] = None
        self.final_transcription: Optional[str] = None
        self.shutdown_performed: bool = False
        
        # Initialize AWS client
        self.aws_client = AWSSageMakerWhisperClient(
            endpoint_name=endpoint_name,
            profile_name=profile_name,
            language=source_language,
        )
        
        # Set up callbacks
        self._setup_callbacks()
        
        logger.info("🔊🚀 AWS Whisper Transcription Processor initialized")
    
    def _setup_callbacks(self):
        """Set up callbacks between AWS client and external interface."""
        
        def on_realtime_transcription(text: str):
            self.realtime_text = text  # Update compatibility attribute
            if self.realtime_transcription_callback:
                self.realtime_transcription_callback(text)
        
        def on_final_transcription(text: str):
            self.final_transcription = text  # Update compatibility attribute
            if self.full_transcription_callback:
                self.full_transcription_callback(text)
        
        def on_recording_start():
            if self.on_recording_start_callback:
                self.on_recording_start_callback()
        
        def on_silence_start():
            if self.silence_active_callback:
                self.silence_active_callback(True)
        
        def on_silence_stop():
            if self.silence_active_callback:
                self.silence_active_callback(False)
        
        # Connect callbacks
        self.aws_client.realtime_transcription_callback = on_realtime_transcription
        self.aws_client.full_transcription_callback = on_final_transcription
        self.aws_client.on_recording_start = on_recording_start
        self.aws_client.on_turn_detection_start = on_silence_start
        self.aws_client.on_turn_detection_stop = on_silence_stop
        self.aws_client.potential_sentence_end = self.potential_sentence_end
        self.aws_client.before_final_sentence = self.before_final_sentence
    
    def feed_audio(self, chunk: bytes, audio_meta_data: Optional[Dict[str, Any]] = None):
        """Feed audio to AWS client (compatible with RealtimeSTT interface)."""
        if not self.shutdown_performed:
            self.aws_client.feed_audio(chunk, audio_meta_data)
    
    def transcribe_loop(self):
        """Compatibility method - AWS processing is handled asynchronously."""
        # This method is called by the background task but AWS processing
        # is handled asynchronously through callbacks, so we just need to
        # keep this method alive
        import time
        while not self.shutdown_performed:
            time.sleep(0.1)  # Small sleep to prevent busy waiting
    
    def text(self, callback: Callable[[str], None]):
        """Set final transcription callback (compatible with RealtimeSTT interface)."""
        self.aws_client.text(callback)
    
    def abort_generation(self):
        """Abort current generation (compatibility method)."""
        logger.info("🔊🛑 AWS Whisper abort generation requested")
        # For AWS, we can't abort in-flight requests, but we can clear buffers
        self.aws_client.audio_buffer = []
        self.aws_client.is_processing = False
    
    def perform_final(self, audio_bytes: Optional[bytes] = None):
        """Perform final transcription (compatibility method)."""
        logger.info("🔊🏁 AWS Whisper perform final requested")
        if self.realtime_text and self.full_transcription_callback:
            self.full_transcription_callback(self.realtime_text)
    
    def shutdown(self):
        """Shutdown the AWS transcription processor."""
        if not self.shutdown_performed:
            logger.info("🔊🛑 Shutting down AWS Whisper Transcription Processor")
            self.shutdown_performed = True
            self.aws_client.shutdown()
