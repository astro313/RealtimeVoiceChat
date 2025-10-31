# bedrock_llm.py
import boto3
import json
import logging
import os
import sys
import time
import uuid
from typing import Generator, List, Dict, Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)

try:
    import importlib.util
    dotenv_spec = importlib.util.find_spec("dotenv")
    if dotenv_spec:
        from dotenv import load_dotenv
        load_dotenv()
        logger.debug("🤖⚙️ Loaded environment variables from .env file.")
    else:
        logger.debug("🤖⚙️ python-dotenv not installed, skipping .env load.")
except ImportError:
    logger.debug("🤖💥 Error importing dotenv, skipping .env load.")

# AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE_NAME = os.getenv("AWS_PROFILE_NAME")


class BedrockLLM:
    """
    AWS Bedrock client for Claude Haiku model inference.
    
    Provides streaming text generation using AWS Bedrock's Claude Haiku model
    with a compatible interface to the existing LLM class.
    """
    
    def __init__(
        self,
        model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0",
        region_name: str = None,
        profile_name: str = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Initialize AWS Bedrock Claude Haiku client.
        
        Args:
            model_id: Bedrock model ID for Claude Haiku
            region_name: AWS region name
            profile_name: AWS profile name
            system_prompt: System prompt for conversations
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        """
        self.model_id = model_id
        self.region_name = region_name or AWS_REGION
        self.profile_name = profile_name or AWS_PROFILE_NAME
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Client initialization
        self.bedrock_runtime = None
        self._client_initialized = False
        self._client_init_lock = Lock()
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._requests_lock = Lock()
        
        # System prompt message
        self.system_prompt_message = None
        if self.system_prompt:
            self.system_prompt_message = {"role": "system", "content": self.system_prompt}
            logger.info(f"🤖💬 Bedrock system prompt set.")
        
        logger.info(f"🤖⚙️ Bedrock LLM initialized with model: {self.model_id}")
    
    def _init_bedrock_client(self):
        """Initialize the AWS Bedrock Runtime client."""
        try:
            if self.profile_name:
                session = boto3.Session(profile_name=self.profile_name)
                self.bedrock_runtime = session.client(
                    'bedrock-runtime',
                    region_name=self.region_name
                )
            else:
                self.bedrock_runtime = boto3.client(
                    'bedrock-runtime',
                    region_name=self.region_name
                )
            
            logger.info(f"🤖🔌 AWS Bedrock client initialized with model: {self.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"🤖💥 Failed to initialize AWS Bedrock client: {e}")
            return False
    
    def _lazy_initialize_clients(self) -> bool:
        """Initialize Bedrock client on first use (thread-safe)."""
        if self._client_initialized:
            return self.bedrock_runtime is not None
        
        with self._client_init_lock:
            if self._client_initialized:
                return self.bedrock_runtime is not None
            
            logger.debug(f"🤖🔄 Lazy initializing Bedrock client")
            init_ok = self._init_bedrock_client()
            self._client_initialized = True
            return init_ok
    
    def _register_request(self, request_id: str, request_type: str, stream_obj: Optional[Any]):
        """Register an active generation stream for cancellation tracking."""
        with self._requests_lock:
            if request_id in self._active_requests:
                logger.warning(f"🤖⚠️ Request ID {request_id} already registered. Overwriting.")
            self._active_requests[request_id] = {
                "type": request_type,
                "stream": stream_obj,
                "start_time": time.time()
            }
            logger.debug(f"🤖ℹ️ Registered active request: {request_id} (Type: {request_type})")
    
    def _cancel_single_request_unsafe(self, request_id: str) -> bool:
        """Cancel a single request (thread-unsafe)."""
        request_data = self._active_requests.pop(request_id, None)
        if not request_data:
            logger.debug(f"🤖🗑️ Request {request_id} already removed before cancellation attempt.")
            return False
        
        logger.info(f"🤖🗑️ Cancelled request {request_id}")
        return True
    
    def cancel_generation(self, request_id: Optional[str] = None) -> bool:
        """Cancel active generation streams."""
        cancelled_any = False
        with self._requests_lock:
            ids_to_cancel = []
            if request_id is None:
                if not self._active_requests:
                    logger.debug("🤖🗑️ Cancel all requested, but no active requests found.")
                    return False
                logger.info(f"🤖🗑️ Attempting to cancel ALL active generation requests ({len(self._active_requests)}).")
                ids_to_cancel = list(self._active_requests.keys())
            else:
                if request_id not in self._active_requests:
                    logger.warning(f"🤖🗑️ Cancel requested for ID '{request_id}', but it's not an active request.")
                    return False
                logger.info(f"🤖🗑️ Attempting to cancel generation request: {request_id}")
                ids_to_cancel.append(request_id)
            
            for req_id in ids_to_cancel:
                if self._cancel_single_request_unsafe(req_id):
                    cancelled_any = True
        return cancelled_any
    
    def _prepare_messages(self, text: str, history: Optional[List[Dict[str, str]]] = None, use_system_prompt: bool = True) -> List[Dict[str, str]]:
        """Prepare messages for Bedrock API call."""
        messages = []
        
        # Add conversation history
        if history:
            messages.extend(history)
        
        # Add current user message
        if len(messages) == 0 or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": text})
        
        return messages
    
    def _create_bedrock_payload(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Create payload for Bedrock API call."""
        # Extract system message if present
        system_content = None
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append(msg)
        
        # Use provided system prompt or fallback to instance system prompt
        if system_content is None and self.system_prompt:
            system_content = self.system_prompt
        
        payload = {
            "messages": filtered_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "anthropic_version": "bedrock-2023-05-31"
        }
        
        if system_content:
            payload["system"] = system_content
        
        return payload
    
    def generate(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        request_id: Optional[str] = None,
        **kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Generate text using AWS Bedrock Claude Haiku model.
        
        Args:
            text: The user's input prompt/text
            history: Optional list of previous messages
            use_system_prompt: Whether to use system prompt
            request_id: Optional unique ID for this request
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Yields:
            str: Generated text tokens
        """
        # Initialize client if needed
        if not self._lazy_initialize_clients():
            raise RuntimeError("Bedrock client failed to initialize")
        
        req_id = request_id if request_id else f"bedrock-{uuid.uuid4()}"
        logger.info(f"🤖💬 Starting Bedrock generation (Request ID: {req_id})")
        
        # Prepare messages
        messages = self._prepare_messages(text, history, use_system_prompt)
        logger.debug(f"🤖💬 [{req_id}] Prepared messages count: {len(messages)}")
        
        # Create payload
        payload = self._create_bedrock_payload(messages, **kwargs)
        logger.info(f"🤖💬 [{req_id}] Bedrock payload prepared")
        logger.debug(f"🤖💬 [{req_id}] Payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Make streaming request to Bedrock
            response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )
            
            # Register the request for cancellation tracking
            self._register_request(req_id, "bedrock", response)
            
            # Process streaming response
            yield from self._yield_bedrock_chunks(response['body'], req_id)
            
            logger.info(f"🤖✅ Finished Bedrock generation successfully (request_id: {req_id})")
            
        except Exception as e:
            logger.error(f"🤖💥 Error in Bedrock generation pipeline for {req_id}: {e}", exc_info=True)
            raise
        finally:
            # Clean up request tracking
            logger.debug(f"🤖ℹ️ [{req_id}] Entering finally block for Bedrock generate.")
            with self._requests_lock:
                if req_id in self._active_requests:
                    logger.debug(f"🤖🗑️ [{req_id}] Removing request from tracking in finally block.")
                    self._cancel_single_request_unsafe(req_id)
                else:
                    logger.debug(f"🤖🗑️ [{req_id}] Request already removed from tracking before finally block.")
    
    def _yield_bedrock_chunks(self, response_stream, request_id: str) -> Generator[str, None, None]:
        """
        Process Bedrock streaming response and yield content chunks.
        
        Args:
            response_stream: The streaming response from Bedrock
            request_id: The unique ID for this request
        
        Yields:
            str: Content chunks from the stream
        """
        token_count = 0
        try:
            for event in response_stream:
                # Check for cancellation before processing chunk
                with self._requests_lock:
                    if request_id not in self._active_requests:
                        logger.info(f"🤖🗑️ Bedrock stream {request_id} cancelled or finished externally during iteration.")
                        break
                
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    
                    if chunk_obj['type'] == 'content_block_delta':
                        delta = chunk_obj.get('delta')
                        if delta and delta.get('type') == 'text_delta':
                            content = delta.get('text', '')
                            if content:
                                token_count += 1
                                yield content
                    elif chunk_obj['type'] == 'message_stop':
                        # End of message
                        logger.debug(f"🤖✅ [{request_id}] Bedrock signalled message stop.")
                        break
                    elif chunk_obj['type'] == 'content_block_stop':
                        # End of content block
                        logger.debug(f"🤖✅ [{request_id}] Bedrock signalled content block stop.")
                        break
            
            logger.debug(f"🤖✅ [{request_id}] Finished yielding {token_count} Bedrock tokens.")
            
        except Exception as e:
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests
            
            if is_cancelled:
                logger.warning(f"🤖⚠️ Bedrock stream error likely due to cancellation for {request_id}: {e}")
            else:
                logger.error(f"🤖💥 Error during Bedrock streaming ({request_id}): {e}", exc_info=True)
                raise
    
    def prewarm(self, max_retries: int = 1) -> bool:
        """
        Prewarm the Bedrock connection with a simple test generation.
        
        Args:
            max_retries: Number of retry attempts
            
        Returns:
            bool: True if prewarm successful, False otherwise
        """
        prompt = "Respond with only the word 'OK'."
        logger.info(f"🤖🔥 Attempting prewarm for Bedrock model '{self.model_id}'...")
        
        if not self._lazy_initialize_clients():
            logger.error("🤖🔥💥 Prewarm failed: Could not initialize Bedrock client.")
            return False
        
        attempts = 0
        while attempts <= max_retries:
            prewarm_start_time = time.time()
            prewarm_request_id = f"prewarm-bedrock-{uuid.uuid4()}"
            generator = None
            full_response = ""
            token_count = 0
            first_token_time = None
            
            try:
                logger.info(f"🤖🔥 Prewarm Attempt {attempts + 1}/{max_retries+1} calling generate (ID: {prewarm_request_id})...")
                generator = self.generate(
                    text=prompt,
                    history=None,
                    use_system_prompt=True,
                    request_id=prewarm_request_id,
                    temperature=0.1
                )
                
                gen_start_time = time.time()
                for token in generator:
                    if first_token_time is None:
                        first_token_time = time.time()
                        logger.info(f"🤖🔥⏱️ Prewarm TTFT: {(first_token_time - gen_start_time):.4f}s")
                    full_response += token
                    token_count += 1
                
                gen_end_time = time.time()
                logger.info(f"🤖🔥ℹ️ Prewarm consumed {token_count} tokens in {(gen_end_time - gen_start_time):.4f}s. Full response: '{full_response}'")
                
                if token_count == 0 and not full_response:
                    logger.warning(f"🤖🔥⚠️ Prewarm yielded no response content, but generation finished.")
                
                prewarm_end_time = time.time()
                logger.info(f"🤖🔥✅ Prewarm successful. Total time: {(prewarm_end_time - prewarm_start_time):.4f}s.")
                return True
                
            except Exception as e:
                logger.error(f"🤖🔥💥 Prewarm attempt {attempts + 1}/{max_retries+1} error: {e}")
                if attempts < max_retries:
                    attempts += 1
                    wait_time = 2 * attempts
                    logger.info(f"🤖🔥🔄 Retrying prewarm in {wait_time}s...")
                    time.sleep(wait_time)
                    self._client_initialized = False
                    continue
                else:
                    logger.error(f"🤖🔥💥 Prewarm failed permanently after {attempts + 1} attempts.")
                    return False
            finally:
                if generator and hasattr(generator, 'close'):
                    try:
                        generator.close()
                    except Exception as close_err:
                        logger.warning(f"🤖🔥⚠️ [{prewarm_request_id}] Error closing generator in prewarm finally: {close_err}")
                generator = None
        
        logger.error(f"🤖🔥💥 Prewarm failed after exhausting retries.")
        return False
    
    def measure_inference_time(self, num_tokens: int = 10, **kwargs: Any) -> Optional[float]:
        """
        Measure inference time for a target number of tokens.
        
        Args:
            num_tokens: Target number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            float: Time in milliseconds or None if failed
        """
        if num_tokens <= 0:
            logger.warning("🤖⏱️ Cannot measure inference time for 0 or negative tokens.")
            return None
        
        if not self._lazy_initialize_clients():
            logger.error(f"🤖⏱️💥 Measurement failed: Could not initialize Bedrock client.")
            return None
        
        measurement_prompt = "Repeat the following sequence exactly: one two three four five six seven eight nine ten eleven twelve"
        req_id = f"measure-bedrock-{uuid.uuid4()}"
        logger.info(f"🤖⏱️ Measuring inference time for {num_tokens} tokens (Request ID: {req_id})")
        
        token_count = 0
        start_time = None
        end_time = None
        generator = None
        
        try:
            generator = self.generate(
                text=measurement_prompt,
                history=None,
                use_system_prompt=False,
                request_id=req_id,
                **kwargs
            )
            
            start_time = time.time()
            for token in generator:
                token_count += 1
                if token_count >= num_tokens:
                    end_time = time.time()
                    logger.debug(f"🤖⏱️ [{req_id}] Reached target {num_tokens} tokens.")
                    break
            
            if end_time is None:
                end_time = time.time()
                logger.debug(f"🤖⏱️ [{req_id}] Generation finished naturally after {token_count} tokens.")
            
        except Exception as e:
            logger.error(f"🤖⏱️💥 Error during inference time measurement ({req_id}): {e}")
            return None
        finally:
            if generator and hasattr(generator, 'close'):
                try:
                    generator.close()
                except Exception as close_err:
                    logger.warning(f"🤖⏱️⚠️ [{req_id}] Error closing generator in measure_inference_time finally: {close_err}")
            generator = None
        
        if start_time is None or end_time is None:
            logger.error(f"🤖⏱️💥 [{req_id}] Measurement failed: Start or end time not recorded.")
            return None
        
        if token_count == 0:
            logger.warning(f"🤖⏱️⚠️ [{req_id}] Measurement invalid: 0 tokens were generated.")
            return None
        
        duration_sec = end_time - start_time
        duration_ms = duration_sec * 1000
        
        logger.info(
            f"🤖⏱️✅ Measured ~{duration_ms:.2f} ms for {token_count} tokens "
            f"(target: {num_tokens}) for Bedrock model '{self.model_id}'. (Request ID: {req_id})"
        )
        
        return duration_ms


# Context manager for Bedrock generation
class BedrockGenerationContext:
    """Context manager for safely handling Bedrock generation streams."""
    
    def __init__(
        self,
        bedrock_llm: BedrockLLM,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        **kwargs: Any
    ):
        self.bedrock_llm = bedrock_llm
        self.prompt = prompt
        self.history = history
        self.use_system_prompt = use_system_prompt
        self.kwargs = kwargs
        self.generator: Optional[Generator[str, None, None]] = None
        self.request_id: str = f"ctx-bedrock-{uuid.uuid4()}"
        self._entered: bool = False
    
    def __enter__(self) -> Generator[str, None, None]:
        if self._entered:
            raise RuntimeError("BedrockGenerationContext cannot be re-entered")
        self._entered = True
        logger.debug(f"🤖▶️ [{self.request_id}] Entering BedrockGenerationContext.")
        
        try:
            self.generator = self.bedrock_llm.generate(
                self.prompt,
                self.history,
                self.use_system_prompt,
                request_id=self.request_id,
                **self.kwargs
            )
            return self.generator
        except Exception as e:
            logger.error(f"🤖💥 [{self.request_id}] Failed generator creation in context: {e}")
            self.bedrock_llm.cancel_generation(self.request_id)
            self._entered = False
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"🤖◀️ [{self.request_id}] Exiting BedrockGenerationContext (Exc: {exc_type}).")
        self.bedrock_llm.cancel_generation(self.request_id)
        
        if self.generator and hasattr(self.generator, 'close'):
            try:
                logger.debug(f"🤖🗑️ [{self.request_id}] Explicitly closing generator in context exit.")
                self.generator.close()
            except Exception as e:
                logger.warning(f"🤖⚠️ [{self.request_id}] Error closing generator in context exit: {e}")
        
        self.generator = None
        self._entered = False
        return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger(__name__)
    main_logger.info("🤖🚀 --- Running Bedrock LLM Example ---")
    
    try:
        bedrock_llm = BedrockLLM(
            model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
            system_prompt="You are a helpful assistant. Be concise and accurate."
        )
        
        # Prewarm
        main_logger.info("🤖🔥 --- Running Bedrock Prewarm ---")
        prewarm_success = bedrock_llm.prewarm(max_retries=1)
        
        if prewarm_success:
            main_logger.info("🤖✅ Bedrock Prewarm OK.")
            
            # Measure inference time
            main_logger.info("🤖⏱️ --- Running Bedrock Inference Time Measurement ---")
            inf_time = bedrock_llm.measure_inference_time(num_tokens=10, temperature=0.1)
            if inf_time is not None:
                main_logger.info(f"🤖⏱️ --- Measured Inference Time: {inf_time:.2f} ms ---")
            
            # Test generation
            main_logger.info("🤖▶️ --- Running Bedrock Generation ---")
            try:
                with BedrockGenerationContext(bedrock_llm, "What is the capital of France? Respond briefly.") as generator:
                    print("\nBedrock Response: ", end="", flush=True)
                    response_text = ""
                    for token in generator:
                        print(token, end="", flush=True)
                        response_text += token
                    print("\n")
                main_logger.info("🤖✅ Bedrock generation complete.")
            except Exception as e:
                main_logger.error(f"🤖💥 Bedrock Generation Error: {e}")
        else:
            main_logger.error("🤖❌ Bedrock Prewarm Failed. Skipping tests.")
    
    except Exception as e:
        main_logger.error(f"🤖💥 Failed to initialize or run Bedrock: {e}", exc_info=True)
    
    main_logger.info("🤖🏁 --- Bedrock LLM Example Script Finished ---")
