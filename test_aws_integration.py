#!/usr/bin/env python3
"""
Test script for AWS SageMaker Whisper and AWS Bedrock Claude Haiku integration.

This script tests both components independently to verify they're working correctly.
"""

import os
import sys
import logging
import asyncio
import numpy as np
import base64
import io
import wave
from typing import Optional

# Add the code directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Loaded environment variables from .env file")
except ImportError:
    logger.warning("⚠️ dotenv not available, using system environment variables")

def create_test_audio() -> np.ndarray:
    """Create a simple test audio signal (sine wave saying 'hello')."""
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a simple sine wave
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some envelope to make it more speech-like
    envelope = np.exp(-t * 2) * (1 - np.exp(-t * 5))
    audio = audio * envelope
    
    # Convert to int16
    audio = (audio * 32767).astype(np.int16)
    
    logger.info(f"✅ Created test audio: {len(audio)} samples, {duration}s duration")
    return audio

def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio numpy array to WAV bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    buffer.seek(0)
    return buffer.read()

def test_aws_whisper():
    """Test AWS SageMaker Whisper endpoint."""
    logger.info("🔊 Testing AWS SageMaker Whisper...")
    
    try:
        import boto3
        import json
        
        # Get configuration from environment
        endpoint_name = os.getenv("AWS_WHISPER_ENDPOINT")
        profile_name = os.getenv("AWS_PROFILE_NAME")
        region_name = os.getenv("AWS_REGION", "us-east-1")
        
        if not endpoint_name:
            logger.error("❌ AWS_WHISPER_ENDPOINT not set in environment")
            return False
        
        if not profile_name:
            logger.error("❌ AWS_PROFILE_NAME not set in environment")
            return False
        
        logger.info(f"🔊 Using endpoint: {endpoint_name}")
        logger.info(f"🔊 Using profile: {profile_name}")
        logger.info(f"🔊 Using region: {region_name}")
        
        # Create AWS client (using your working pattern)
        session = boto3.Session(profile_name=profile_name)
        runtime_client = session.client('runtime.sagemaker', region_name=region_name)
        
        # Create test audio
        test_audio = create_test_audio()
        
        # Convert to WAV bytes
        wav_bytes = audio_to_wav_bytes(test_audio, 16000)
        
        logger.info(f"🔊 Sending {len(wav_bytes)} bytes to endpoint...")
        
        # Call the endpoint directly (following your working pattern)
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="audio/wav",
            Body=wav_bytes
        )
        
        result = json.loads(response['Body'].read())
        logger.info(f"🔊 Raw response: {result}")
        
        # Handle both string and list responses from SageMaker
        text_raw = result.get('text', '')
        if isinstance(text_raw, list):
            # If text is a list, join the items
            transcription = ' '.join(str(item) for item in text_raw).strip()
        else:
            # If text is a string, use it directly
            transcription = str(text_raw).strip()
        
        if transcription:
            logger.info(f"🔊✅ Whisper transcription result: '{transcription}'")
            logger.info("🔊✅ AWS SageMaker Whisper test PASSED")
            return True
        else:
            logger.error("🔊❌ No transcription received from Whisper endpoint")
            return False
            
    except Exception as e:
        logger.error(f"🔊❌ AWS SageMaker Whisper test error: {e}")
        import traceback
        logger.error(f"🔊❌ Full traceback: {traceback.format_exc()}")
        return False

def test_aws_bedrock():
    """Test AWS Bedrock Claude Haiku."""
    logger.info("🤖 Testing AWS Bedrock Claude Haiku...")
    
    try:
        from bedrock_llm import BedrockLLM
        
        # Get configuration from environment
        model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-haiku-20240307-v1:0")
        profile_name = os.getenv("AWS_PROFILE_NAME")
        region_name = os.getenv("AWS_REGION", "us-east-1")
        
        if not profile_name:
            logger.error("❌ AWS_PROFILE_NAME not set in environment")
            return False
        
        logger.info(f"🤖 Using model: {model_id}")
        logger.info(f"🤖 Using profile: {profile_name}")
        logger.info(f"🤖 Using region: {region_name}")
        
        # Create client
        bedrock_llm = BedrockLLM(
            model_id=model_id,
            profile_name=profile_name,
            region_name=region_name,
            system_prompt="You are a helpful assistant. Be concise and accurate."
        )
        
        # Test prewarm
        logger.info("🤖🔥 Testing Bedrock prewarm...")
        prewarm_success = bedrock_llm.prewarm(max_retries=1)
        
        if not prewarm_success:
            logger.error("🤖❌ Bedrock prewarm failed")
            return False
        
        logger.info("🤖✅ Bedrock prewarm successful")
        
        # Test generation
        logger.info("🤖💬 Testing Bedrock generation...")
        test_prompt = "What is the capital of France? Answer briefly."
        
        try:
            response_text = ""
            token_count = 0
            
            for token in bedrock_llm.generate(
                text=test_prompt,
                history=None,
                use_system_prompt=True,
                temperature=0.1
            ):
                response_text += token
                token_count += 1
                if token_count == 1:
                    logger.info("🤖⚡ Received first token from Bedrock")
            
            if response_text:
                logger.info(f"🤖✅ Bedrock response ({token_count} tokens): '{response_text.strip()}'")
                logger.info("🤖✅ AWS Bedrock Claude Haiku test PASSED")
                return True
            else:
                logger.error("🤖❌ No response received from Bedrock")
                return False
                
        except Exception as e:
            logger.error(f"🤖❌ Bedrock generation error: {e}")
            import traceback
            logger.error(f"🤖❌ Full traceback: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        logger.error(f"🤖❌ AWS Bedrock test error: {e}")
        import traceback
        logger.error(f"🤖❌ Full traceback: {traceback.format_exc()}")
        return False

def test_llm_module_bedrock():
    """Test the LLM module with Bedrock backend."""
    logger.info("🧠 Testing LLM module with Bedrock backend...")
    
    try:
        from llm_module import LLM
        
        # Get configuration from environment
        model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-haiku-20240307-v1:0")
        
        logger.info(f"🧠 Testing LLM module with model: {model_id}")
        
        # Create LLM instance with Bedrock backend
        llm = LLM(
            backend="bedrock",
            model=model_id,
            system_prompt="You are a helpful assistant. Be concise."
        )
        
        # Test prewarm
        logger.info("🧠🔥 Testing LLM module prewarm...")
        prewarm_success = llm.prewarm(max_retries=1)
        
        if not prewarm_success:
            logger.error("🧠❌ LLM module prewarm failed")
            return False
        
        logger.info("🧠✅ LLM module prewarm successful")
        
        # Test generation
        logger.info("🧠💬 Testing LLM module generation...")
        test_prompt = "List three colors. Be brief."
        
        try:
            response_text = ""
            token_count = 0
            
            for token in llm.generate(
                text=test_prompt,
                history=None,
                use_system_prompt=True,
                temperature=0.1
            ):
                response_text += token
                token_count += 1
                if token_count == 1:
                    logger.info("🧠⚡ Received first token from LLM module")
            
            if response_text:
                logger.info(f"🧠✅ LLM module response ({token_count} tokens): '{response_text.strip()}'")
                logger.info("🧠✅ LLM module with Bedrock test PASSED")
                return True
            else:
                logger.error("🧠❌ No response received from LLM module")
                return False
                
        except Exception as e:
            logger.error(f"🧠❌ LLM module generation error: {e}")
            import traceback
            logger.error(f"🧠❌ Full traceback: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        logger.error(f"🧠❌ LLM module test error: {e}")
        import traceback
        logger.error(f"🧠❌ Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting AWS integration tests...")
    logger.info("=" * 60)
    
    # Check required environment variables
    required_env_vars = ["AWS_PROFILE_NAME", "AWS_REGION"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.error("❌ Please set up your .env file based on .env.example")
        return False
    
    # Show configuration
    logger.info(f"🔧 AWS Profile: {os.getenv('AWS_PROFILE_NAME')}")
    logger.info(f"🔧 AWS Region: {os.getenv('AWS_REGION')}")
    logger.info(f"🔧 Whisper Endpoint: {os.getenv('AWS_WHISPER_ENDPOINT', 'Not set')}")
    logger.info(f"🔧 Bedrock Model: {os.getenv('BEDROCK_MODEL_ID', 'Default')}")
    logger.info("=" * 60)
    
    # Run tests
    results = []
    
    # Test 1: AWS Whisper (if enabled)
    use_aws_whisper = os.getenv("USE_AWS_WHISPER", "false").lower() == "true"
    if use_aws_whisper:
        results.append(("AWS SageMaker Whisper", test_aws_whisper()))
    else:
        logger.info("🔊⚠️ Skipping AWS Whisper test (USE_AWS_WHISPER=false)")
    
    # Test 2: AWS Bedrock
    results.append(("AWS Bedrock Claude Haiku", test_aws_bedrock()))
    
    # Test 3: LLM Module with Bedrock
    results.append(("LLM Module with Bedrock", test_llm_module_bedrock()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! AWS integration is working correctly.")
        return True
    else:
        logger.error("💥 SOME TESTS FAILED! Please check the configuration and logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
