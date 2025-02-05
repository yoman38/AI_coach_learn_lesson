import asyncio
import websockets
import json
import logging
from google.cloud import speech
import os
from queue import SimpleQueue

# Set environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\MesProjets\3-learningbyfeedback\API\ggle_cred.json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Google Speech Client
client = speech.SpeechClient()

async def transcribe(websocket):
    logger.info("Client connected")

    # Configure the recognition settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # Matches the client sampling rate
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    # Queue to hold audio chunks
    audio_queue = SimpleQueue()

    async def receive_audio():
        """
        Receive audio data from WebSocket and place it in the queue.
        """
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    audio_queue.put(message)
                else:
                    logger.warning("Non-binary message received")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
        finally:
            # Indicate the end of the audio stream
            audio_queue.put(None)

    def generate_requests():
        """
        Generate streaming requests for the Speech-to-Text API.
        """
        # Send the streaming config as the first request
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        while True:
            audio_chunk = audio_queue.get()
            if audio_chunk is None:  # End of stream
                break
            yield speech.StreamingRecognizeRequest(audio_content=audio_chunk)

    # Task to receive audio asynchronously
    receive_audio_task = asyncio.create_task(receive_audio())

    try:
        # Use the Speech-to-Text API's streaming method
        responses = client.streaming_recognize(requests=generate_requests())

        # Process responses asynchronously
        for response in responses:
            for result in response.results:
                transcript = result.alternatives[0].transcript
                is_final = result.is_final
                logger.info(f"Transcript: {transcript} (Final: {is_final})")
                # Send the transcription back to the client
                await websocket.send(json.dumps({
                    "transcript": transcript,
                    "is_final": is_final,
                }))
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
    finally:
        await receive_audio_task  # Ensure receiving audio is properly cleaned up

async def main():
    """
    Start the WebSocket server to handle multiple clients.
    """
    async with websockets.serve(transcribe, "0.0.0.0", 8000):
        logger.info("WebSocket server started on ws://0.0.0.0:8000")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down.")
