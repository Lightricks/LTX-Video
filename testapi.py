import os
import logging
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging based on .env configuration
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class LTXVideoAPIClient:
    def __init__(
        self, 
        endpoint: Optional[str] = None,
        timeout: int = 300  # Increased timeout for longer video generation
    ):
        """
        Initialize LTX Video API Client
        
        Args:
            endpoint (str, optional): API endpoint. Defaults to localhost.
            timeout (int, optional): Request timeout in seconds. Defaults to 300.
        """
        self.base_url = endpoint or 'http://localhost:8000'
        self.endpoint = f'{self.base_url}/generate/text-to-video'
        self.timeout = timeout
        
    def generate_video(
        self, 
        prompt: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt
        
        Args:
            prompt (str): Detailed text description for video generation
            params (dict, optional): Additional generation parameters
        
        Returns:
            Dict containing video generation result
        """
        # Default generation parameters with maximum frames
        default_params = {
            'guidance_scale': 3.5,
            'num_inference_steps': 40,
            'height': 720,
            'width': 1280,
            'num_frames': 257  # Maximum available frames
        }
        
        # Merge default and user-provided params
        generation_params = {**default_params, **(params or {})}
        
        payload = {
            'prompt': prompt,
            **generation_params
        }
        
        try:
            logger.info(f"Sending video generation request for prompt: {prompt}")
            logger.info(f"Generation parameters: {generation_params}")
            
            response = requests.post(
                self.endpoint, 
                json=payload, 
                timeout=self.timeout
            )
            
            response.raise_for_status()  # Raise exception for bad status codes
            
            logger.info("Video generation request successful")
            return response.json()
        
        except requests.RequestException as e:
            logger.error(f"Video generation failed: {e}")
            logger.error(f"Request details - Endpoint: {self.endpoint}, Payload: {payload}")
            raise

def main():
    """Example usage of LTXVideoAPIClient"""
    try:
        client = LTXVideoAPIClient()
        
        # Example prompt with added detail
        prompt = (
            "A woman with long brown hair walks through a sunlit forest. "
            "She moves gracefully between tall pine trees, her hand gently "
            "brushing against the bark. The camera follows her from behind, "
            "capturing her movement and the dappled light filtering through "
            "the canopy. Her blue jacket contrasts with the green forest, "
            "highly detailed, cinematic, smooth camera movement."
        )
        
        # Generate video with maximum frames
        result = client.generate_video(prompt)
        logger.info("Video generation successful!")
        logger.info(f"Video details: {result}")
    
    except Exception as e:
        logger.error(f"Unexpected error in video generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
