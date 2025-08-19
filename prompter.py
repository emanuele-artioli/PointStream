#!/usr/bin/env python3
"""
Local LLM Prompter Component

This component uses a local LLM with visual capabilities to generate descriptive prompts
for panorama images. It focuses on describing the background environment and setting
while avoiding mentions of people, animals, or transient objects.

The component is designed to work with locally hosted LLMs like LLaVA or similar models.
"""

import cv2
import numpy as np
import logging
import base64
import io
import torch
import os
from typing import Dict, Any, Optional, List
from PIL import Image
from decorators import log_step, time_step
import config

try:
    # Gemini API for vision-language tasks
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
    logging.info("Gemini API available for vision-language processing")
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini API not available, using fallback prompt generation")

try:
    # Alternative: OpenAI-compatible local API (like llama.cpp or Ollama)
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Check for transformers (legacy support)
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class Prompter:
    """Local LLM prompter for generating background descriptions."""
    
    def __init__(self):
        """Initialize the prompter with Gemini API configuration."""
        # Model configuration
        self.model_type = config.get_str('prompter', 'model_type', 'gemini')  # 'gemini', 'api', or 'fallback'
        self.device = config.get_str('prompter', 'device', 'auto')
        
        # Gemini configuration
        self.gemini_model = config.get_str('prompter', 'gemini_model', 'gemini-2.5-flash')
        self.api_key = config.get_str('prompter', 'gemini_api_key', '')
        
        # API configuration (for external API servers)
        self.api_url = config.get_str('prompter', 'api_url', 'http://localhost:11434/api/generate')
        self.api_model_name = config.get_str('prompter', 'api_model_name', 'llava')
        
        # Generation parameters
        self.max_new_tokens = config.get_int('prompter', 'max_new_tokens', 100)
        self.temperature = config.get_float('prompter', 'temperature', 0.7)
        self.do_sample = config.get_bool('prompter', 'do_sample', True)
        
        # Image preprocessing
        self.max_image_size = config.get_int('prompter', 'max_image_size', 512)
        self.image_quality = config.get_int('prompter', 'image_quality', 85)
        
        # System prompt template
        self.system_prompt = config.get_str('prompter', 'system_prompt', 
            "Describe the background of this image in detail. Focus on the environment, "
            "setting, and non-moving elements. Do not mention any people, animals, or "
            "transient objects in the foreground. Describe the scenery, architecture, "
            "landscape, weather, lighting, and atmosphere.")
        
        # Set device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() and TRANSFORMERS_AVAILABLE else 'cpu'
        
        # Initialize model
        self.model = None
        self.processor = None
        self._initialize_model()
        
        logging.info("Prompter initialized")
        logging.info(f"Model type: {self.model_type}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Max image size: {self.max_image_size}")
    
    def _initialize_model(self):
        """Initialize the local LLM model."""
        if self.model_type == 'gemini' and GEMINI_AVAILABLE:
            self._initialize_gemini_client()
        elif self.model_type == 'api' and REQUESTS_AVAILABLE:
            self._initialize_api_client()
        else:
            logging.info("Using fast fallback prompt generation")
            self.model_type = 'fallback'
    
    def _initialize_gemini_client(self):
        """Initialize Gemini API client."""
        try:
            if not self.api_key:
                logging.warning("No Gemini API key provided, using fallback")
                self.model_type = 'fallback'
                return
            
            # Configure Gemini client with proper API
            self.client = genai.Client(api_key=self.api_key)
            
            logging.info(f"Gemini client initialized with model: {self.gemini_model}")
            
        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}")
            self.model_type = 'fallback'
    
    def _initialize_transformers_model(self):
        """Initialize Transformers-based local model."""
        try:
            logging.info(f"Loading transformers model: {self.model_name}")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logging.info("Transformers model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load transformers model: {e}")
            self.model = None
            self.processor = None
            self.model_type = 'fallback'
    
    def _initialize_api_client(self):
        """Initialize API client for local LLM server."""
        try:
            # Test API connection
            test_response = requests.get(self.api_url.replace('/api/generate', '/api/tags'), timeout=5)
            if test_response.status_code == 200:
                logging.info(f"API client connected to: {self.api_url}")
            else:
                raise Exception(f"API test failed with status {test_response.status_code}")
                
        except Exception as e:
            logging.error(f"Failed to connect to API: {e}")
            self.model_type = 'fallback'
    
    @log_step
    @time_step(track_processing=True)
    def generate_prompt(self, panorama: np.ndarray) -> Dict[str, Any]:
        """
        Generate a descriptive prompt for the panorama image.
        
        Args:
            panorama: Panorama image as numpy array
            
        Returns:
            Dictionary containing:
            - prompt: Generated text prompt
            - method: Method used for generation
            - processing_time: Time taken for generation
        """
        if panorama is None:
            return {
                'prompt': 'natural outdoor scene with clear lighting',
                'method': 'fallback',
                'success': False,
                'error': 'no_panorama'
            }
        
        logging.info("Generating background description prompt")
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(panorama)
            
            # Generate prompt based on available method
            if self.model_type == 'gemini':
                result = self._generate_with_gemini(processed_image)
            elif self.model_type == 'api':
                result = self._generate_with_api(processed_image)
            else:
                result = self._generate_fallback(processed_image)
            
            # Post-process the generated text
            processed_prompt = self._postprocess_prompt(result['prompt'])
            
            return {
                'prompt': processed_prompt,
                'method': result['method'],
                'success': True,
                'original_prompt': result.get('prompt', ''),
                'image_size': processed_image.size if hasattr(processed_image, 'size') else panorama.shape[:2]
            }
            
        except Exception as e:
            logging.error(f"Prompt generation failed: {e}")
            return {
                'prompt': 'natural scene with ambient lighting',
                'method': 'error_fallback',
                'success': False,
                'error': str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> Image.Image:
        """
        Preprocess image for LLM input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL
        pil_image = Image.fromarray(image_rgb)
        
        # Resize if too large
        width, height = pil_image.size
        max_dim = max(width, height)
        
        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logging.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return pil_image
    
    def _generate_with_gemini(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate prompt using Gemini API.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Dictionary with generation result
        """
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=self.image_quality)
            image_bytes = img_byte_arr.getvalue()
            
            # Create Gemini request using new API format
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                    self.system_prompt + " Focus only on the permanent background elements and environment."
                ]
            )
            
            # Extract text from response
            prompt_text = response.text.strip() if response.text else ""
            
            # Clean up the prompt
            cleaned_prompt = self._clean_prompt(prompt_text)
            
            return {
                'prompt': cleaned_prompt,
                'method': 'gemini',
                'success': True,
                'raw_response': prompt_text
            }
            
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}")
            return self._generate_fallback(image)

    def _generate_with_transformers(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate prompt using Transformers model.
        
        Args:
            image: Preprocessed PIL image
            
        Returns:
            Generation result dictionary
        """
        try:
            # Prepare inputs
            inputs = self.processor(images=image, text=self.system_prompt, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if self.system_prompt in generated_text:
                generated_text = generated_text.replace(self.system_prompt, "").strip()
            
            return {
                'prompt': generated_text,
                'method': 'transformers'
            }
            
        except Exception as e:
            logging.error(f"Transformers generation failed: {e}")
            raise
    
    def _generate_with_api(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate prompt using local API.
        
        Args:
            image: Preprocessed PIL image
            
        Returns:
            Generation result dictionary
        """
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.image_quality)
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare API request
            payload = {
                "model": self.api_model_name,
                "prompt": self.system_prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens
                }
            }
            
            # Make API call
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            return {
                'prompt': generated_text,
                'method': 'api'
            }
            
        except Exception as e:
            logging.error(f"API generation failed: {e}")
            raise
    
    def _generate_fallback(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate fallback prompt using image analysis.
        
        Args:
            image: Preprocessed PIL image
            
        Returns:
            Fallback generation result
        """
        # Convert back to numpy for analysis
        image_np = np.array(image)
        
        # Analyze image characteristics
        analysis = self._analyze_image_characteristics(image_np)
        
        # Generate descriptive prompt based on analysis
        prompt_parts = []
        
        # Base description
        prompt_parts.append("A detailed background scene featuring")
        
        # Lighting analysis
        if analysis['brightness'] > 150:
            prompt_parts.append("bright, well-lit environment with")
        elif analysis['brightness'] < 80:
            prompt_parts.append("dimly lit, atmospheric setting with")
        else:
            prompt_parts.append("moderately lit scene with")
        
        # Color analysis
        dominant_color = analysis['dominant_color']
        if dominant_color == 'blue':
            prompt_parts.append("blue sky or water elements,")
        elif dominant_color == 'green':
            prompt_parts.append("lush vegetation and natural elements,")
        elif dominant_color == 'brown':
            prompt_parts.append("earthy tones and natural textures,")
        elif dominant_color == 'gray':
            prompt_parts.append("urban or architectural elements,")
        else:
            prompt_parts.append("varied natural colors,")
        
        # Texture analysis
        if analysis['edge_density'] > 0.1:
            prompt_parts.append("rich architectural details and structured elements")
        elif analysis['edge_density'] > 0.05:
            prompt_parts.append("moderate structural complexity")
        else:
            prompt_parts.append("smooth, flowing natural forms")
        
        # Composition
        prompt_parts.append("with natural lighting and atmospheric depth")
        
        generated_prompt = " ".join(prompt_parts)
        
        return {
            'prompt': generated_prompt,
            'method': 'fallback_analysis'
        }
    
    def _analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze basic image characteristics for fallback generation.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with image analysis results
        """
        # Convert to grayscale for some analyses
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Brightness analysis
        brightness = np.mean(gray)
        
        # Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Color analysis
        dominant_color = 'neutral'
        if len(image.shape) == 3:
            # Calculate average color
            avg_color = np.mean(image, axis=(0, 1))
            
            # Determine dominant color
            if avg_color[2] > avg_color[1] and avg_color[2] > avg_color[0]:  # More blue
                dominant_color = 'blue'
            elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:  # More green
                dominant_color = 'green'
            elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:  # More red
                dominant_color = 'red'
            elif np.std(avg_color) < 20:  # Low variance = gray
                dominant_color = 'gray'
            else:
                dominant_color = 'brown'
        
        return {
            'brightness': brightness,
            'edge_density': edge_density,
            'dominant_color': dominant_color
        }
    
    def _postprocess_prompt(self, prompt: str) -> str:
        """
        Post-process the generated prompt to ensure quality.
        
        Args:
            prompt: Raw generated prompt
            
        Returns:
            Cleaned and processed prompt
        """
        if not prompt or len(prompt.strip()) < 10:
            return "natural outdoor scene with clear lighting and environmental details"
        
        # Clean up the prompt
        cleaned = prompt.strip()
        
        # Remove mentions of people, animals, or transient objects
        forbidden_words = [
            'person', 'people', 'man', 'woman', 'child', 'human', 'individual',
            'dog', 'cat', 'bird', 'animal', 'pet', 'wildlife',
            'car', 'vehicle', 'bike', 'motorcycle', 'truck',
            'walking', 'running', 'standing', 'sitting'
        ]
        
        words = cleaned.lower().split()
        filtered_words = []
        
        for word in words:
            # Remove forbidden words and their variations
            word_clean = word.strip('.,!?;:')
            if not any(forbidden in word_clean for forbidden in forbidden_words):
                filtered_words.append(word)
        
        filtered_prompt = ' '.join(filtered_words)
        
        # Ensure minimum length
        if len(filtered_prompt) < 20:
            filtered_prompt += ", featuring natural lighting and atmospheric depth"
        
        # Ensure it ends properly
        if not filtered_prompt.endswith(('.', '!', ',')):
            filtered_prompt += "."
        
        return filtered_prompt
    
    def batch_generate_prompts(self, panoramas: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Generate prompts for multiple panoramas.
        
        Args:
            panoramas: List of panorama images
            
        Returns:
            List of prompt generation results
        """
        results = []
        
        for i, panorama in enumerate(panoramas):
            logging.info(f"Generating prompt for panorama {i+1}/{len(panoramas)}")
            result = self.generate_prompt(panorama)
            results.append(result)
        
        return results
    
    def _clean_prompt(self, prompt: str) -> str:
        """
        Clean and post-process generated prompt text.
        
        Args:
            prompt: Raw generated prompt
            
        Returns:
            Cleaned prompt text
        """
        if not prompt:
            return "natural outdoor scene with clear lighting"
        
        # Remove common unwanted phrases
        unwanted_phrases = [
            "in this image", "the image shows", "this photo", "the photo",
            "i can see", "there are", "there is", "the scene contains"
        ]
        
        cleaned = prompt.lower()
        for phrase in unwanted_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        # Remove mentions of people/animals
        people_terms = ["person", "people", "man", "woman", "child", "human", "figure"]
        animal_terms = ["animal", "dog", "cat", "bird", "horse", "cow"]
        
        for term in people_terms + animal_terms:
            cleaned = cleaned.replace(term, "")
        
        # Clean up extra spaces and punctuation
        cleaned = " ".join(cleaned.split())
        cleaned = cleaned.strip(" .,;:")
        
        # Ensure it starts with a capital letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        # Fallback if too short after cleaning
        if len(cleaned) < 10:
            return "natural outdoor scene with ambient lighting"
        
        return cleaned
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate the quality of a generated prompt.
        
        Args:
            prompt: Generated prompt to validate
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check length
        if len(prompt) < 10:
            issues.append("too_short")
        elif len(prompt) > 200:
            issues.append("too_long")
        
        # Check for forbidden content
        forbidden_words = ['person', 'people', 'man', 'woman', 'animal', 'car', 'vehicle']
        for word in forbidden_words:
            if word in prompt.lower():
                issues.append(f"contains_forbidden_word_{word}")
        
        # Check for coherence
        if not any(keyword in prompt.lower() for keyword in ['scene', 'environment', 'setting', 'lighting', 'background']):
            issues.append("lacks_scene_description")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'score': max(0, 1.0 - len(issues) * 0.2)  # Score decreases with issues
        }