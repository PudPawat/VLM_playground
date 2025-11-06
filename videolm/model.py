"""
VideoLM Model: Qwen2-VL based Video Question Answering
"""

import torch
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image
try:
    from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
except ImportError:
    raise ImportError(
        "Qwen2-VL models require transformers>=4.37.0. "
        "Please install with: pip install transformers>=4.37.0"
    )

from .video_processor import VideoProcessor


class VideoLM:
    """Qwen2-VL based Video Language Model for Question Answering"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
        max_frames: int = 8,
        frame_size: tuple = (448, 448),
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ):
        """
        Initialize VideoLM model
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_frames: Maximum number of frames to extract from video
            frame_size: Target size for frames (width, height)
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_frames = max_frames
        self.frame_size = frame_size
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            max_frames=max_frames,
            frame_size=frame_size
        )
        
        # Load model and processor
        print(f"Loading model {model_name} on {self.device}...")
        self.processor = Qwen2VLProcessor.from_pretrained(model_name)
        
        # Model loading kwargs
        model_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            **model_kwargs
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def answer_question(
        self,
        video_path: Union[str, Path],
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Answer a question about a video
        
        Args:
            video_path: Path to video file
            question: Question to ask about the video
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Answer string
        """
        # Process video
        images = self.video_processor.process_video(video_path)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Prepare inputs using the processor
        # The processor handles both text and images from messages
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Extract images from messages for processing
        image_list = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        
        inputs = self.processor(
            text=[text],
            images=image_list if image_list else None,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        
        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return response_text[0].strip()
    
    def answer_question_from_images(
        self,
        images: List[Image.Image],
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Answer a question about a list of images (video frames)
        
        Args:
            images: List of PIL Images
            question: Question to ask about the images
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Answer string
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Prepare inputs using the processor
        # The processor handles both text and images from messages
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Extract images from messages for processing
        image_list = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        
        inputs = self.processor(
            text=[text],
            images=image_list if image_list else None,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        
        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return response_text[0].strip()
    
    def batch_answer(
        self,
        video_paths: List[Union[str, Path]],
        questions: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Answer multiple questions about multiple videos
        
        Args:
            video_paths: List of video file paths
            questions: List of questions (same length as video_paths)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of answer strings
        """
        if len(video_paths) != len(questions):
            raise ValueError("video_paths and questions must have the same length")
        
        answers = []
        for video_path, question in zip(video_paths, questions):
            answer = self.answer_question(
                video_path=video_path,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            answers.append(answer)
        
        return answers

