"""
Basic usage example for VideoLM
"""

from pathlib import Path
from videolm import VideoLM

def main():
    # Initialize the model
    # You can use different model sizes:
    # - "Qwen/Qwen2-VL-2B-Instruct" (smallest, fastest)
    # - "Qwen/Qwen2-VL-7B-Instruct" (balanced)
    # - "Qwen/Qwen2-VL-72B-Instruct" (largest, best quality)
    
    print("Initializing VideoLM model...")
    model = VideoLM(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        max_frames=8,  # Number of frames to extract from video
        frame_size=(448, 448)  # Frame resolution
    )
    
    # Example: Answer a question about a video
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    question = "What is happening in this video?"
    
    if Path(video_path).exists():
        print(f"\nProcessing video: {video_path}")
        print(f"Question: {question}")
        
        answer = model.answer_question(
            video_path=video_path,
            question=question,
            max_new_tokens=512,
            temperature=0.7
        )
        
        print(f"\nAnswer: {answer}")
    else:
        print(f"\nVideo file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")

if __name__ == "__main__":
    main()

