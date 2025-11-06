"""
Batch processing example for VideoLM
"""

from pathlib import Path
from videolm import VideoLM
from tqdm import tqdm

def main():
    # Initialize model
    print("Initializing VideoLM model...")
    model = VideoLM(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        max_frames=8,
        frame_size=(448, 448)
    )
    
    # Example: Process multiple videos with questions
    video_qa_pairs = [
        ("sample_video/scene0000_00.mp4", "What is the main activity in this video?"),
        ("../sample_video/scene0001_00.mp4", "How many people are in the video?"),
        # ("path/to/video3.mp4", "What objects can you see?"),
    ]
    
    # Filter out non-existent files
    valid_pairs = [
        (v, q) for v, q in video_qa_pairs 
        if Path(v).exists()
    ]
    
    if not valid_pairs:
        print("No valid video files found. Please update the video paths.")
        return
    
    video_paths = [v for v, _ in valid_pairs]
    questions = [q for _, q in valid_pairs]
    
    print(f"\nProcessing {len(video_paths)} videos...")
    
    # Batch process
    answers = model.batch_answer(
        video_paths=video_paths,
        questions=questions,
        max_new_tokens=512,
        temperature=0.7
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for video_path, question, answer in zip(video_paths, questions, answers):
        print(f"\nVideo: {Path(video_path).name}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 60)

if __name__ == "__main__":
    main()

